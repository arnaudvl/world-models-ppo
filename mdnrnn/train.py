import argparse
import numpy as np
import os
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm_notebook
from vae import VAE
from mdnrnn import MDNRNN, gmm_loss
from utils.loaders import GymDataset, collate_fn, generate_obs, save_checkpoint
from utils.vars import (H, W, BATCH_SIZE, CHANNELS, LATENT_SIZE, SEQ_LEN,
                        ACTION_SIZE, HIDDEN_SIZE, N_GAUSS, LR, GRAD_ACCUMULATION_STEPS)


def run(data_dir: str = './env/data',
        vae_dir: str = './vae/model',
        mdnrnn_dir: str = './mdnrnn/model',
        epochs: int = 20
        ) -> None:
    """
    Train mdnrnn using saved environment rollouts.

    Parameters
    ----------
    data_dir
        Directory with train and test data.
    vae_dir
        Directory to load VAE model from.
    mdnrnn_dir
        Directory to optionally load MDNRNN model from and save trained model to.
    epochs
        Number of training epochs.
    """
    # set random seed and deterministic backend
    SEED = 123
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # use GPU if available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # define input transformations
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    # define train and test datasets
    dir_train = os.path.join(data_dir, 'train/')
    dir_test = os.path.join(data_dir, 'test/')
    dataset_train = GymDataset(dir_train, seq_len=SEQ_LEN, transform=transform_train)
    dataset_test = GymDataset(dir_test, seq_len=SEQ_LEN, transform=transform_test)
    dataset_test.load_batch(0)  # 1 batch of data used for test set
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  collate_fn=collate_fn)

    # define and load VAE model
    vae = VAE(CHANNELS, LATENT_SIZE)
    load_vae_file = os.path.join(vae_dir, 'best.tar')
    state_vae = torch.load(load_vae_file)
    vae.load_state_dict(state_vae['state_dict'])
    vae.to(device)

    # set save and optional load directories for the MDNRNN model
    load_mdnrnn_file = os.path.join(mdnrnn_dir, 'best.tar')
    try:
        state_mdnrnn = torch.load(load_mdnrnn_file)
    except FileNotFoundError:
        state_mdnrnn = None

    # define and load MDNRNN model
    mdnrnn = MDNRNN(LATENT_SIZE, ACTION_SIZE, HIDDEN_SIZE, N_GAUSS, rewards_terminal=False)
    if state_mdnrnn is not None:
        mdnrnn.load_state_dict(state_mdnrnn['state_dict'])
    mdnrnn.zero_grad()
    mdnrnn.to(device)

    # optimizer
    params = [p for p in mdnrnn.parameters() if p.requires_grad]
    optimizer = RMSprop(params, lr=LR, alpha=.9)
    if state_mdnrnn is not None:
        optimizer.load_state_dict(state_mdnrnn['optimizer'])

    # learning rate scheduling
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    if state_mdnrnn is not None:
        lr_scheduler.load_state_dict(state_mdnrnn['scheduler'])

    # helper function
    def img2latent(obs, batch_size):
        """ Function to go from image to latent space. """
        with torch.no_grad():
            obs = obs.view(-1, CHANNELS, H, W)
            _, mu, logsigma = vae(obs)
            latent = (mu + logsigma.exp() * torch.randn_like(mu)).view(batch_size, SEQ_LEN, LATENT_SIZE)
        return latent

    # define test fn
    def test():
        """ One test epoch """
        mdnrnn.eval()
        test_loss = 0
        n_test = len(dataloader_test.dataset)
        with torch.no_grad():
            for (obs, action, next_obs) in generate_obs(dataloader_test):

                batch_size = len(obs)

                # place on device
                try:
                    obs = torch.stack(obs).to(device)
                    next_obs = torch.stack(next_obs).to(device)
                    action = torch.stack(action).to(device)
                except:
                    print('Did not manage to stack test observations and actions.')
                    n_test -= batch_size
                    continue

                # convert to latent space
                latent_obs = img2latent(obs, batch_size)
                next_latent_obs = img2latent(next_obs, batch_size)

                # need to flip dims to feed into LSTM from [batch, seq_len, dim] to [seq_len, batch, dim]
                latent_obs, action, next_latent_obs = [arr.transpose(1, 0)
                                                       for arr in [latent_obs, action, next_latent_obs]]

                # forward pass model
                mus, sigmas, logpi = mdnrnn(action, latent_obs)

                # compute loss
                loss = gmm_loss(next_latent_obs, mus, sigmas, logpi)
                test_loss += loss.item()

        test_loss /= n_test
        return test_loss

    # train
    n_batch_train = len(dataset_train.batch_list)
    optimizer.zero_grad()

    cur_best = None

    tq_episode = tqdm_notebook(range(epochs))
    for epoch in tq_episode:

        mdnrnn.train()
        loss_train = 0
        n_batch = 0

        tq_batch = tqdm_notebook(range(n_batch_train))
        for i in tq_batch:  # loop over training data for each epoch

            dataset_train.load_batch(i)
            dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True,
                                                           collate_fn=collate_fn)

            tq_minibatch = tqdm_notebook(generate_obs(dataloader_train), total=len(dataloader_train), leave=False)
            for j, (obs, action, next_obs) in enumerate(tq_minibatch):

                n_batch += 1

                # place on device
                batch_size = len(obs)
                try:
                    obs = torch.stack(obs).to(device)
                    next_obs = torch.stack(next_obs).to(device)
                    action = torch.stack(action).to(device)
                except:
                    print('Did not manage to stack observations and actions.')
                    continue

                # convert to latent space
                latent_obs = img2latent(obs, batch_size)
                next_latent_obs = img2latent(next_obs, batch_size)

                # need to flip dims to feed into LSTM from [batch, seq_len, dim] to [seq_len, batch, dim]
                latent_obs, action, next_latent_obs = [arr.transpose(1, 0)
                                                       for arr in [latent_obs, action, next_latent_obs]]

                # forward pass model
                mus, sigmas, logpi = mdnrnn(action, latent_obs)

                # compute loss
                loss = gmm_loss(next_latent_obs, mus, sigmas, logpi)

                # backward pass
                loss.backward()

                # store loss value
                loss_train += loss.item()
                loss_train_avg = loss_train / (n_batch * BATCH_SIZE)

                # apply gradients and learning rate scheduling with optional gradient accumulation
                if (j + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                tq_minibatch.set_postfix(loss_train=loss_train_avg)

            tq_batch.set_postfix(loss_train=loss_train_avg)

        lr_scheduler.step()

        # evaluate on test set
        loss_test_avg = test()

        # checkpointing
        best_filename = os.path.join(mdnrnn_dir, 'best.tar')
        filename = os.path.join(mdnrnn_dir, 'checkpoint.tar')
        is_best = not cur_best or loss_test_avg < cur_best
        if is_best:
            cur_best = loss_test_avg

        save_checkpoint({
            'epoch': epoch,
            'state_dict': mdnrnn.state_dict(),
            'precision': loss_test_avg,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }, is_best, filename, best_filename)

        tq_episode.set_postfix(loss_train=loss_train_avg, loss_test=loss_test_avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MDNRNN on saved Carracing data.")
    parser.add_argument('--data_dir', type=str, default='./env/data')
    parser.add_argument('--vae_dir', type=str, default='./vae/model')
    parser.add_argument('--mdnrnn_dir', type=str, default='./mdnrnn/model')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    run(args.data_dir, args.vae_dir, args.mdnrnn_dir, args.epochs)
