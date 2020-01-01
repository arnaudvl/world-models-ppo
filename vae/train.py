import argparse
import numpy as np
import os
import torch
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from vae import VAE
from loaders import GymDataset, collate_fn, generate_obs, save_checkpoint
from losses import loss_vae


# set variables
H, W = 64, 64
BATCH_SIZE = 64
CHANNELS = 3
LATENT_SIZE = 32
LR = 1e-3
GRAD_ACCUMULATION_STEPS = 1


def run(data_dir: str = './env/data',
        model_dir: str = './vae/model',
        epochs: int = 20
        ) -> None:
    """
    Train VAE using saved environment rollouts.

    Parameters
    ----------
    data_dir
        Directory with train and test data.
    model_dir
        Directory to optionally load model from and save trained model to.
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
        transforms.RandomHorizontalFlip(p=.5),
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
    dataset_train = GymDataset(dir_train, transform=transform_train)
    dataset_test = GymDataset(dir_test, transform=transform_test)
    dataset_test.load_batch(0)  # 1 batch of data used for test set
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  collate_fn=collate_fn)

    # set save and optional load directories
    load_file = os.path.join(model_dir, 'best.tar')
    try:
        state = torch.load(load_file)
    except FileNotFoundError:
        state = None

    # define and load model
    model = VAE(CHANNELS, LATENT_SIZE)
    if state is not None:
        model.load_state_dict(state['state_dict'])
    model.zero_grad()
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-6, weight_decay=0)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])

    # learning rate scheduling
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    if state is not None:
        lr_scheduler.load_state_dict(state['scheduler'])

    # define test fn
    def test():
        """ One test epoch. """
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for obs, _, _ in generate_obs(dataloader_test):
                obs = torch.stack(obs).to(device)
                obs_recon, mu, logsigma = model(obs)
                test_loss += loss_vae(obs_recon, obs, mu, logsigma).item()
        test_loss /= len(dataloader_test.dataset)
        return test_loss

    # train
    n_batch_train = len(dataset_train.batch_list)
    optimizer.zero_grad()

    cur_best = None
    loss_list = []

    for epoch in range(epochs):

        model.train()
        loss_train = 0
        n_batch = 0

        for i in range(n_batch_train):  # loop over training data for each epoch

            dataset_train.load_batch(i)
            dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True,
                                                           collate_fn=collate_fn)

            for j, (obs, _, _) in enumerate(generate_obs(dataloader_train)):

                n_batch += 1

                # place on device
                obs = torch.stack(obs).to(device)

                # forward pass
                obs_recon, mu, logsigma = model(obs)

                # eval loss fn
                loss = loss_vae(obs_recon, obs, mu, logsigma)

                # backward pass
                loss.backward()

                # store loss value
                loss_train += loss.item()
                loss_train_avg = loss_train / (n_batch * BATCH_SIZE)

                # apply gradients and learning rate scheduling with optional gradient accumulation
                if (j + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            loss_list.append(loss_train_avg)

        # learning rate scheduling
        lr_scheduler.step()

        # evaluate on test set
        loss_test_avg = test()

        # checkpointing
        best_filename = os.path.join(model_dir, 'best.tar')
        filename = os.path.join(model_dir, 'checkpoint.tar')
        is_best = not cur_best or loss_test_avg < cur_best
        if is_best:
            cur_best = loss_test_avg

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': loss_test_avg,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }, is_best, filename, best_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE on saved Carracing data.")
    parser.add_argument('--data_dir', type=str, default='./env/data')
    parser.add_argument('--model_dir', type=str, default='./vae/model')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.epochs)
