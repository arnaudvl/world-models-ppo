import torch
import torch.nn as nn
from typing import Callable, Union
from vae.vae import VAE
from mdnrnn.mdnrnn import MDNRNN
from utils.vars import CHANNELS, LATENT_SIZE, HIDDEN_SIZE, N_GAUSS


class WorldModel(nn.Module):
    def __init__(self,
                 action_dim: int,
                 output_dim: int,
                 vae_path: str = './vae/model/best.tar',
                 mdnrnn_path: str = './mdnrnn/model/best.tar',
                 activation: Callable = torch.relu,
                 output_activation: Union[Callable, str] = None,
                 output_squeeze: bool = False) -> None:
        super(WorldModel, self).__init__()

        # define VAE model
        self.latent_size = LATENT_SIZE
        self.vae = VAE(CHANNELS, LATENT_SIZE)
        vae_state = torch.load(vae_path)
        self.vae.load_state_dict(vae_state['state_dict'])
        for param in self.vae.parameters():
            param.requires_grad = False

        # define MDNRNN model
        self.n_gauss = N_GAUSS
        self.mdnrnn = MDNRNN(LATENT_SIZE, action_dim, HIDDEN_SIZE, N_GAUSS, rewards_terminal=False)
        mdnrnn_state = torch.load(mdnrnn_path)
        self.mdnrnn.load_state_dict(mdnrnn_state['state_dict'])
        for param in self.mdnrnn.parameters():
            param.requires_grad = False

        # controller
        ctr_size = LATENT_SIZE + N_GAUSS + 2 * (LATENT_SIZE * N_GAUSS)
        self.fc1 = nn.Linear(ctr_size, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # VAE
        _, mu, logsigma = self.vae(x)
        latent = (mu + logsigma.exp() * torch.randn_like(mu)).view(-1, self.latent_size)

        # MDNRNN
        mus, sigmas, logpi = self.mdnrnn(a.unsqueeze(0), latent.unsqueeze(0))

        # reshape
        mus = torch.squeeze(mus, dim=0).view(-1, self.n_gauss * self.latent_size)
        sigmas = torch.squeeze(sigmas, dim=0).view(-1, self.n_gauss * self.latent_size)
        logpi = torch.squeeze(logpi, dim=0).view(-1, self.n_gauss)

        # controller
        ctr_in = [latent, mus, sigmas, logpi]
        x = torch.cat(ctr_in, dim=1)
        x = self.activation(self.fc1(x))
        if self.output_activation is None:
            x = self.fc2(x)
        else:
            x = self.output_activation(self.fc2(x))
        return x.squeeze() if self.output_squeeze else x
