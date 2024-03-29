import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def loss_vae(x_recon, x, mu, logsigma):
    """ VAE loss function.  Reconstruction + KL divergence losses summed over all elements and batch. """
    # MSE reconstruction loss
    BCE = F.mse_loss(x_recon, x, size_average=False)

    # KL divergence
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels: int, latent_size: int) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels: int, latent_size: int) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Parameters
    ----------
    img_channels
        Number of channels in the image.
    latent_size
        Latent dimension.
    """
    def __init__(self, img_channels: int, latent_size: int) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        x_recon = self.decoder(z)
        return x_recon, mu, logsigma
