import torch
from torch.nn import functional as F


def loss_vae(x_recon, x, mu, logsigma):
    """ VAE loss function.  Reconstruction + KL divergence losses summed over all elements and batch. """
    # MSE reconstruction loss
    BCE = F.mse_loss(x_recon, x, size_average=False)

    # KL divergence
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD
