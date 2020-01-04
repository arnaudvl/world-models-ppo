import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import Tuple


def gmm_loss(batch: torch.Tensor,
             mus: torch.Tensor,
             sigmas: torch.Tensor,
             logpi: torch.Tensor,
             reduce: bool = True) -> torch.Tensor:
    """
    Computes the GMM loss. The loss is the negative log probability of the batch
    of instances under the GMM model described by `mus`, `sigmas`, `logpi`.
    It handles multiple batch dimensions bs1, bs2, ... which is useful in the
    presence of both a batch and time step axis. The number of mixtures is gs and
    the number of features equals fs.

    Parameters
    ----------
    batch: (bs1, bs2, *, fs)
        Batch of instances.
    mus: (bs1, bs2, *, gs, fs)
        GMM means.
    sigmas: (bs1, bs2, *, gs, fs)
        GMM sigmas.
    logpi: (bs1, bs2, *, gs)
        Log of GMM weights.
    reduce
        Take mean of log probabilities if True.

    Returns
    -------
    GMM loss:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    The loss is not reduced along the feature dimension.
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob


class MDNRNN(nn.Module):
    def __init__(self,
                 latents: int,
                 actions: int,
                 hiddens: int,
                 gaussians: int,
                 rewards_terminal: bool = False
                 ) -> None:
        """
        Recurrent Mixture Density Network.

        Parameters
        ----------
        latents
            Latent dimension.
        actions
            Action space.
        hiddens
            Hidden dimension.
        gaussians
            Number of Gaussians for the MDN.
        rewards_terminal
            True if rewards and terminal state from environment used.
        """
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.rewards_terminal = rewards_terminal
        if rewards_terminal:
            gmm_out = (2 * latents + 1) * gaussians + 2
        else:
            gmm_out = (2 * latents + 1) * gaussians
        self.gmm_linear = nn.Linear(hiddens, gmm_out)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions: torch.Tensor, latents: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        actions: (sequence length, batch size, action dim)
        latents: (sequence length, batch size, latent dim)

        Returns
        -------
        Parameters of the MDN and optionally predictions for the reward and state terminality.
        mus: (sequence length, batch size, nb Gaussians, latent dim)
        sigmas: (sequence length, batch size, nb Gaussians, latent dim)
        logpi: (sequence length, batch size, nb Gaussians)
        rs: (sequence length, batch size)
        ds: (sequence length, batch size)
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        if self.rewards_terminal:
            rs = gmm_outs[:, :, -2]
            ds = gmm_outs[:, :, -1]
            return mus, sigmas, logpi, rs, ds
        else:
            return mus, sigmas, logpi
