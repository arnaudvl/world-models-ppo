import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Callable, Tuple, Union
from .agents import WorldModel


class GaussianPolicyWM(nn.Module):
    def __init__(self,
                 action_dim: int,
                 activation: Callable = torch.relu,
                 output_activation: Union[Callable, str] = None) -> None:
        super(GaussianPolicyWM, self).__init__()

        self.mu = WorldModel(
            action_dim,
            action_dim,
            activation=activation,
            output_activation=output_activation
        )

        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, a: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x, a)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi
