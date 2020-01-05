from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from typing import Callable, Tuple, Union
from rl.algos.ppo.agents import WorldModel
from rl.algos.ppo.policies import GaussianPolicyWM


class ActorCriticWM(nn.Module):
    def __init__(self,
                 action_space: Union[Box, Discrete] = None,
                 activation: Callable = torch.relu,
                 output_activation: Union[Callable, str] = None) -> None:
        super(ActorCriticWM, self).__init__()

        if isinstance(action_space, Box):
            self.policy = GaussianPolicyWM(
                action_space.shape[0],
                activation=activation,
                output_activation=output_activation
            )

        self.value_function = WorldModel(
            action_space.shape[0],
            1,
            activation=activation,
            output_activation=None,
            output_squeeze=True
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_function(x, a)
        return pi, logp, logp_pi, v
