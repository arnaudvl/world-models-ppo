import numpy as np
from os import listdir
import torch
from typing import Tuple


def collate_fn(batch):
    return tuple(zip(*batch))


def generate_obs(iterable):
    for obj in iterable:
        yield obj


class GymDataset(object):
    def __init__(self,
                 root: str,
                 seq_len: int = 0,
                 transform: callable = None,
                 obs_per_batch: int = int(5e4)) -> None:
        """
        Load previously saved Gym environment data.

        Arguments
        ---------
        root
            Directory with saved observations.
        seq_len
            Sequence length of an observation.
        transform
            Function applying data augmentation or transforming the observations.
        obs_per_batch
            Number of observations in a batch of loaded data.
        """
        self.root = root
        self.batch_list = [batch for batch in listdir(root) if batch.endswith('.npz')]
        self.seq_len = seq_len
        self.transform = transform
        if obs_per_batch is not None:
            self.obs_per_batch = obs_per_batch
        else:  # load first batch to infer number of observations
            self.obs_per_batch = np.load(self.root + self.batch_list[0])['b'].shape[0]

    def load_batch(self, i: int) -> None:
        batch_dir = self.root + self.batch_list[i]
        batch = np.load(batch_dir)
        self.observation, self.action = batch['a'], batch['b']

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get observation and action
        observation = self.observation[i:i + self.seq_len + 1].astype(np.float32)
        action = self.action[i:i + self.seq_len + 1].astype(np.float32)

        # check if observation already scaled and rescale if needed
        if observation.max() <= 1.:
            observation *= 255

        # apply observation transformation and data augmentation
        if self.transform is not None:
            obs_tmp = []
            for i_tmp in range(observation.shape[0]):
                obs_tmp.append(self.transform(observation[i_tmp].astype(np.uint8)))
            observation = torch.stack(obs_tmp)
        else:
            observation = torch.as_tensor(observation, dtype=torch.float32)

        # convert to tensors
        action = torch.as_tensor(action, dtype=torch.float32)

        # return sequence or single instances
        if self.seq_len > 0:
            next_observation = observation[1:]
            observation = observation[:-1]
        else:
            observation = torch.squeeze(observation, dim=0)
            action = torch.squeeze(action, dim=0)
            next_observation = None

        return observation, action, next_observation

    def __len__(self) -> int:
        return self.obs_per_batch
