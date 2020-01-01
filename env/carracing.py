import argparse
import gym
import numpy as np
import os
from typing import Tuple


def run_env(n_episode: int = 50,
            n_step: int = 1000,
            scale: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carracing environment rollouts.

    Parameters
    ----------
    n_episode
        Number of episodes to run environment.
    n_step
        Number of max steps per episode.
    scale
        Whether to scale the observations.

    Returns
    -------
    Observations and rewards from rollouts
    """
    obs_store, a_store = [], []
    env = gym.make('CarRacing-v0')
    for _ in range(n_episode):
        seed = np.random.randint(int(1e4))
        env.seed(seed)
        obs = env.reset()
        for _ in range(n_step):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            obs_scale = obs / 255 if scale else obs
            obs_store.append(np.expand_dims(obs_scale, axis=0))
            a_store.append(np.expand_dims(action, axis=0))
            if done:
                break
    env.close()

    obs_store_np = np.empty((len(obs_store),) + obs_store[0].shape[1:], dtype=np.float16)
    for i in range(len(obs_store)):
        obs_store_np[i] = obs_store[i]
    a_store_np = np.concatenate(a_store, axis=0).astype(np.float16)

    return obs_store_np, a_store_np


def save_run(a: np.ndarray,
             b: np.ndarray,
             filepath: str
             ) -> None:
    """
    Save arrays a and b.

    Parameters
    ----------
    a
        Numpy array.
    b
        Numpy array.
    filepath
        Path to save to.
    """
    np.savez_compressed(filepath, a=a, b=b)


def run(n_fold_train: int = 10,
        n_fold_test: int = 1,
        n_episode: int = 50,
        n_step: int = 1000,
        scale: bool = True,
        filepath: str = './data/'
        ) -> None:
    """
    Run n_fold rollouts of the environment for n_episode episodes per rollout.

    Parameters
    ----------
    n_fold_train
        Number of environment rollout folds saved for training.
    n_fold_test
        Number of environment rollout folds saved for testing.
    n_episode
        Number of episodes to run environment.
    n_step
        Number of max steps per episode.
    scale
        Whether to scale the observations.
    filepath
        Path to save observations and rewards to.
    """
    dir_train = os.path.join(filepath, 'train')
    if not os.path.isdir(dir_train):
        os.mkdir(dir_train)

    for fold in range(n_fold_train):
        obs, act = run_env(n_episode, n_step, scale)
        save_run(obs, act, os.path.join(dir_train, 'carracing_' + str(fold)))

    dir_test = os.path.join(filepath, 'test')
    if not os.path.isdir(dir_test):
        os.mkdir(dir_test)

    for fold in range(n_fold_test):
        obs, act = run_env(n_episode, n_step, scale)
        save_run(obs, act, os.path.join(dir_test, 'carracing_' + str(fold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect carracing env rollouts.")
    parser.add_argument('--n_fold_train', type=int, default=10)
    parser.add_argument('--n_fold_test', type=int, default=1)
    parser.add_argument('--n_episode', type=int, default=50)
    parser.add_argument('--n_step', type=int, default=1000)
    parser.add_argument('--filepath', type=str, default='./env/data/')
    args = parser.parse_args()
    run(args.n_fold_train, args.n_fold_test, args.n_episode, args.n_step, True, args.filepath)
