import gym
import numpy as np


def run_env(n_episode: int = 50, n_step: int = 1000, scale: bool = True):
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


def save_run(a: np.ndarray, b: np.ndarray, filepath: str = '../data/carracing'):
    np.savez_compressed(filepath, a=a, b=b)


def run(n_fold: int = 10,
        n_episode: int = 50,
        n_step: int = 1000,
        scale: bool = True,
        filepath: str = '../data/carracing'
        ):
    for fold in range(n_fold):
        obs, act = run_env(n_episode, n_step, scale)
        save_run(obs, act, filepath + '_' + str(fold))
