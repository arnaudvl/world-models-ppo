import cv2
import numpy as np
from typing import Tuple


def preprocess_obs(obs: np.ndarray,
                   grayscale: bool = False,
                   minmax: Tuple[float, float] = None,
                   obs_old: np.ndarray = None,
                   frames: int = 1,
                   resize: Tuple[int, int] = None):
    """
    Pre-process an observation from the environment.

    Parameters
    ----------
    obs
        Environment observation.
    grayscale
        Whether to apply gray-scaling.
    minmax
        Min and max range to scale observation.
    obs_old
        Previous observation, used for frame appending.
    frames
        Number of frames.
    resize
        Pixel size (H, W) of observation to be returned.

    Returns
    -------
    Preprocessed observation.
    """
    if resize is not None:  # resize observation
        obs = cv2.resize(obs, dsize=resize, interpolation=cv2.INTER_LINEAR)

    if grayscale:
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        obs = obs[..., np.newaxis]  # add dimension

    if minmax is not None:
        min, max = minmax[0], minmax[1]
        obs_min, obs_max = obs.min(), obs.max()
        obs = ((obs - obs_min) / (obs_max - obs_min)) * (max - min) + min

    if obs_old is not None and frames > 1:  # keep last frames
        obs = np.concatenate((obs, obs_old[:, :, :-1]), axis=-1)

    if obs.shape[-1] < frames:  # make sure output obs has right nb of channels
        pad = np.zeros(obs.shape[:2] + (frames - obs.shape[-1],))
        obs = np.concatenate((obs, pad), axis=-1)

    return obs
