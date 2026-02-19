from __future__ import annotations

from typing import Tuple

import numpy as np


def normalise_data_chebyshev(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each column of data to [-1, 1].

    Parameters:
        data: (n_samples, d) input array.

    Returns:
        scaled_data: normalized data in [-1, 1]
        data_min: minimum values per column (shape: d,)
        data_max: maximum values per column (shape: d,)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scaled = 2 * (data - data_min) / (data_max - data_min) - 1
    return scaled, data_min, data_max


def make_snapshots(data: np.ndarray, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create snapshot pairs (X, Y) from a single trajectory.

    Parameters:
        data: array of shape (n_samples, dim)
        lag: positive integer time lag

    Returns:
        X: data[:-lag]
        Y: data[lag:]
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n_samples = data.shape[0]
    if lag >= n_samples:
        raise ValueError(f"lag={lag} is too large for data length {n_samples}")
    return data[: -lag], data[lag:]
