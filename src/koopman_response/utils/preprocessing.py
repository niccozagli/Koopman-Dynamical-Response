from __future__ import annotations

from typing import Tuple

import numpy as np


def cosine_trapezoid_weights(
    space_coord: np.ndarray,
    n_features: int | None = None,
) -> np.ndarray:
    """
    Build trapezoidal quadrature weights with cosine meridional weighting.

    The input coordinates are normalized to [0, 1], then weighted by
    cos(pi * x / 2). This matches the EBM latitude metric used for positive
    meridional coordinates.
    """
    coord = np.asarray(space_coord, dtype=float)
    if coord.ndim != 1:
        raise ValueError("space_coord must be a 1D array")
    if coord.size == 0:
        raise ValueError("space_coord must not be empty")
    if not np.all(np.isfinite(coord)):
        raise ValueError("space_coord must be finite")
    if n_features is not None:
        n = int(n_features)
        if n < 1:
            raise ValueError("n_features must be >= 1")
        if coord.size not in {1, n}:
            raise ValueError("space_coord length must be 1 or match n_features")
    else:
        n = coord.size

    x_min = coord.min()
    x_max = coord.max()
    if np.isclose(x_max, x_min):
        x_grid = np.linspace(0.0, 1.0, n)
    else:
        if coord.size != n:
            raise ValueError("non-degenerate space_coord must match n_features")
        x_grid = (coord - x_min) / (x_max - x_min)

    dx_weight = np.empty_like(x_grid)
    if x_grid.size < 2:
        dx_weight[...] = 1.0
    else:
        dx_weight[0] = 0.5 * (x_grid[1] - x_grid[0])
        dx_weight[-1] = 0.5 * (x_grid[-1] - x_grid[-2])
        if x_grid.size > 2:
            dx_weight[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])

    cos_weight = np.cos(0.5 * np.pi * x_grid)
    return np.clip(cos_weight * dx_weight, a_min=0.0, a_max=None)


def minmax_scale(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max scale each column of data to a target range.

    Parameters:
        data: (n_samples, d) input array.
        feature_range: target range (min, max), default (-1, 1).

    Returns:
        scaled_data: normalized data in feature_range
        data_min: minimum values per column (shape: d,)
        data_max: maximum values per column (shape: d,)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    a, b = feature_range
    scaled = (b - a) * (data - data_min) / (data_max - data_min) + a
    return scaled, data_min, data_max


def standardize(
    data: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize each column of data to zero mean and unit variance.

    Parameters:
        data: (n_samples, d) input array.
        eps: small value to avoid division by zero.

    Returns:
        scaled_data: standardized data
        mean: mean values per column (shape: d,)
        std: standard deviation per column (shape: d,), with zeros replaced by 1.0
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std_safe = np.where(std < eps, 1.0, std)
    scaled = (data - mean) / std_safe
    return scaled, mean, std_safe


def standardize_global(
    data: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Standardize data using a single global mean and std over all entries.

    Parameters:
        data: (n_samples, d) input array.
        eps: small value to avoid division by zero.

    Returns:
        scaled_data: standardized data
        mean: global mean (scalar)
        std: global standard deviation (scalar), with zeros replaced by 1.0
    """
    mean = float(np.mean(data))
    std = float(np.std(data))
    std_safe = 1.0 if std < eps else std
    scaled = (data - mean) / std_safe
    return scaled, mean, std_safe


def make_snapshots(
    data: np.ndarray,
    lag: int = 1,
    stride: int = 1,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create snapshot pairs (X, Y) from a single trajectory.

    Parameters:
        data: array of shape (n_samples, dim)
        lag: positive integer time lag
        stride: positive integer subsampling stride (use every stride-th sample)
        dt: base time step between consecutive samples in data

    Returns:
        X: data[:-lag]
        Y: data[lag:]
        dt_eff: effective time step (dt * lag * stride)
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if stride > 1:
        data = data[::stride]
    n_samples = data.shape[0]
    if lag >= n_samples:
        raise ValueError(f"lag={lag} is too large for data length {n_samples}")
    dt_eff = float(dt) * int(lag) * int(stride)
    return data[:-lag], data[lag:], dt_eff
