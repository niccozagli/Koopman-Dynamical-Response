from __future__ import annotations

from typing import Sequence, Tuple, cast

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.linalg import eig
from scipy.signal import correlate, correlation_lags


def get_spectral_properties(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the sorted (decreasing orders in terms of absolute value) of eigenvalues
    and eigenvectors of the Koopman matrix.
    """
    eig_result = cast(
        tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]],
        eig(K, left=True, right=True),
    )

    eigenvalues, left_eigenvectors, right_eigenvectors = eig_result

    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    right_eigenvectors = right_eigenvectors[:, sorted_indices]
    left_eigenvectors = left_eigenvectors[:, sorted_indices]

    diag = np.diag(left_eigenvectors.T.conj() @ right_eigenvectors)
    scale_factors = 1.0 / np.sqrt(diag)
    right_eigenvectors_normalised = right_eigenvectors * scale_factors[np.newaxis, :]
    left_eigenvectors_normalised = (
        left_eigenvectors * scale_factors[np.newaxis, :].conj()
    )
    return eigenvalues, right_eigenvectors_normalised, left_eigenvectors_normalised


def check_if_complex(obs: np.ndarray):
    return np.iscomplex(obs).any()


def get_acf(
    obs: np.ndarray,
    Dt: float,
    nlags: int = 1500,
):
    is_complex = check_if_complex(obs)
    if is_complex:
        obs_real, obs_imag = np.real(obs), np.imag(obs)
        cf_real = np.asarray(
            sm.tsa.acf(obs_real, nlags=nlags, qstat=False, alpha=None)
        ) * np.var(obs_real)
        cf_imag = np.asarray(
            sm.tsa.acf(obs_imag, nlags=nlags, qstat=False, alpha=None)
        ) * np.var(obs_imag)
        cf = cf_real + cf_imag
    else:
        cf = np.asarray(sm.tsa.acf(obs, nlags=nlags)) * np.var(obs)

    lags = np.linspace(0, nlags * Dt, nlags + 1)
    return lags, cf


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_lag: int | None = None,
    demean: bool = True,
    normalization: str = "unbiased",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation using scipy.signal.correlate.

    Note: for complex inputs, scipy.signal.correlate computes the conjugate
    correlation by conjugating the second input.

    Normalization:
        - "unbiased": divide each lag by (N - lag) (default, fewer samples at long lags)
        - "biased": divide each lag by N

    Parameters:
        x: 1D signal (real or complex).
        y: 1D signal (real or complex).
        dt: sampling interval.
        max_lag: maximum lag (in samples). Defaults to len(x) - 1.
        demean: if True, subtract mean from x and y before correlation.
        normalization: "unbiased" or "biased".

    Returns:
        lags: time lags (same units as dt).
        corr: cross-correlation values for non-negative lags.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    n = x.shape[0]
    if n == 0:
        raise ValueError("x and y must have at least one sample")

    if max_lag is None:
        max_lag = n - 1
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    if demean:
        x = x - np.mean(x)
        y = y - np.mean(y)

    corr_full = correlate(x, y, mode="full", method="auto")
    lags_full = correlation_lags(n, n, mode="full")

    nonneg = lags_full >= 0
    lags = lags_full[nonneg][: max_lag + 1]
    corr = corr_full[nonneg][: max_lag + 1]

    normalization = normalization.lower()
    if normalization == "unbiased":
        denom = (n - lags).astype(float)
        corr = corr / denom
    elif normalization == "biased":
        corr = corr / float(n)
    else:
        raise ValueError("normalization must be 'unbiased' or 'biased'")

    return lags * dt, corr


def find_index(indices: Sequence[Tuple[int, ...]], target: Tuple[int, ...]) -> int:
    try:
        return indices.index(target)
    except ValueError:
        return -1
