from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.dictionaries import Dictionary


class EDMD:
    """
    Extended Dynamic Mode Decomposition (EDMD) using a user-provided dictionary.

    The algorithm is agnostic to the data source: it operates only on trajectory
    data or snapshot pairs (X, Y).
    """

    def __init__(self, dictionary: Dictionary, dt_eff: float | None = None):
        self.dictionary = dictionary
        self.dt_eff = dt_eff
        self.G: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None

    @property
    def n_features(self) -> int:
        return self.dictionary.n_features

    def fit_snapshots(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 10_000,
        show_progress: bool = True,
        fit_dictionary: bool = True,
    ) -> np.ndarray:
        """
        Fit EDMD on snapshot pairs (X, Y).
        """
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if fit_dictionary:
            self.dictionary.fit(X)

        n_samples = X.shape[0]
        n_features = self.dictionary.n_features
        sample_phi = self.dictionary.evaluate_batch(X[:1])
        dtype = sample_phi.dtype

        G = np.zeros((n_features, n_features), dtype=dtype)
        A = np.zeros((n_features, n_features), dtype=dtype)

        iterator = range(0, n_samples, batch_size)
        for start in tqdm(iterator, disable=not show_progress):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            Phi_X = self.dictionary.evaluate_batch(X_batch)
            Phi_Y = self.dictionary.evaluate_batch(Y_batch)

            G += Phi_X.conj().T @ Phi_X
            A += Phi_X.conj().T @ Phi_Y

        G /= n_samples
        A /= n_samples

        self.G = G
        self.A = A
        self.K = np.linalg.solve(G, A)
        return self.K  # type:ignore

    def gram(self) -> np.ndarray:
        if self.G is None:
            raise ValueError("G is not set. Run fit_snapshots() first.")
        return self.G
