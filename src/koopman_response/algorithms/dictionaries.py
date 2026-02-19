from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import DefaultDict, Dict, List, Tuple, cast

import numpy as np
from scipy.special import eval_chebyt

from koopman_response.utils.signal import find_index


def chebyshev_indices(degree: int, dim: int) -> List[Tuple[int, ...]]:
    indices = [
        cast(Tuple[int, ...], i)
        for i in product(range(degree + 1), repeat=dim)
        if sum(i) <= degree
    ]
    return indices


class Dictionary(ABC):
    """Abstract dictionary interface for EDMD-style algorithms."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate dictionary at a single point x (shape: (dim,))."""

    @abstractmethod
    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        """Evaluate dictionary for a batch (shape: (n_samples, dim))."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of dictionary features."""


class ChebyshevDictionary(Dictionary):
    """Tensorized Chebyshev dictionary on [-1, 1]^dim."""

    def __init__(self, degree: int, dim: int = 3):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.degree = degree
        self.dim = dim
        self.indices = chebyshev_indices(degree, dim)

    @property
    def n_features(self) -> int:
        return len(self.indices)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected x with dim={self.dim}, got {x.shape[0]}")
        return np.array(
            [
                np.prod([eval_chebyt(k, x[d]) for d, k in enumerate(idx)])
                for idx in self.indices
            ],
            dtype=np.float64,
        )

    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != self.dim:
            raise ValueError(
                f"Expected data with dim={self.dim}, got {data.shape[1]}"
            )
        n_samples = data.shape[0]
        n_features = len(self.indices)
        psi = np.empty((n_samples, n_features), dtype=np.float64)
        for n, idx in enumerate(self.indices):
            col = np.ones(n_samples, dtype=np.float64)
            for d, k in enumerate(idx):
                col *= eval_chebyt(k, data[:, d])
            psi[:, n] = col
        return psi

    def chebyshev_U_to_T_matrix(self, n: int) -> np.ndarray:
        """
        Create a matrix M such that:
        M[n, m] gives the coefficient of T_m in U_n(x)
        Returns an (n x n) matrix.
        """
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(0, i + 1, 2):
                m[i, j] = 2
        m[0, 0] = 1  # U_0(x) = T_0(x)
        return m

    def spectral_derivative_tensor_chebyshev_explicit(
        self, c_flat: np.ndarray, direction: int
    ) -> np.ndarray:
        """
        Compute the spectral derivative of a tensorized Chebyshev T_n basis expansion
        along the specified direction (0..dim-1), using the exact formula.
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim - 1}]")

        index_map = {idx: n for n, idx in enumerate(self.indices)}
        dc_dict: DefaultDict[Tuple[int, ...], float] = defaultdict(float)

        for n, idx in enumerate(self.indices):
            coeff = c_flat[n]
            deg = idx[direction]

            if deg == 0:
                continue  # T_0 -> 0

            scale = deg  # from derivative rule
            u_index = deg - 1

            if u_index % 2 == 0:  # even
                for m in range(0, u_index + 1, 2):
                    new_idx = list(idx)
                    new_idx[direction] = m
                    new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                    if new_idx_tuple in index_map:
                        dc_dict[new_idx_tuple] += coeff * scale * 2
                # subtract the constant 1 term
                new_idx = list(idx)
                new_idx[direction] = 0
                new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                if new_idx_tuple in index_map:
                    dc_dict[new_idx_tuple] -= coeff * scale
            else:  # odd
                for m in range(1, u_index + 1, 2):
                    new_idx = list(idx)
                    new_idx[direction] = m
                    new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                    if new_idx_tuple in index_map:
                        dc_dict[new_idx_tuple] += coeff * scale * 2

        dc_flat = np.zeros_like(c_flat)
        for idx, val in dc_dict.items():
            dc_flat[index_map[idx]] = val

        return dc_flat

    def build_derivative_matrix(self, direction: int) -> np.ndarray:
        """
        Constructs the matrix A^{(direction)} such that:
        A @ c = coefficients of d/dx_i f(x), when f(x) = sum c_n psi_n(x)
        """
        n = len(self.indices)
        a = np.zeros((n, n))
        eye = np.eye(n)
        for i in range(n):
            a[:, i] = self.spectral_derivative_tensor_chebyshev_explicit(
                eye[:, i], direction
            )
        return a

    def get_decomposition_observables(self) -> Dict[str, np.ndarray]:
        """
        Return Chebyshev coefficient vectors for common observables.
        """
        if self.dim != 3:
            raise ValueError("Observable decomposition is defined for dim=3.")

        dictionary_decomposition: Dict[str, np.ndarray] = {}

        def coeff_for_degree(degree: Tuple[int, int, int], weight: float = 1.0):
            index = find_index(self.indices, degree)
            coeffs = np.zeros(len(self.indices))
            coeffs[index] = weight
            return coeffs

        dictionary_decomposition["z"] = coeff_for_degree((0, 0, 1))

        # x^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((2, 0, 0), 1 / 2)
        dictionary_decomposition["x^2"] = coeffs

        # y^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((0, 2, 0), 1 / 2)
        dictionary_decomposition["y^2"] = coeffs

        # z^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((0, 0, 2), 1 / 2)
        dictionary_decomposition["z^2"] = coeffs

        # xy
        dictionary_decomposition["xy"] = coeff_for_degree((1, 1, 0))

        return dictionary_decomposition
