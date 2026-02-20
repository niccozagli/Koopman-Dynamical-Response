from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TSVDRegularizer:
    """
    Truncated SVD regularizer for EDMD.

    Given G and A, computes an orthonormalized reduced Koopman operator:
        K_r = S_r^{-1/2} U_r^* A U_r S_r^{-1/2}
    where G = U S U^* and r is chosen by rank or relative threshold.
    """

    rel_threshold: float = 1e-6
    rank: Optional[int] = None

    Ur: Optional[np.ndarray] = None
    Sr: Optional[np.ndarray] = None
    Kr: Optional[np.ndarray] = None

    def solve(self, G: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if G.shape[0] != G.shape[1]:
            raise ValueError("G must be square")
        if A.shape != G.shape:
            raise ValueError("A must have the same shape as G")

        U, S, Vh = np.linalg.svd(G, full_matrices=False)
        if self.rank is not None:
            r = int(self.rank)
        else:
            r = int(np.sum(S > self.rel_threshold * S[0]))
        if r < 1:
            raise ValueError("Truncation rank is < 1. Increase rel_threshold or rank.")

        Ur = U[:, :r]
        Sr = S[:r]

        Sr_inv_sqrt = np.diag(1.0 / np.sqrt(Sr))
        Kr = Sr_inv_sqrt @ (Ur.conj().T @ A @ Ur) @ Sr_inv_sqrt

        self.Ur = Ur
        self.Sr = Sr
        self.Kr = Kr
        return Kr, Ur, Sr

    def lift_eigenvectors(self, W: np.ndarray) -> np.ndarray:
        if self.Ur is None:
            raise ValueError("TSVD must be solved before lifting eigenvectors")
        if self.Sr is None:
            raise ValueError("TSVD must be solved before lifting eigenvectors")
        Sr_inv_sqrt = np.diag(1.0 / np.sqrt(self.Sr))
        return self.Ur @ Sr_inv_sqrt @ W

    def gram_inverse(self) -> np.ndarray:
        if self.Ur is None or self.Sr is None:
            raise ValueError("TSVD must be solved before computing gram inverse")
        Sr_inv = np.diag(1.0 / self.Sr)
        return self.Ur @ Sr_inv @ self.Ur.conj().T
