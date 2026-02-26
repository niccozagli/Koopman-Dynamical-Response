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
    U: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None

    def factorize(
        self,
        G: np.ndarray,
        method: str = "eigh",
        symmetrize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorize the Gram matrix once and cache the result.

        Parameters:
            G: Gram matrix (square).
            method: "eigh" (symmetric eigendecomposition) or "svd".
            symmetrize: if True, use (G + G^*) / 2 before eigh.
        """
        if G.shape[0] != G.shape[1]:
            raise ValueError("G must be square")

        if method == "eigh":
            G_use = 0.5 * (G + G.conj().T) if symmetrize else G
            S, U = np.linalg.eigh(G_use)
            idx = np.argsort(S)[::-1]
            S = S[idx]
            U = U[:, idx]
            S = np.maximum(S, 0.0)
        elif method == "svd":
            U, S, _ = np.linalg.svd(G, full_matrices=False)
        else:
            raise ValueError("method must be 'eigh' or 'svd'")

        self.U = U
        self.S = S
        return U, S

    def _truncate(
        self,
        rel_threshold: Optional[float],
        rank: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.U is None or self.S is None:
            raise ValueError("Gram factorization missing. Call factorize() first.")

        rel_val = self.rel_threshold if rel_threshold is None else float(rel_threshold)
        rank_val = self.rank if rank is None else rank
        if rank_val is not None:
            r = int(rank_val)
        else:
            if self.S.size == 0:
                raise ValueError("Empty spectrum; cannot truncate.")
            if self.S[0] <= 0:
                raise ValueError("Largest eigenvalue is non-positive; cannot truncate.")
            r = int(np.sum(self.S > rel_val * self.S[0]))
        if r < 1:
            raise ValueError("Truncation rank is < 1. Increase rel_threshold or rank.")

        Ur = self.U[:, :r]
        Sr = self.S[:r]
        return Ur, Sr

    def solve_from_factorization(
        self,
        A: np.ndarray,
        rel_threshold: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if self.U is None or self.S is None:
            raise ValueError("Gram factorization missing. Call factorize() first.")
        if A.shape[0] != self.U.shape[0]:
            raise ValueError("A must have the same shape as the Gram factorization.")

        if rel_threshold is not None:
            self.rel_threshold = float(rel_threshold)
        if rank is not None:
            self.rank = int(rank)

        Ur, Sr = self._truncate(rel_threshold, rank)
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
