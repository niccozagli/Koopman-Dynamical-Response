from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.kernels import Kernel


def _kernel_matrix(
    kernel: Kernel,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int | None,
    show_progress: bool,
) -> np.ndarray:
    n_rows = X.shape[0]
    n_cols = Y.shape[0]

    sample = kernel(X[:1], Y[:1])
    dtype = sample.dtype
    K = np.empty((n_rows, n_cols), dtype=dtype)

    if batch_size is None or batch_size >= n_rows:
        K[:, :] = kernel(X, Y)
        return K

    iterator = range(0, n_rows, batch_size)
    for start in tqdm(iterator, disable=not show_progress):
        end = min(start + batch_size, n_rows)
        K[start:end, :] = kernel(X[start:end], Y)
    return K


class KernelDMD:
    """
    Kernel Dynamic Mode Decomposition (KDMD).

    This class mirrors EDMD but uses a kernel to avoid explicit feature maps.
    """

    def __init__(
        self,
        kernel: Kernel,
        dt_eff: float | None = None,
        reg: float = 0.0,
        use_pinv: bool = False,
    ):
        if reg < 0:
            raise ValueError("reg must be non-negative")
        self.kernel = kernel
        self.dt_eff = dt_eff
        self.reg = float(reg)
        self.use_pinv = bool(use_pinv)

        self.G: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None

    def fit_snapshots(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 5_000,
        show_progress: bool = True,
        fit_kernel: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit KDMD on snapshot pairs (X, Y) and compute Gram matrices.

        Uses the convention:
            G_hat[i, j] = k(X[i], X[j])
            A_hat[i, j] = k(Y[i], X[j])
        """
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if fit_kernel:
            self.kernel.fit(X)

        n_samples = X.shape[0]
        Kxx = _kernel_matrix(self.kernel, X, X, batch_size, show_progress)
        Kyx = _kernel_matrix(self.kernel, Y, X, batch_size, show_progress)

        G = Kxx / n_samples
        A = Kyx / n_samples

        self.G = G
        self.A = A
        self.K = None
        return G, A

    def solve_koopman(
        self,
        reg: float | None = None,
        use_pinv: bool | None = None,
    ) -> np.ndarray:
        """
        Solve for the Koopman matrix K given stored G and A.
        """
        if self.G is None or self.A is None:
            raise ValueError("G and A are not set. Run fit_snapshots() first.")

        G = self.G
        A = self.A
        reg_val = self.reg if reg is None else float(reg)
        use_pinv_val = self.use_pinv if use_pinv is None else bool(use_pinv)

        if reg_val < 0:
            raise ValueError("reg must be non-negative")
        if reg_val > 0.0:
            G = G + reg_val * np.eye(G.shape[0], dtype=G.dtype)

        if use_pinv_val:
            K = np.linalg.pinv(G) @ A
        else:
            K = np.linalg.solve(G, A)

        self.K = K
        return K

    def gram(self) -> np.ndarray:
        if self.G is None:
            raise ValueError("G is not set. Run fit_snapshots() first.")
        return self.G
