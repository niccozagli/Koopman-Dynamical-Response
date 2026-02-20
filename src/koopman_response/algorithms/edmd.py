from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.dictionaries import Dictionary
from koopman_response.algorithms.spectrum import KoopmanSpectrum
from koopman_response.utils.koopman import (
    Koopman_correlation_function,
    get_spectral_properties,
)


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
        return self.K # type:ignore

    def evaluate_dictionary(self, data: np.ndarray) -> np.ndarray:
        return self.dictionary.evaluate_batch(data)

    def evaluate_koopman_eigenfunctions(
        self, data: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate Koopman eigenfunctions at data points.
        """
        psi = self.dictionary.evaluate_batch(data)
        return psi @ eigenvectors

    def gram(self) -> np.ndarray:
        if self.G is None:
            raise ValueError("G is not set. Run fit_snapshots() first.")
        return self.G


class Tikhonov:
    def __init__(self, alpha: float = 1e-7):
        self.alpha = alpha

    def tikhonov(self, edmd: EDMD) -> np.ndarray:
        if edmd.G is None or edmd.A is None:
            raise ValueError("G and A are not set. Run fit_snapshots() first.")
        eye = np.eye(edmd.G.shape[0], dtype=edmd.G.dtype)
        return np.linalg.solve(edmd.G + self.alpha * eye, edmd.A)


class TSVD:
    def __init__(self, rel_threshold: float = 1e-6):
        self.rel_threshold: float = rel_threshold
        self.Ur: Optional[np.ndarray] = None
        self.Sr: Optional[np.ndarray] = None
        self.Kreduced: Optional[np.ndarray] = None
        self.reduced_right_eigvecs: Optional[np.ndarray] = None
        self.reduced_left_eigvecs: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.Gr: Optional[np.ndarray] = None
        self.lambdas: Optional[np.ndarray] = None
        self.Gr_inv: Optional[np.ndarray] = None

    def decompose(self, edmd: EDMD) -> np.ndarray:
        if edmd.G is None or edmd.A is None:
            raise ValueError("G and A are not set. Run fit_snapshots() first.")

        U, S, Vt = np.linalg.svd(edmd.G, full_matrices=False)
        r = np.sum(S > self.rel_threshold * S[0])
        Ur = U[:, :r]
        Sr_inv = np.diag(1 / S[:r])
        K_reduced = Sr_inv @ (Ur.T.conj() @ edmd.A @ Ur)
        Gr_inv = Ur @ Sr_inv @ Ur.T

        self.Ur = Ur
        self.Sr = S[:r]
        self.Gr = np.diag(S[:r])
        self.Kreduced = K_reduced
        self.Gr_inv = Gr_inv
        return K_reduced

    def get_spectral_properties(self) -> None:
        if self.Kreduced is None:
            raise RuntimeError("You must call decompose() before spectral properties.")

        eigenvalues, right_eigvecs, left_eigvecs = get_spectral_properties(self.Kreduced)
        self.reduced_right_eigvecs = right_eigvecs
        self.reduced_left_eigvecs = left_eigvecs
        self.eigenvalues = eigenvalues

    def find_continuous_time_eigenvalues(self, dt: float, tau: int = 1) -> None:
        if self.eigenvalues is None:
            return
        self.lambdas = np.log(self.eigenvalues) / (dt * tau)

    def project_reduced_space(self, dictionary_projections: np.ndarray) -> np.ndarray:
        if self.Ur is None:
            raise RuntimeError("You must call decompose() before mapping eigenvectors.")
        return self.Ur.conj().T @ dictionary_projections


class ProjectionKoopmanSpace:
    def __init__(self, threshold_lambda: float = -2):
        self.threshold: float = threshold_lambda
        self.lambdas: Optional[np.ndarray] = None
        self.Vn: Optional[np.ndarray] = None
        self.Gn: Optional[np.ndarray] = None
        self.Gr: Optional[np.ndarray] = None

    def set_subspace(self, tsvd: TSVD) -> None:
        if tsvd.lambdas is None or tsvd.reduced_right_eigvecs is None or tsvd.Gr is None:
            raise RuntimeError(
                "TSVD must be performed and continuous-time eigenvalues computed first."
            )

        lambdas = tsvd.lambdas
        indx = np.where(np.real(lambdas) > self.threshold)[0]

        lambdas_good = lambdas[indx]
        Vn = tsvd.reduced_right_eigvecs[:, indx]
        Gn = Vn.T.conj() @ tsvd.Gr @ Vn

        self.lambdas = lambdas_good
        self.Vn = Vn
        self.Gn = Gn
        self.Gr = tsvd.Gr

    def project_to_koopman_space(self, reduced_svd_projections: np.ndarray) -> np.ndarray:
        if self.Gn is None or self.Vn is None or self.Gr is None:
            raise RuntimeError("You must call set_subspace() before mapping eigenvectors.")

        return (
            np.linalg.pinv(self.Gn)
            @ self.Vn.conj().T
            @ self.Gr
            @ reduced_svd_projections
        )

    def reconstruct_correlation_function(
        self, coefficients_f: np.ndarray, coefficients_g: np.ndarray
    ):
        if self.Gn is None:
            raise RuntimeError("You must call set_subspace() before mapping eigenvectors.")

        def koopman_reconstruction(t: float):
            return Koopman_correlation_function(
                t=t,
                M=self.Gn,
                alpha1=coefficients_f,
                alpha2=coefficients_g,
                eigenvalues=self.lambdas,
            )

        return koopman_reconstruction


Projection_Koopman_Space = ProjectionKoopmanSpace
