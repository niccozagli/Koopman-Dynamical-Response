from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from koopman_response.algorithms.dictionaries import Dictionary
from koopman_response.utils.koopman import get_spectral_properties


@dataclass
class KoopmanSpectrum:
    """
    Container for Koopman spectral objects and eigenfunction evaluation.
    """

    K: np.ndarray
    dictionary: Dictionary
    eigenvalues: np.ndarray
    right_eigvecs: np.ndarray
    left_eigvecs: np.ndarray

    @classmethod
    def from_koopman_matrix(cls, K: np.ndarray, dictionary: Dictionary) -> "KoopmanSpectrum":
        eigenvalues, right_eigvecs, left_eigvecs = get_spectral_properties(K)
        return cls(
            K=K,
            dictionary=dictionary,
            eigenvalues=eigenvalues,
            right_eigvecs=right_eigvecs,
            left_eigvecs=left_eigvecs,
        )

    def evaluate_eigenfunctions(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate all Koopman eigenfunctions at a batch of points.

        Returns shape (n_samples, n_eig).
        """
        phi = self.dictionary.evaluate_batch(data)
        return phi @ self.right_eigvecs

    def eigenfunction(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable eigenfunction phi_i(x).
        """
        if i < 0 or i >= self.right_eigvecs.shape[1]:
            raise IndexError("eigenfunction index out of bounds")
        v = self.right_eigvecs[:, i]

        def _phi(x: np.ndarray) -> np.ndarray:
            return self.dictionary.evaluate(x) @ v

        return _phi

    def eigenfunction_inner_product(self, G: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix of Koopman eigenfunctions:

            Xi^* G Xi

        where Xi is the matrix of right eigenvectors (columns), and G is the
        EDMD Gram matrix.
        """
        return self.right_eigvecs.conj().T @ G @ self.right_eigvecs

    def psi_inner(self, data: np.ndarray, f_values: np.ndarray) -> np.ndarray:
        """
        Compute <psi, f>_0 from trajectory samples:

            <psi, f>_0 = (1/N) * Phi^* f

        where Phi is the dictionary evaluation on data.
        """
        phi = self.dictionary.evaluate_batch(data)
        f_vals = np.asarray(f_values)
        if f_vals.ndim == 1:
            f_vals = f_vals[:, None]
        if f_vals.shape[0] != phi.shape[0]:
            raise ValueError("f_values and data must have matching first dimension")
        inner = (phi.conj().T @ f_vals) / phi.shape[0]
        if inner.ndim == 2 and inner.shape[1] == 1:
            return inner[:, 0]
        return inner

    def phi_inner(self, G: np.ndarray, psi_inner: np.ndarray) -> np.ndarray:
        """
        Compute <phi, f>_0 from <psi, f>_0:

            <phi, f>_0 = Xi^* <psi, f>_0
        """
        return self.right_eigvecs.conj().T @ psi_inner

    def best_coefficients(self, G: np.ndarray, psi_inner: np.ndarray) -> np.ndarray:
        """
        Compute coefficients for the best decomposition in the Koopman basis:

            f_hat = G_phi^+ <phi, f>_0
                  = (Xi^* G Xi)^+ (Xi^* <psi, f>_0)
        """
        G_phi = self.eigenfunction_inner_product(G)
        phi_inner = self.phi_inner(G, psi_inner)
        coeffs = np.linalg.pinv(G_phi) @ phi_inner
        if coeffs.ndim == 2 and coeffs.shape[1] == 1:
            return coeffs[:, 0]
        return coeffs

    def correlation_function(
        self,
        G: np.ndarray,
        coeff_f: np.ndarray,
        coeff_g: np.ndarray,
        eigenvalues: np.ndarray | None = None,
    ):
        """
        Return a callable C_fg(k) using Koopman eigenfunctions (discrete time):

            C_fg(k) = coeff_g^* @ G_phi @ (coeff_f * lambda^k)

        where G_phi = Xi^* G Xi and lambda are the eigenvalues used in the
        power. If eigenvalues is None, self.eigenvalues are used.
        """
        eigs = self.eigenvalues if eigenvalues is None else eigenvalues
        G_phi = self.eigenfunction_inner_product(G)

        coeff_f = np.asarray(coeff_f).reshape(-1)
        coeff_g = np.asarray(coeff_g).reshape(-1)
        eigs = np.asarray(eigs).reshape(-1)

        # drop the static mode
        coeff_f = coeff_f[1:]
        coeff_g = coeff_g[1:]
        eigs = eigs[1:]
        G_phi = G_phi[1:, 1:]
        row = coeff_g.conj() @ G_phi

        def _corr(k):
            if np.isscalar(k):
                return row @ (coeff_f * (eigs ** k))
            k_arr = np.asarray(k)
            pow_k = np.power(eigs, k_arr[:, None])
            return (pow_k * coeff_f[None, :]) @ row

        return _corr
