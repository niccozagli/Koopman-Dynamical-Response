from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.dictionaries import Dictionary
from koopman_response.algorithms.kernels import Kernel
from koopman_response.utils.koopman import get_spectral_properties


def _iter_trajectory_segments(trajectories: np.ndarray | Sequence[np.ndarray]):
    if isinstance(trajectories, np.ndarray):
        if trajectories.ndim == 2:
            yield trajectories
            return
        if trajectories.ndim == 3:
            for i in range(trajectories.shape[0]):
                yield trajectories[i]
            return
        raise ValueError("trajectories must be a 2D or 3D array")
    for segment in trajectories:
        yield segment


@dataclass
class KoopmanSpectrumEDMD:
    """
    Container for Koopman spectral objects and eigenfunction evaluation.
    """

    K: np.ndarray
    dictionary: Dictionary
    eigenvalues: np.ndarray
    right_eigvecs: np.ndarray
    left_eigvecs: np.ndarray

    @classmethod
    def from_koopman_matrix(
        cls, K: np.ndarray, dictionary: Dictionary
    ) -> "KoopmanSpectrumEDMD":
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

    def continuous_time_eigenvalues(self, dt_eff: float) -> np.ndarray:
        """
        Convert discrete-time eigenvalues to continuous-time using dt_eff.
        """
        return np.log(self.eigenvalues) / dt_eff

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

    def correlation_function_discrete(
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


@dataclass
class KoopmanSpectrumKDMD:
    """
    Container for Koopman spectral objects in KDMD (no explicit dictionary).
    """

    K: np.ndarray
    eigenvalues: np.ndarray
    right_eigvecs: np.ndarray
    left_eigvecs: np.ndarray
    kernel: Kernel | None = None
    reference_data: np.ndarray | None = None
    U_r: np.ndarray | None = None
    S_r: np.ndarray | None = None

    @classmethod
    def from_koopman_matrix(
        cls,
        K: np.ndarray,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> "KoopmanSpectrumKDMD":
        eigenvalues, right_eigvecs, left_eigvecs = get_spectral_properties(K)
        return cls(
            K=K,
            eigenvalues=eigenvalues,
            right_eigvecs=right_eigvecs,
            left_eigvecs=left_eigvecs,
            kernel=kernel,
            reference_data=reference_data,
            U_r=U_r,
            S_r=S_r,
        )

    def continuous_time_eigenvalues(self, dt_eff: float) -> np.ndarray:
        """
        Convert discrete-time eigenvalues to continuous-time using dt_eff.
        """
        return np.log(self.eigenvalues) / dt_eff

    def _resolve_kdmd_params(
        self,
        kernel: Kernel | None,
        reference_data: np.ndarray | None,
        U_r: np.ndarray | None,
        S_r: np.ndarray | None,
    ) -> tuple[Kernel, np.ndarray, np.ndarray, np.ndarray]:
        kernel_val = self.kernel if kernel is None else kernel
        ref_val = self.reference_data if reference_data is None else reference_data
        U_r_val = self.U_r if U_r is None else U_r
        S_r_val = self.S_r if S_r is None else S_r
        if kernel_val is None:
            raise ValueError("kernel must be provided or stored on KoopmanSpectrumKDMD")
        if ref_val is None:
            raise ValueError("reference_data must be provided or stored on KoopmanSpectrumKDMD")
        if U_r_val is None or S_r_val is None:
            raise ValueError("U_r and S_r must be provided or stored on KoopmanSpectrumKDMD")
        return kernel_val, np.asarray(ref_val), U_r_val, S_r_val

    def eigenfunction_inner_product(self, G: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix of KDMD eigenfunctions:

            Xi^* G Xi

        where Xi is the matrix of right eigenvectors (columns), and G is the
        kernel Gram matrix in feature space.
        """
        return self.right_eigvecs.conj().T @ G @ self.right_eigvecs

    def estimate_eigenfunction_inner_product(
        self,
        trajectories: np.ndarray | Sequence[np.ndarray],
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
        batch_size: int = 5_000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Estimate <phi_k, phi_l>_0 from trajectory segments using batching.

        trajectories can be:
            - array of shape (n_samples, dim)
            - array of shape (n_segments, n_samples, dim)
            - sequence of arrays with shape (n_samples, dim)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )
        inv_sqrt = 1.0 / np.sqrt(S_r_val)
        basis = U_r_val * inv_sqrt[None, :]

        G = None
        n_total = 0
        segments = _iter_trajectory_segments(trajectories)

        for segment in segments:
            seg_arr = np.asarray(segment, dtype=float)
            if seg_arr.ndim != 2:
                raise ValueError("each trajectory segment must be a 2D array")
            n_samples = seg_arr.shape[0]
            iterator = range(0, n_samples, batch_size)
            for start in tqdm(iterator, disable=not show_progress):
                end = min(start + batch_size, n_samples)
                phi = kernel_val(seg_arr[start:end], ref) @ basis
                if G is None:
                    n_features = phi.shape[1]
                    G = np.zeros((n_features, n_features), dtype=phi.dtype)
                G += phi.conj().T @ phi
                n_total += phi.shape[0]

        if G is None or n_total == 0:
            raise ValueError("no samples provided in trajectories")
        G /= n_total
        return self.eigenfunction_inner_product(G)

    def evaluate_eigenfunctions(
        self,
        data: np.ndarray,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Evaluate KDMD eigenfunctions at data points using kernel features.

        Uses:
            phi(x) = k(x, X_ref) @ U_r @ diag(1/sqrt(S_r)) @ v_k
        """
        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )
        data_arr = np.asarray(data, dtype=float)
        squeeze = False
        if data_arr.ndim == 1:
            data_arr = data_arr[None, :]
            squeeze = True
        if data_arr.ndim != 2:
            raise ValueError("data must be a 1D or 2D array")
        if ref.ndim != 2:
            raise ValueError("reference_data must be a 2D array")
        if U_r_val.shape[0] != ref.shape[0]:
            raise ValueError("U_r rows must match reference_data length")
        if U_r_val.shape[1] != S_r_val.shape[0]:
            raise ValueError("U_r columns must match S_r length")
        if self.right_eigvecs.shape[0] != U_r_val.shape[1]:
            raise ValueError("right_eigvecs must match reduced dimension")

        inv_sqrt = 1.0 / np.sqrt(S_r_val)
        transform = U_r_val * inv_sqrt[None, :]
        basis = transform @ self.right_eigvecs

        n_samples = data_arr.shape[0]
        if batch_size is None or batch_size >= n_samples:
            phi = kernel_val(data_arr, ref) @ basis
        else:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            phi = np.empty((n_samples, basis.shape[1]), dtype=basis.dtype)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                phi[start:end] = kernel_val(data_arr[start:end], ref) @ basis

        return phi[0] if squeeze else phi

    def eigenfunction(
        self,
        k: int,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable KDMD eigenfunction phi_k(x).
        """
        if k < 0 or k >= self.right_eigvecs.shape[1]:
            raise IndexError("eigenfunction index out of bounds")
        v_k = self.right_eigvecs[:, k]
        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )

        def _phi(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim != 1:
                raise ValueError("x must be a 1D array")
            inv_sqrt = 1.0 / np.sqrt(S_r_val)
            transform = U_r_val * inv_sqrt[None, :]
            return kernel_val(x_arr[None, :], ref) @ (transform @ v_k)

        return _phi

    def correlation_function_continuous(
        self,
        G: np.ndarray,
        coeff_f: np.ndarray,
        coeff_g: np.ndarray,
        eigenvalues: np.ndarray | None = None,
    ):
        """
        Return a callable C_fg(t) using Koopman eigenfunctions (continuous time):

            C_fg(t) = coeff_g^* @ G_phi @ (coeff_f * exp(t * lambda))

        where G_phi = Xi^* G Xi and lambda are continuous-time eigenvalues used in
        the exponential. If eigenvalues is None, self.eigenvalues are used.
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

        def _corr(t):
            if np.isscalar(t):
                return row @ (coeff_f * np.exp(t * eigs))
            t_arr = np.asarray(t)
            exp_t = np.exp(np.outer(t_arr, eigs))
            return (exp_t * coeff_f[None, :]) @ row

        return _corr
