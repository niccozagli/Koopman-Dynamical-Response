import numpy as np

from koopman_response.algorithms.kernels import GaussianKernel, Kernel
from koopman_response.algorithms.spectrum import KoopmanSpectrumKDMD


class LinearKernel(Kernel):
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        return np.asarray(X, dtype=float) @ np.asarray(Y, dtype=float).T

    def grad_x(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        return np.broadcast_to(
            Y_arr[None, :, :],
            (X_arr.shape[0], Y_arr.shape[0], Y_arr.shape[1]),
        ).copy()


def make_spectrum(
    right_eigvecs: np.ndarray | None = None,
    kernel: Kernel | None = None,
    reference_data: np.ndarray | None = None,
    U_r: np.ndarray | None = None,
    S_r: np.ndarray | None = None,
) -> KoopmanSpectrumKDMD:
    if right_eigvecs is None:
        right_eigvecs = np.eye(3, dtype=np.complex128)
    n_modes = right_eigvecs.shape[1]
    return KoopmanSpectrumKDMD(
        K=np.eye(n_modes, dtype=np.complex128),
        eigenvalues=np.ones(n_modes, dtype=np.complex128),
        right_eigvecs=right_eigvecs,
        left_eigvecs=np.eye(n_modes, dtype=np.complex128),
        kernel=kernel,
        reference_data=reference_data,
        U_r=U_r,
        S_r=S_r,
    )


def test_response_coefficients_matches_pinv():
    spectrum = make_spectrum()
    G_phi = np.array(
        [
            [2.0, 0.25 + 0.1j, 0.0],
            [0.25 - 0.1j, 1.0, 0.2],
            [0.0, 0.2, 0.5],
        ],
        dtype=np.complex128,
    )
    Delta = np.array([1.0 + 0.5j, -0.2j, 0.3], dtype=np.complex128)

    actual = spectrum.response_coefficients(Delta, G_phi=G_phi)
    expected = np.linalg.pinv(G_phi) @ Delta

    np.testing.assert_allclose(actual, expected)


def test_response_coefficients_supports_rank_and_relative_threshold():
    spectrum = make_spectrum()
    G_phi = np.diag([4.0, 2.0, 1e-4])
    Delta = np.array([4.0, 2.0, 1.0])

    rank_one = spectrum.response_coefficients(Delta, G_phi=G_phi, rank=1)
    rel_cut = spectrum.response_coefficients(Delta, G_phi=G_phi, rel_threshold=1e-2)

    np.testing.assert_allclose(rank_one, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rel_cut, np.array([1.0, 1.0, 0.0]))


def test_response_coefficients_uses_stored_reference_count_for_gram():
    spectrum = make_spectrum(
        right_eigvecs=np.eye(2, dtype=np.complex128),
        reference_data=np.zeros((4, 2)),
        S_r=np.array([8.0, 4.0]),
    )
    Delta = np.array([2.0, 1.0])

    actual = spectrum.response_coefficients(Delta)

    np.testing.assert_allclose(actual, np.array([1.0, 1.0]))


def test_response_coefficients_from_trajectory_matches_manual_delta_solve():
    reference_data = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    U_r = np.eye(3)
    S_r = np.array([4.0, 2.0, 1.0])
    right_eigvecs = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    spectrum = make_spectrum(
        right_eigvecs=right_eigvecs,
        kernel=LinearKernel(),
        reference_data=reference_data,
        U_r=U_r,
        S_r=S_r,
    )
    trajectories = np.array([[0.5, 1.0], [1.5, -0.5], [-1.0, 2.0]])
    X_values = {0: np.array([2.0, -1.0, 0.5]), 1: 1.5}
    G_phi = spectrum.eigenfunction_gram(S_r=S_r, n_samples=3)

    manual_delta = spectrum.delta_from_trajectory(
        trajectories,
        X_values,
        batch_size=2,
        show_progress=False,
    )
    manual = spectrum.response_coefficients(manual_delta, G_phi=G_phi)
    actual, Delta, returned_gram = spectrum.response_coefficients_from_trajectory(
        trajectories,
        X_values,
        G_phi=G_phi,
        batch_size=2,
        show_progress=False,
        return_delta=True,
        return_gram=True,
    )

    np.testing.assert_allclose(Delta, manual_delta)
    np.testing.assert_allclose(returned_gram, G_phi)
    np.testing.assert_allclose(actual, manual)


def test_zero_perturbation_returns_zero_coefficients():
    reference_data = np.array([[1.0, 0.0], [0.0, 1.0]])
    spectrum = make_spectrum(
        right_eigvecs=np.eye(2, dtype=np.complex128),
        kernel=LinearKernel(),
        reference_data=reference_data,
        U_r=np.eye(2),
        S_r=np.ones(2),
    )
    trajectories = np.array([[0.5, 1.0], [1.5, -0.5]])

    coeffs = spectrum.response_coefficients_from_trajectory(
        trajectories,
        X_values=np.zeros_like(trajectories),
        G_phi=np.eye(2),
        show_progress=False,
    )

    np.testing.assert_allclose(coeffs, np.zeros(2))


def test_gaussian_kernel_grad_x_matches_finite_difference():
    kernel = GaussianKernel(sigma=1.7)
    X = np.array([[0.3, -0.4]])
    Y = np.array([[1.2, 0.5], [-0.7, 0.8]])
    grad = kernel.grad_x(X, Y)[0]
    eps = 1e-6

    for d in range(X.shape[1]):
        step = np.zeros_like(X)
        step[0, d] = eps
        finite_diff = (kernel(X + step, Y) - kernel(X - step, Y)) / (2.0 * eps)
        np.testing.assert_allclose(grad[:, d], finite_diff[0], rtol=1e-6, atol=1e-8)
