import numpy as np
import pytest

from koopman_response.algorithms import GaussianKernel, WeightedGaussianKernel
from koopman_response.utils import cosine_trapezoid_weights


def test_weighted_gaussian_matches_scaled_gaussian():
    X = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 2.0]])
    Y = np.array([[0.5, -1.0, 1.5], [2.0, 0.0, -0.5]])
    weights = np.array([0.25, 2.0, 0.0])
    sigma = 1.7

    scale = np.sqrt(weights)
    expected = GaussianKernel(sigma=sigma)(X * scale[None, :], Y * scale[None, :])
    actual = WeightedGaussianKernel(sigma=sigma, weights=weights)(X, Y)

    np.testing.assert_allclose(actual, expected)


def test_weighted_gaussian_with_no_weights_matches_gaussian():
    X = np.array([[1.0, -2.0], [0.0, 1.0]])
    Y = np.array([[0.5, -1.0], [2.0, 0.0]])
    sigma = 0.9

    expected = GaussianKernel(sigma=sigma)(X, Y)
    actual = WeightedGaussianKernel(sigma=sigma)(X, Y)

    np.testing.assert_allclose(actual, expected)


def test_weighted_gaussian_grad_x_matches_finite_difference():
    kernel = WeightedGaussianKernel(sigma=1.3, weights=np.array([0.2, 1.5, 0.0]))
    X = np.array([[0.3, -0.4, 2.0]])
    Y = np.array([[1.2, 0.5, -1.0], [-0.7, 0.8, 0.25]])
    grad = kernel.grad_x(X, Y)[0]
    eps = 1e-6

    for d in range(X.shape[1]):
        step = np.zeros_like(X)
        step[0, d] = eps
        finite_diff = (kernel(X + step, Y) - kernel(X - step, Y)) / (2.0 * eps)
        np.testing.assert_allclose(grad[:, d], finite_diff[0], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "kernel",
    [
        GaussianKernel(sigma=1.3),
        WeightedGaussianKernel(sigma=1.3),
        WeightedGaussianKernel(sigma=1.3, weights=np.array([0.2, 1.5, 0.4])),
    ],
)
def test_grad_x_dot_matches_dense_contraction(kernel):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    Y = rng.normal(size=(4, 3))
    V = rng.normal(size=(6, 3))

    dense = np.einsum("ijd,id->ij", kernel.grad_x(X, Y), V)
    fast = kernel.grad_x_dot(X, Y, V)

    np.testing.assert_allclose(fast, dense, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "weights, match",
    [
        (np.array([[1.0, 2.0]]), "1D"),
        (np.array([]), "must not be empty"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, -0.1]), "nonnegative"),
        (np.array([0.0, 0.0]), "positive"),
    ],
)
def test_weighted_gaussian_rejects_invalid_weights(weights, match):
    with pytest.raises(ValueError, match=match):
        WeightedGaussianKernel(weights=weights)


def test_weighted_gaussian_rejects_weight_dimension_mismatch():
    kernel = WeightedGaussianKernel(weights=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="does not match"):
        kernel(np.zeros((3, 4)))


def test_cosine_trapezoid_weights_match_notebook_formula():
    space_coord = np.array([10.0, 20.0, 40.0])

    actual = cosine_trapezoid_weights(space_coord)
    expected = np.array(
        [
            1.0 / 6.0,
            np.sqrt(3.0) / 4.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_cosine_trapezoid_weights_single_point_grid():
    actual = cosine_trapezoid_weights(np.array([42.0]))

    np.testing.assert_allclose(actual, np.array([1.0]))


def test_cosine_trapezoid_weights_degenerate_grid_with_n_features():
    actual = cosine_trapezoid_weights(np.array([7.0]), n_features=4)
    expected = np.array(
        [
            1.0 / 6.0,
            np.sqrt(3.0) / 6.0,
            1.0 / 6.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_cosine_trapezoid_weights_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="1D"):
        cosine_trapezoid_weights(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must not be empty"):
        cosine_trapezoid_weights(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        cosine_trapezoid_weights(np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="must be >= 1"):
        cosine_trapezoid_weights(np.array([0.0]), n_features=0)
    with pytest.raises(ValueError, match="match n_features"):
        cosine_trapezoid_weights(np.array([0.0, 1.0]), n_features=3)
