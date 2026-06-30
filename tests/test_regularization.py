import numpy as np
import pytest

from koopman_response.algorithms.regularization import TSVDRegularizer


def test_eigsh_fixed_rank_matches_full_leading_values():
    values = np.geomspace(1.0, 1e-6, 30)
    G = np.diag(values)

    full = TSVDRegularizer()
    full.factorize(G, method="eigh")

    partial = TSVDRegularizer()
    partial.factorize(G, method="eigsh", rank=5)

    np.testing.assert_allclose(partial.S, full.S[:5], rtol=1e-10, atol=1e-12)
    assert partial.U.shape == (30, 5)
    assert not partial.factorization_is_full


def test_eigsh_relative_threshold_matches_full_truncation_rank():
    values = np.array([1.0, 0.4, 0.08, 0.02, 0.009, 0.004, 0.001, 1e-4, 1e-5, 1e-6])
    G = np.diag(values)
    A = 0.5 * G
    rel_threshold = 1e-2

    full = TSVDRegularizer()
    full.factorize(G, method="eigh")
    Kr_full, Ur_full, Sr_full = full.solve_from_factorization(
        A,
        rel_threshold=rel_threshold,
    )

    partial = TSVDRegularizer()
    partial.factorize(
        G,
        method="eigsh",
        rel_threshold=rel_threshold,
        initial_rank=2,
    )
    Kr_partial, Ur_partial, Sr_partial = partial.solve_from_factorization(A)

    assert Kr_partial.shape == Kr_full.shape == (4, 4)
    assert Ur_partial.shape == Ur_full.shape == (10, 4)
    np.testing.assert_allclose(Sr_partial, Sr_full, rtol=1e-10, atol=1e-12)


def test_solve_from_partial_factorization_rejects_missing_rank():
    G = np.diag(np.geomspace(1.0, 1e-6, 20))
    A = G.copy()

    tsvd = TSVDRegularizer()
    tsvd.factorize(G, method="eigsh", rank=3)

    with pytest.raises(ValueError, match="Requested rank exceeds"):
        tsvd.solve_from_factorization(A, rank=4)


def test_solve_from_partial_factorization_rejects_unreached_threshold():
    G = np.diag(np.geomspace(1.0, 1e-3, 20))
    A = G.copy()

    tsvd = TSVDRegularizer()
    tsvd.factorize(
        G,
        method="eigsh",
        rel_threshold=1e-2,
        initial_rank=2,
        max_rank=3,
    )

    with pytest.raises(ValueError, match="Relative threshold was not reached"):
        tsvd.solve_from_factorization(A, rel_threshold=1e-2)
