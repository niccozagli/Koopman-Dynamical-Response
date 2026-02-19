from __future__ import annotations

import numpy as np


def Koopman_correlation_function(t, M, alpha1, alpha2, eigenvalues, to_include=None):
    if to_include is None:
        to_include = len(eigenvalues)

    alpha1 = alpha1[1 : to_include + 1]
    alpha2 = alpha2[1 : to_include + 1]
    eigenvalues = eigenvalues[1 : to_include + 1]
    M = M[1 : to_include + 1, 1 : to_include + 1]

    return np.conj(alpha2) @ M @ (alpha1 * np.exp(t * eigenvalues))
