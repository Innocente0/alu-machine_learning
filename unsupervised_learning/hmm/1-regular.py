#!/usr/bin/env python3
"""Regular markov chain"""

import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    n = P.shape[0]

    if np.any(P <= 0):
        power = np.linalg.matrix_power(P, 100)
        if np.any(power <= 0):
            return None

    A = P.T - np.eye(n)
    A = np.vstack((A, np.ones((1, n))))
    b = np.zeros(n + 1)
    b[-1] = 1

    try:
        steady = np.linalg.lstsq(A, b, rcond=None)[0]
    except Exception:
        return None

    return steady[np.newaxis, :]
