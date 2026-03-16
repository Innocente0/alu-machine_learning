#!/usr/bin/env python3
"""Absorbing Markov chain"""

import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if not np.allclose(np.sum(P, axis=1), 1):
        return False

    n = P.shape[0]

    if np.all(np.diag(P) == 1):
        return True

    absorbing_states = np.where(np.isclose(np.diag(P), 1))[0]
    if absorbing_states.size == 0:
        return False

    non_absorbing = np.setdiff1d(np.arange(n), absorbing_states)

    Q = P[np.ix_(non_absorbing, non_absorbing)]

    for _ in range(n):
        Q = np.matmul(Q, Q)
        if np.all(np.sum(Q, axis=1) < 1):
            return True

    return False
