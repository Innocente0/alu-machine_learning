#!/usr/bin/env python3
"""Markov chain module"""

import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain state after t iterations"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 0:
        return None

    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not np.isclose(np.sum(s), 1):
        return None

    return np.matmul(s, np.linalg.matrix_power(P, t))
