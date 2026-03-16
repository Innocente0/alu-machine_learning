#!/usr/bin/env python3
"""Maximization step for a GMM"""

import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    nk = np.sum(g, axis=1)
    if np.any(nk == 0):
        return None, None, None

    pi = nk / n
    m = (g @ X) / nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = (weighted_diff.T @ diff) / nk[i]

    return pi, m, S
