#!/usr/bin/env python3
"""Expectation step for a GMM"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    probs = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])
    if np.any(probs is None):
        return None, None

    total = np.sum(probs, axis=0)
    g = probs / total
    l = np.sum(np.log(total))

    return g, l
