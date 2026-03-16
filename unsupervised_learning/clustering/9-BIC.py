#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using BIC"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using BIC"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin:
        return None, None, None, None
    if kmax == kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    ks = np.arange(kmin, kmax + 1)

    results = []
    log_likelihoods = []
    bics = []

    for k in ks:
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or log_likelihood is None:
            return None, None, None, None

        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        log_likelihoods.append(log_likelihood)
        bics.append(bic)

    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)

    best_idx = np.argmin(bics)
    best_k = ks[best_idx]
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, bics
