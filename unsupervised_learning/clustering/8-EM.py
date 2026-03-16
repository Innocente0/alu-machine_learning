#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, prev_log_likelihood = expectation(X, pi, m, S)
    if g is None or prev_log_likelihood is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(
            prev_log_likelihood))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_likelihood))

        if abs(log_likelihood - prev_log_likelihood) <= tol:
            if verbose and i % 10 != 0 and i != iterations:
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, log_likelihood))
            return pi, m, S, g, log_likelihood

        prev_log_likelihood = log_likelihood

    return pi, m, S, g, log_likelihood
