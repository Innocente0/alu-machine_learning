#!/usr/bin/env python3
"""Initializes variables for t-SNE P affinities"""

import numpy as np


def P_init(X, perplexity):
    """initializes D, P, betas, and H for t-SNE"""
    n, d = X.shape

    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2 * np.matmul(X, X.T)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
