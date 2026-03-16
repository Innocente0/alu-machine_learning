#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                new_C[i] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                new_C[i] = np.mean(points, axis=0)

        if np.array_equal(C, new_C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
