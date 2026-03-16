#!/usr/bin/env python3
"""Calculates intra-cluster variance"""

import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2
    return np.sum(np.min(distances, axis=1))