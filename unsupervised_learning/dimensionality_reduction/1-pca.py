#!/usr/bin/env python3
"""PCA"""

import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(ndim, int) or ndim <= 0 or ndim > min(X.shape):
        return None

    X = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    T = U[:, :ndim] * S[:ndim]
    return T
