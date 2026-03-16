#!/usr/bin/env python3
"""PCA transformation module"""

import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(ndim, int) or ndim <= 0 or ndim > X.shape[1]:
        return None

    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    T = np.matmul(U[:, :ndim], np.diag(S[:ndim]))

    return T
