#!/usr/bin/env python3
"""PCA module"""

import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    _, s, vh = np.linalg.svd(X, full_matrices=False)
    explained_variance = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.searchsorted(explained_variance, var, side='right') + 1

    W = vh.T[:, :nd]
    return W
