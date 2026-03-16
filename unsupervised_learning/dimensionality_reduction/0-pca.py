#!/usr/bin/env python3
"""PCA module"""

import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    _, s, vh = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(cumulative > var) + 1

    return vh.T[:, :nd]
