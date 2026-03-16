#!/usr/bin/env python3
"""PCA module"""

import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    _, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s) / np.sum(s)
    nd = np.where(cumulative >= var)[0][0] + 1

    return vh.T[:, :nd]
