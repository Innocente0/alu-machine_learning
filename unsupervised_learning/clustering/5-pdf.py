#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """calculates the probability density function of a Gaussian"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0]:
        return None
    if S.shape != (m.shape[0], m.shape[0]):
        return None

    d = m.shape[0]

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        if det <= 0:
            return None
    except np.linalg.LinAlgError:
        return None

    diff = X - m
    exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)
    coeff = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    P = coeff * np.exp(exponent)

    return np.maximum(P, 1e-300)
