#!/usr/bin/env python3
"""Calculates gradients for t-SNE"""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates the gradients of Y"""
    Q, num = Q_affinities(Y)

    PQ = P - Q
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    dY = np.sum((PQ * num)[:, :, np.newaxis] * diff, axis=1)

    return dY, Q
