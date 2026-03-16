#!/usr/bin/env python3
"""Calculates Q affinities for t-SNE"""

import numpy as np


def Q_affinities(Y):
    """calculates the Q affinities"""
    sum_Y = np.sum(Y ** 2, axis=1)
    D = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * np.matmul(Y, Y.T)

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
