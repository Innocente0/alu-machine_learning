#!/usr/bin/env python3
"""Calculates Shannon entropy and P affinities for one point"""

import numpy as np


def HP(Di, beta):
    """calculates the Shannon entropy and P affinities"""
    if not isinstance(Di, np.ndarray) or Di.ndim != 1:
        return None, None
    if not isinstance(beta, np.ndarray) or beta.shape != (1,):
        return None, None

    A = np.exp(-Di * beta[0])
    sum_A = np.sum(A)
    Pi = A / sum_A
    Hi = np.log2(sum_A) + beta[0] * np.sum(Di * Pi) / np.log(2)

    return Hi, Pi
