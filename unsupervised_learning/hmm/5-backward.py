#!/usr/bin/env python3
"""Backward algorithm for a Hidden Markov Model"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for an HMM"""
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    if np.any(Observation < 0) or np.any(Observation >= M):
        return None, None
    if not np.allclose(np.sum(Emission, axis=1), 1):
        return None, None
    if not np.allclose(np.sum(Transition, axis=1), 1):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] * B[:, t + 1],
            axis=1
        )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
