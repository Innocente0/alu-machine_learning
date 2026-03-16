#!/usr/bin/env python3
"""Forward algorithm for a Hidden Markov Model"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for an HMM"""
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

    F = np.zeros((N, T))

    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (Transition.T @ F[:, t - 1])

    P = np.sum(F[:, T - 1])

    return P, F
