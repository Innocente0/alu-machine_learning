#!/usr/bin/env python3
"""Viterbi algorithm for a Hidden Markov Model"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states"""
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

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        temp = V[:, t - 1][:, np.newaxis] * Transition
        B[:, t] = np.argmax(temp, axis=0)
        V[:, t] = np.max(temp, axis=0) * Emission[:, Observation[t]]

    P = np.max(V[:, T - 1])
    last_state = np.argmax(V[:, T - 1])

    path = [0] * T
    path[-1] = last_state

    for t in range(T - 1, 0, -1):
        path[t - 1] = B[path[t], t]

    return path, P
