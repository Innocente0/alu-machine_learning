#!/usr/bin/env python3
"""Baum-Welch algorithm for an HMM"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """forward algorithm"""
    N, T = Emission.shape[0], Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (Transition.T @ F[:, t - 1])

    P = np.sum(F[:, -1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """backward algorithm"""
    N, T = Emission.shape[0], Observation.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] * B[:, t + 1],
            axis=1
        )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for an HMM"""
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    M, M2 = Transition.shape
    M3, N = Emission.shape
    T = Observations.shape[0]

    if M != M2 or M != M3:
        return None, None
    if Initial.shape != (M, 1):
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= N):
        return None, None
    if not np.allclose(np.sum(Transition, axis=1), 1):
        return None, None
    if not np.allclose(np.sum(Emission, axis=1), 1):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    for _ in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            numer = (
                F[:, t][:, np.newaxis]
                * Transition
                * Emission[:, Observations[t + 1]][np.newaxis, :]
                * B[:, t + 1][np.newaxis, :]
            )
            denom = np.sum(numer)
            xi[:, :, t] = numer / denom

        gamma = np.sum(xi, axis=1)

        Transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1)[:, np.newaxis]

        gamma_full = np.hstack(
    (gamma, np.sum(xi[:, :, T - 2], axis=0)[:, None])
)

        for j in range(N):
            Emission[:, j] = np.sum(
                gamma_full[:, Observations == j], axis=1
            )

        Emission = Emission / np.sum(gamma_full, axis=1)[:, np.newaxis]

    return Transition, Emission
