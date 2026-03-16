#!/usr/bin/env python3
"""Calculates symmetric P affinities for t-SNE"""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates the symmetric P affinities of a data set"""
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        beta_min = None
        beta_max = None

        Di = np.concatenate((D[i, :i], D[i, i + 1:]))
        Hi, Pi = HP(Di, betas[i])
        H_diff = Hi - H

        while np.abs(H_diff) > tol:
            if H_diff > 0:
                beta_min = betas[i, 0]
                if beta_max is None:
                    betas[i, 0] *= 2
                else:
                    betas[i, 0] = (betas[i, 0] + beta_max) / 2
            else:
                beta_max = betas[i, 0]
                if beta_min is None:
                    betas[i, 0] /= 2
                else:
                    betas[i, 0] = (betas[i, 0] + beta_min) / 2

            Hi, Pi = HP(Di, betas[i])
            H_diff = Hi - H

        P[i, np.concatenate((np.arange(i), np.arange(i + 1, n)))] = Pi

    P = (P + P.T) / (2 * n)
    return P
