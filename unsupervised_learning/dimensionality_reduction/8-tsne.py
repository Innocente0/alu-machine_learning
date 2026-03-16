#!/usr/bin/env python3
"""Performs t-SNE"""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """performs a t-SNE transformation"""
    X = pca(X, idims)
    n = X.shape[0]
    P = P_affinities(X, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    dY = np.zeros((n, ndims))
    iY = np.zeros((n, ndims))

    for i in range(iterations + 1):
        if i != 0:
            if i <= 100:
                dYi, Q = grads(Y, P * 4)
            else:
                dYi, Q = grads(Y, P)

            a = 0.5 if i < 20 else 0.8
            iY = a * iY - lr * dYi
            Y = Y + iY
            Y = Y - np.mean(Y, axis=0)

        if i != 0 and i % 100 == 0:
            if i <= 100:
                _, Q = grads(Y, P * 4)
                C = cost(P * 4, Q)
            else:
                _, Q = grads(Y, P)
                C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

    return Y
