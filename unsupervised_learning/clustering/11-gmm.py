#!/usr/bin/env python3
"""GMM with sklearn"""

import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    gm = sklearn.mixture.GaussianMixture(n_components=k)
    gm.fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_
    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
