#!/usr/bin/env python3
"""Policy gradient helper"""

import numpy as np


def policy(matrix, weight):
    """computes the policy with a weight of a matrix"""
    z = np.matmul(matrix, weight)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
