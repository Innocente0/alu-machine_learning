#!/usr/bin/env python3
"""
module: 14-saddle_up

Provides np_matmul(mat1, mat2), which performs matrix multiplication
of two NumPy ndarrays and returns the result.
"""


import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiply two NumPy arrays using matrix multiplication and return the result.
    """
    return mat1 @ mat2
