#!/usr/bin/env python3
"""
module: 13-cats_got_your_tongue

Provides np_cat(mat1, mat2, axis=0) which concatenates two NumPy arrays
along the specified axis and returns the result.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two NumPy ndarrays along the given axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
