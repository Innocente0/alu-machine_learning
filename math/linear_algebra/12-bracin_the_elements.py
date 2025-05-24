#!/usr/bin/env python3
"""
module: 12-bracin_the_elements

Provides np_elementwise(mat1, mat2), which returns a tuple containing the
element-wise sum, difference, product, and quotient of two NumPy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    on mat1 and mat2. Assumes broadcastable shapes and no empty arrays.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
