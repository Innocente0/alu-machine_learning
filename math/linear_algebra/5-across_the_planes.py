#!/usr/bin/env python3
"""
module: 5-across_the_planes

Provides add_matrices2D(mat1, mat2), which returns a new matrix equal
to the element-wise sum of two equally-shaped 2D lists, or None if
shapes differ.
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-by-element.
    Return a new matrix, or None if they don’t have the same dimensions.
    """
    if len(mat1) != len(mat2):
        return None

    result = []
    for row_idx in range(len(mat1)):
        row1 = mat1[row_idx]
        row2 = mat2[row_idx]
        if len(row1) != len(row2):
            return None

        new_row = []
        for col_idx in range(len(row1)):
            new_row.append(row1[col_idx] + row2[col_idx])
        result.append(new_row)

    return result
