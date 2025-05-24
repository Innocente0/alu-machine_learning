#!/usr/bin/env python3
"""
module: 8-ridin_bareback

Provides mat_mul(mat1, mat2) to multiply two 2D matrices.
Returns a new matrix, or None if dimensions are incompatible.
"""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices mat1 and mat2.
    
    Return a new matrix product, or None if
    the number of columns in mat1 does not match
    the number of rows in mat2.
    """
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    # Check if multiplication is possible
    if cols1 != rows2:
        return None

    # Initialize result matrix with zeros
    result = []
    for i in range(rows1):
        new_row = []
        for j in range(cols2):
            total = 0
            for k in range(cols1):
                total += mat1[i][k] * mat2[k][j]
            new_row.append(total)
        result.append(new_row)

    return result
