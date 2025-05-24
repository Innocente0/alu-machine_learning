#!/usr/bin/env python3
"""
module: 5-across_the_planes

Provides add_matrices2D(mat1, mat2), which returns a new matrix equal to
the element-wise sum of two equally-shaped 2D lists, or None if their shapes differ.
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-by-element.
    Return a new matrix, or None if they don’t have the same dimensions.
    """
    # 1) Check same number of rows
    if len(mat1) != len(mat2):
        return None

    # 2) Prepare the result
    result = []

    # 3) Loop over each row
    for row_idx in range(len(mat1)):
        row1 = mat1[row_idx]
        row2 = mat2[row_idx]

        # 4) Check each row has same length
        if len(row1) != len(row2):
            return None

        # 5) Sum elements in this row
        new_row = []
        for col_idx in range(len(row1)):
            new_row.append(row1[col_idx] + row2[col_idx])

        # 6) Add summed row to result
        result.append(new_row)

    return result
