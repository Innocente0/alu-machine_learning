#!/usr/bin/env python3
"""
module: 7-gettin_cozy

Provides cat_matrices2D(mat1, mat2, axis=0) which concatenates two 2D matrices
along the specified axis (0 for rows, 1 for columns) and returns a new matrix.
Returns None if the matrices cannot be concatenated due to shape mismatch.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices element-wise along rows (axis=0) or columns (axis=1).

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.
        axis (int): 0 to stack rows, 1 to extend columns.

    Returns:
        list of lists: New concatenated matrix, or None if shapes differ.
    """
    # Axis 0: append mat2's rows below mat1
    if axis == 0:
        # Check column counts match (or derive if one is empty)
        cols1 = len(mat1[0]) if mat1 else None
        cols2 = len(mat2[0]) if mat2 else None
        if cols1 is not None and cols2 is not None and cols1 != cols2:
            return None

        # Build a new matrix with copied rows
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]

    # Axis 1: append mat2's columns to the right of mat1
    if axis == 1:
        # Must have same number of rows
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for row1, row2 in zip(mat1, mat2):
            new_matrix.append(row1.copy() + row2.copy())
        return new_matrix

    # Invalid axis
    return None
