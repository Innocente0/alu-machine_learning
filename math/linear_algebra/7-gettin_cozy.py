#!/usr/bin/env python3
"""
module: 7-gettin_cozy

Provides cat_matrices2D(mat1, mat2), which concatenates two 2D matrices
along the specified axis (0 for stacking rows, 1 for extending columns).
Returns None if the matrices cannot be concatenated due to shape mismatch.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along rows (axis=0) or columns (axis=1).

    Return a new matrix, or None if their shapes are incompatible.
    """
    # Axis 0: stack mat2's rows under mat1's rows
    if axis == 0:
        # Check both have the same number of columns
        cols1 = len(mat1[0]) if mat1 else None
        cols2 = len(mat2[0]) if mat2 else None
        if cols1 is not None and cols2 is not None and cols1 != cols2:
            return None
        # Copy each row so originals stay unchanged
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]

    # Axis 1: append mat2's columns to the right of mat1's columns
    if axis == 1:
        # Both must have the same number of rows
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for row1, row2 in zip(mat1, mat2):
            # Copy each row before concatenating
            new_matrix.append(row1.copy() + row2.copy())
        return new_matrix

    # Invalid axis value
    return None
