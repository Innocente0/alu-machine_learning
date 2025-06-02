#!/usr/bin/env python3
"""
module: 2-cofactor

Provides a function `cofactor(matrix)` that returns the cofactor matrix of
a non‐empty square matrix represented as a list of lists.
Raises TypeError or ValueError for invalid inputs.
"""


def cofactor(matrix):
    """
    Calculate and return the cofactor matrix of `matrix`.

    - If `matrix` is not a list of lists, raise:
        TypeError("matrix must be a list of lists")
    - If `matrix` is empty or not square, raise:
        ValueError("matrix must be a non-empty square matrix")
    - Otherwise, return a new list of lists where each entry [i][j]
      is the cofactor C₍ᵢⱼ₎ = (-1)^(i+j) * det(minor of row i, column j).
    """
    # 1) Check that `matrix` is a list of lists
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    # 2) Ensure it is non-empty and square
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # 3) Helper to compute determinant of any square submatrix
    def _det(mat):
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        total = 0
        for c in range(size):
            # form sub‐minor by removing row 0, col c
            sub = [r[:c] + r[c + 1 :] for r in mat[1:]]
            cofac = ((-1) ** c) * mat[0][c] * _det(sub)
            total += cofac
        return total

    # 4) Compute cofactor matrix
    cofactor_mat = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            # build submatrix by excluding row i, column j
            submatrix = [
                matrix[r][:j] + matrix[r][j + 1 :]
                for r in range(n) if r != i
            ]
            minor_det = _det(submatrix)
            cofactor_value = ((-1) ** (i + j)) * minor_det
            row_cof.append(cofactor_value)
        cofactor_mat.append(row_cof)

    return cofactor_mat
