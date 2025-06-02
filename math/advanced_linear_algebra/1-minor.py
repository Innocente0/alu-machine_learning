#!/usr/bin/env python3
"""
module: 1-minor

Provides a function `minor(matrix)` that returns the minor matrix of
a non‐empty square matrix represented as a list of lists.
Raises TypeError or ValueError for invalid inputs.
"""


def minor(matrix):
    """
    Calculate and return the minor matrix of `matrix`.

    - If `matrix` is not a list of lists, raise:
        TypeError("matrix must be a list of lists")
    - If `matrix` is empty or not square, raise:
        ValueError("matrix must be a non-empty square matrix")
    - Otherwise, return a new list of lists where each entry [i][j]
      is the determinant of the submatrix formed by deleting row i
      and column j from the original.
    """
    # 1) Check that `matrix` is a list of lists
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2) Ensure it's non-empty and square
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # 3) Handle 1×1 matrix: minor is [[1]]
    if n == 1:
        return [[1]]

    # 4) Helper: compute determinant of a square (list of lists)
    def _det(mat):
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det_val = 0
        for c in range(size):
            # build sub‐minor
            sub = [
                row[:c] + row[c+1 :]
                for row in mat[1:]
            ]
            cofac = ((-1) ** c) * mat[0][c] * _det(sub)
            det_val += cofac
        return det_val

    # 5) Build the minor matrix of size n×n
    minor_mat = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # build the submatrix by removing row i, column j
            submatrix = [
                matrix[r][:j] + matrix[r][j+1 :]
                for r in range(n) if r != i
            ]
            minor_entry = _det(submatrix)
            minor_row.append(minor_entry)
        minor_mat.append(minor_row)

    return minor_mat
