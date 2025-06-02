#!/usr/bin/env python3
"""
module: 0-determinant

Provides a function `determinant(matrix)` that returns the determinant
of a square matrix (list of lists). Raises errors for invalid input.
"""


def determinant(matrix):
    """
    Calculate and return the determinant of `matrix`.

    - If `matrix` is not a list of lists, raise TypeError("matrix must be a list of lists").
    - If `matrix` is not square, raise ValueError("matrix must be a square matrix").
    - `[[]]` is treated as 0×0, whose determinant is 1.
    """
    # 1) Ensure matrix is a list of lists
    if not isinstance(matrix, list) or any(
        not isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    # 2) 0×0 matrix case: `[[]]`
    if matrix == [[]]:
        return 1

    n = len(matrix)
    # 3) Check squareness: each row must have length n
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # 4) Base case for 1×1
    if n == 1:
        return matrix[0][0]

    # 5) Base case for 2×2
    if n == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    # 6) Recursive expansion by first row
    det = 0
    for col in range(n):
        # Build minor by removing row 0 and column col
        minor = []
        for r in range(1, n):
            row_minor = matrix[r][:col] + matrix[r][col + 1:]
            minor.append(row_minor)

        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor)
        det += cofactor

    return det