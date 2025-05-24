#!/usr/bin/env python3
"""
module: 2-size_me_please

Provides a single function, matrix_shape(matrix),
which returns the dimensions of a nested list (matrix)
as a list of integers.
"""


def matrix_shape(matrix):
    shape = []
    current = matrix
    while isinstance(current, list):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]
    return shape


if __name__ == "__main__": 
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))

    mat2 = [
        [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20],
         [21, 22, 23, 24, 25],
         [26, 27, 28, 29, 30]]
    ]
    print(matrix_shape(mat2))
