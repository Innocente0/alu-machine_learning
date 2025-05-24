#!/usr/bin/env python3

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
    # demo / manual test code only runs when you do `python3 2-main.py`
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))         # prints [2, 2]

    mat2 = [
      [[1, 2, 3, 4, 5],
       [6, 7, 8, 9, 10],
       [11, 12, 13, 14, 15]],
      [[16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25],
       [26, 27, 28, 29, 30]]
    ]
    print(matrix_shape(mat2))         # prints [2, 3, 5]
