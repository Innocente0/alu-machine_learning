#!/usr/bin/env python3
def matrix_transpose(matrix):
    """
    Return a new matrix which is the transpose of the 2D list `matrix`.
    Transpose means rows become columns and columns become rows.
    """
    # Number of rows in the input becomes number of columns in the output
    num_rows = len(matrix)
    # Number of columns in the input (we assume at least one row exists)
    num_cols = len(matrix[0])

    # Build the transposed matrix row by row
    transposed = []
    for j in range(num_cols):            # for each column index j in the original
        new_row = []
        for i in range(num_rows):        # collect the element at (i, j) from each row i
            new_row.append(matrix[i][j])
        transposed.append(new_row)

    return transposed
