#!/usr/bin/env python3
def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.
    
    Args:
        matrix: A list of lists representing the matrix
    
    Returns:
        The cofactor matrix as a list of lists
    
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not a non-empty square matrix
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is empty
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Check if all elements are lists
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is square
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")
    
    # Helper function to calculate determinant
    def determinant(mat):
        size = len(mat)
        
        # Base case for 1x1 matrix
        if size == 1:
            return mat[0][0]
        
        # Base case for 2x2 matrix
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        # Recursive case for larger matrices
        det = 0
        for col in range(size):
            # Create minor matrix by removing first row and current column
            minor = []
            for i in range(1, size):
                minor_row = []
                for j in range(size):
                    if j != col:
                        minor_row.append(mat[i][j])
                minor.append(minor_row)
            
            # Calculate cofactor and add to determinant
            cofactor_val = ((-1) ** col) * mat[0][col] * determinant(minor)
            det += cofactor_val
        
        return det
    
    # Helper function to get minor matrix by removing row i and column j
    def get_minor(mat, row, col):
        minor = []
        for i in range(len(mat)):
            if i != row:
                minor_row = []
                for j in range(len(mat[i])):
                    if j != col:
                        minor_row.append(mat[i][j])
                minor.append(minor_row)
        return minor
    
    # Calculate cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Get minor matrix by removing row i and column j
            minor = get_minor(matrix, i, j)
            
            # Calculate cofactor: (-1)^(i+j) * determinant of minor
            if len(minor) == 0:  # For 1x1 matrix, minor is empty
                cofactor_val = 1
            else:
                cofactor_val = ((-1) ** (i + j)) * determinant(minor)
            
            cofactor_row.append(cofactor_val)
        cofactor_matrix.append(cofactor_row)
    
    return cofactor_matrix
