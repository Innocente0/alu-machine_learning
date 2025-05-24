def matrix_shape(matrix):
    """
    Returns a list of the sizes of each dimension of `matrix`.
    E.g. [[1,2],[3,4]] → [2,2]; 
          [[[...]]] → [2,3,5]
    """
    # Base case: if it isn’t a list, there are no more dimensions
    if not isinstance(matrix, list):
        return []
    # First dimension size is how many items are in this list
    size = len(matrix)
    # Recurse into the first element to discover deeper dimensions
    return [size] + matrix_shape(matrix[0])


# Examples
mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))   # → [2, 2]

mat2 = [
    [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]],
    [[16,17,18,19,20], [21,22,23,24,25], [26,27,28,29,30]]
]
print(matrix_shape(mat2))   # → [2, 3, 5]
