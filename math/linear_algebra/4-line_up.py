#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    """
    Add two 1-D arrays (lists) element-wise and return a new list.
    If the lists have different lengths, return None.
    """
    # 1) Check that they’re the same length
    if len(arr1) != len(arr2):
        return None

    # 2) Build a new list by summing corresponding elements
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
