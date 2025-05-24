#!/usr/bin/env python3
"""
module: 4-line_up

Provides add_arrays(arr1, arr2) which returns a new list containing the
element-wise sums of two equally sized lists, or None if their lengths differ.
"""


def add_arrays(arr1, arr2):
    """
    Add two 1-D arrays (lists) element-wise.
    Return a new list of sums, or None if lengths don’t match.
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
