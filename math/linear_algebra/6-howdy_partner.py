#!/usr/bin/env python3
"""
module: 6-howdy_partner

Provides cat_arrays(arr1, arr2), which returns a new list containing
all elements of arr1 followed by all elements of arr2.
"""


def cat_arrays(arr1, arr2):
    """
    Concatenate two 1-D lists (arrays) and return the combined list.
    """
    # 1) Create a new empty list
    result = []
    # 2) Append each element from the first list
    for x in arr1:
        result.append(x)
    # 3) Then append each element from the second list
    for x in arr2:
        result.append(x)
    # 4) Return the concatenated result
    return result
