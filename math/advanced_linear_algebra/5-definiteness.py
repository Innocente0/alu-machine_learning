#!/usr/bin/env python3
"""
Module for calculating matrix definiteness.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n) whose definiteness should be
                calculated

    Returns:
        String indicating the definiteness: 'Positive definite',
        'Positive semi-definite', 'Negative semi-definite',
        'Negative definite', 'Indefinite', or None

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is valid (not empty and 2D)
    if matrix.size == 0 or matrix.ndim != 2:
        return None

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric (required for definiteness)
    # Use allclose to handle floating point precision issues
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)

        # Handle complex eigenvalues (shouldn't happen for symmetric
        # matrices, but just in case)
        if np.any(np.iscomplex(eigenvalues)):
            return None

        # Convert to real values to avoid any complex number issues
        eigenvalues = np.real(eigenvalues)

        # Define tolerance for zero comparison
        tol = 1e-8

        # Count positive, negative, and zero eigenvalues
        positive = np.sum(eigenvalues > tol)
        negative = np.sum(eigenvalues < -tol)
        zero = np.sum(np.abs(eigenvalues) <= tol)

        n = len(eigenvalues)

        # Determine definiteness based on eigenvalues
        if positive == n:
            return "Positive definite"
        elif positive + zero == n and zero > 0:
            return "Positive semi-definite"
        elif negative == n:
            return "Negative definite"
        elif negative + zero == n and zero > 0:
            return "Negative semi-definite"
        elif positive > 0 and negative > 0:
            return "Indefinite"
        else:
            return None

    except np.linalg.LinAlgError:
        # If eigenvalue computation fails
        return None
    