#!/usr/bin/env python3
"""Defines a single neuron for binary classification (initialization only)."""

import numpy as np
class Neuron:
    """Single neuron performing binary classification (parameters only)."""

    def __init__(self, nx):
        """
        nx: number of input features.

        Raises
        ------
        TypeError: if nx is not an integer.
        ValueError: if nx < 1.
        """
        # 1) Validate nx (order is important)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2) Initialize parameters
        # Weights: row vector (1, nx) from standard normal distribution
        self.W = np.random.randn(1, nx)

        # Bias: scalar 0
        self.b = 0

        # Activated output: start at 0 (scalar)
        self.A = 0
