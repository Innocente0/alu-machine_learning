#!/usr/bin/env python3
"""Initializes the Q-table"""

import numpy as np


def q_init(env):
    """returns a Q-table initialized to zero"""
    return np.zeros((env.observation_space.n, env.action_space.n))
