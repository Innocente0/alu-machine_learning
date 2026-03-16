#!/usr/bin/env python3
"""Loads a FrozenLake environment"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the FrozenLakeEnv environment"""
    if desc is None and map_name is None:
        map_name = "8x8"

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
