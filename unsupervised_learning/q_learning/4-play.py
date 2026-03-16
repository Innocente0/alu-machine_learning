#!/usr/bin/env python3
"""Play an episode using a trained Q-table"""

import time
import numpy as np


def play(env, Q, max_steps=100):
    """has the trained agent play an episode"""
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    total_reward = 0

    for _ in range(max_steps):
        env.render()
        action = np.argmax(Q[state])

        step = env.step(action)
        if len(step) == 5:
            new_state, reward, terminated, truncated, _ = step
            done = terminated or truncated
        else:
            new_state, reward, done, _ = step

        total_reward += reward
        state = new_state

        if done:
            env.render()
            break

        time.sleep(0.5)

    return total_reward
