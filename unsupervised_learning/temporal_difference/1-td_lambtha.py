#!/usr/bin/env python3
"""TD(lambda) algorithm"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """performs the TD(lambda) algorithm"""
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            step = env.step(action)

            if len(step) == 5:
                next_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V = V + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = next_state
            if done:
                break

    return V
