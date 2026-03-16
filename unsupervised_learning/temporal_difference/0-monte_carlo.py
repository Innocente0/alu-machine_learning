#!/usr/bin/env python3
"""Monte Carlo algorithm"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """performs the Monte Carlo algorithm"""
    for _ in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = new_state
            if done:
                break

        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = gamma * G + reward

            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
