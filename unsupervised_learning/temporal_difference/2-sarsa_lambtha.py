#!/usr/bin/env python3
"""SARSA(lambda)"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """performs SARSA(lambda)"""
    def epsilon_greedy(state, eps):
        """selects an action using epsilon-greedy"""
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(Q.shape[1])
        return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        action = epsilon_greedy(state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            step = env.step(action)

            if len(step) == 5:
                next_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step

            next_action = epsilon_greedy(next_state, epsilon)

            next_q = 0 if done else Q[next_state, next_action]
            delta = reward + gamma * next_q - Q[state, action]

            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

            if done:
                break

        epsilon = min_epsilon + (
            epsilon - min_epsilon
        ) * np.exp(-epsilon_decay)

    return Q
