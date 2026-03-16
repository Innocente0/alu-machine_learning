#!/usr/bin/env python3
"""Q-learning"""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning"""
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            step = env.step(action)

            if len(step) == 5:
                new_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                new_state, reward, done, _ = step

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            episode_reward += reward
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay
        )
        total_rewards.append(episode_reward)

    return Q, total_rewards
