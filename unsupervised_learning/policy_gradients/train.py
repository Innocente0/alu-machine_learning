#!/usr/bin/env python3
"""Train a policy-gradient agent"""

import numpy as np


def softmax(z):
    """computes softmax"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy(matrix, weight):
    """computes the policy with a weight of a matrix"""
    return softmax(np.matmul(matrix, weight))


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient"""
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    dsoftmax = -probs
    dsoftmax[0, action] += 1
    grad = np.matmul(state.T, dsoftmax)

    return action, grad


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """implements a full training"""
    weights = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state = reset_out[0][None, :]
        else:
            state = reset_out[None, :]

        episode_rewards = []
        episode_grads = []
        score = 0

        done = False
        while not done:
            action, grad = policy_gradient(state, weights)

            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            episode_rewards.append(reward)
            episode_grads.append(grad)
            score += reward
            state = next_state[None, :]

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score),
              end="\r", flush=False)

        for i, grad in enumerate(episode_grads):
            discount = 0
            for j, reward in enumerate(episode_rewards[i:]):
                discount += reward * (gamma ** j)
            weights += alpha * grad * discount

    print()
    return scores
