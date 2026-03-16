#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout"""

import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.core import Processor


class AtariProcessor(Processor):
    """Processor for Atari observations"""

    def process_observation(self, observation):
        """Convert observation to uint8"""
        return observation.astype("uint8")

    def process_state_batch(self, batch):
        """Normalize state batch"""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip reward"""
        return np.clip(reward, -1., 1.)


def build_model(height, width, channels, actions):
    """Build the policy network"""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(channels, height, width)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def make_env():
    """Create a Breakout environment"""
    env_names = [
        "Breakout-v0",
        "ALE/Breakout-v5",
        "BreakoutNoFrameskip-v4",
    ]

    for name in env_names:
        try:
            env = gym.make(name)
            return env
        except Exception:
            continue
    raise ValueError("Could not load a Breakout environment")


if __name__ == "__main__":
    env = make_env()
    np.random.seed(0)
    env.seed(0)

    nb_actions = env.action_space.n
    height, width, channels = env.observation_space.shape

    model = build_model(height, width, channels, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = EpsGreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=5000,
        target_model_update=1e-2,
        policy=policy,
        processor=processor,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type="avg",
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.fit(
        env,
        nb_steps=50000,
        visualize=False,
        verbose=2,
    )

    dqn.model.save("policy.h5")
    