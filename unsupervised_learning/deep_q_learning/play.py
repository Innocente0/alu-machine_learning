#!/usr/bin/env python3
"""Play Breakout using a trained DQN policy network"""

import gym
import numpy as np
from tensorflow.keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
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


def make_env():
    """Create a Breakout environment"""
    env_names = [
        "Breakout-v0",
        "ALE/Breakout-v5",
        "BreakoutNoFrameskip-v4",
    ]

    for name in env_names:
        try:
            env = gym.make(name, render_mode="human")
            return env
        except TypeError:
            try:
                env = gym.make(name)
                return env
            except Exception:
                continue
        except Exception:
            continue
    raise ValueError("Could not load a Breakout environment")


if __name__ == "__main__":
    env = make_env()
    np.random.seed(0)

    nb_actions = env.action_space.n
    model = load_model("policy.h5")

    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=0,
        target_model_update=1e-2,
        policy=policy,
        processor=processor,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type="avg",
    )

    dqn.compile(optimizer="adam", metrics=["mae"])
    dqn.test(env, nb_episodes=1, visualize=True)
    