import numpy as np
import random

from rl.core import Env


class MultiInputTestEnv(Env):
    def __init__(self, observation_shape):
        self.observation_shape = observation_shape

    def step(self, action):
        return self._get_obs(), random.choice([0, 1]), random.choice([True, False]), {}

    def reset(self):
        return self._get_obs()

    def _get_obs(self):
        if type(self.observation_shape) is list:
            return [np.random.random(s) for s in self.observation_shape]
        else:
            return np.random.random(self.observation_shape)

    def __del__(self):
        pass


class MultiAgentTestEnv(Env):
    def __init__(self, observation_shape, num_agents=3):
        self.observation_shape = observation_shape
        self.num_agents = num_agents

    def step(self, action):
        assert isinstance(action, list), "MultiAgentEnv takes list of actions"
        return (
            [self._get_obs()] * self.num_agents,  # observations
            [random.choice([0, 1])] * self.num_agents,  # rewards
            [random.choice([True, False])] * self.num_agents,  # terminal
            {},   # info
        )

    def reset(self):
        return [self._get_obs()] * self.num_agents

    def _get_obs(self):
        if type(self.observation_shape) is list:
            return [np.random.random(s) for s in self.observation_shape]
        else:
            return np.random.random(self.observation_shape)

    def __del__(self):
        pass
