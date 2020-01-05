from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Concatenate

from rl.agents.ddpg import DDPGAgent
from rl.agents.multi import MultiAgent
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

from ..util import MultiAgentTestEnv


def test_single_mult_agent():
    num_agents = 3
    test_env = MultiAgentTestEnv((3,), num_agents)

    nb_actions = 2

    agents = []

    for _ in range(num_agents):
        actor = Sequential()
        actor.add(Flatten(input_shape=(2, 3)))
        actor.add(Dense(nb_actions))

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(2, 3), name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(1)(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

        memory = SequentialMemory(limit=10, window_length=2)
        agents.append(
            DDPGAgent(
                actor=actor, critic=critic,
                critic_action_input=action_input, memory=memory,
                nb_actions=2, nb_steps_warmup_critic=5,
                nb_steps_warmup_actor=5, batch_size=4)
        )

    multi_agent = MultiAgent(agents)
    multi_agent.compile('sgd')
    multi_agent.fit(test_env, nb_steps=10)
