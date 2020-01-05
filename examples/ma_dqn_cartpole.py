import argparse

import numpy as np
import gym
# Import multi-environment tests
import ma_gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.multi import MultiAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# Original cartpole for testing
#ENV_NAME = 'ma_CartPole-v0'
#AGENT_COUNT = 1

# Real PongDuel enviroments
#ENV_NAME = 'PongDuel-v0'
ENV_NAME = 'Checkers-v0'
AGENT_COUNT = 2

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
# NOTE: takeing first item of the action space since this is an array
nb_actions = env.action_space[0].n


def setup_agent():
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space[0].shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    return dqn


def setup_multi_agent():
    agents = [
        setup_agent()
        for _ in range(AGENT_COUNT)
    ]
    multi_agent = MultiAgent(agents)
    multi_agent.compile(Adam(lr=1e-3), metrics=['mae'])

    return multi_agent


def learn_evaluate_agent(multi_agent):
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    multi_agent.fit(env, nb_steps=50000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    multi_agent.save_weights(
        'ma_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    multi_agent.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running the agent')
    parser.add_argument('--pdb', action='store_true',
                        help='Trigger pdb on error')
    args = parser.parse_args()

    try:
        multi_agent = setup_multi_agent()
        learn_evaluate_agent(multi_agent)
    except:  # noqa: E722
        import pdb
        import traceback
        import sys
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        if args.pdb:
            pdb.post_mortem(tb)
