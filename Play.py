import gym
import logging
import gym_mtd
import numpy as np
import gym
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tqdm import tqdm

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)

steps = 1000
episodes = 50

env = gym.make('MTD-v0', time_limit=steps, utenv=1, setting=1, ca=0)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape

model = Sequential()
model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
logging.info(model.summary())

memory = SequentialMemory(limit=128, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=128,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


if os.path.isfile(f'duel_dqn_attacker_weights.h5f'):
    dqn.load_weights(f'duel_dqn_attacker_weights.h5f')

dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

dqn.save_weights(f'duel_dqn_attacker_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=1, visualize=False)
