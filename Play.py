import gym
import logging
import gym_mtd
import numpy as np
import gym
import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from rl.agents.dqn import *
from rl.policy import *
from rl.memory import *

from tqdm import tqdm

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

debug = False
steps = 8000
episodes = 10

rootLogger.setLevel(logging.INFO if debug is False else logging.DEBUG)

env = gym.make('MTD-v0', time_limit=steps, utenv=1, setting=1, ca=0)

# for i in range(episodes):
#     for j in range(steps):
#         action = env.action_space.sample()
#         observ, reward, ended, info = env.step(action)
#         print(f'{observ, reward, action, ended}')
#     env.reset()
#     break

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape

model = Sequential()
model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
logging.info(model.summary())

memory = SequentialMemory(limit=1024, window_length=1)
policy = EpsGreedyQPolicy(eps=.1)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=128,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


if os.path.isfile(f'duel_dqn_attacker_weights.h5f'):
    dqn.load_weights(f'duel_dqn_attacker_weights.h5f')

dqn.fit(env, nb_steps=steps * episodes, log_interval=steps, visualize=debug, verbose=2)

dqn.save_weights(f'duel_dqn_attacker_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=2, visualize=False)
