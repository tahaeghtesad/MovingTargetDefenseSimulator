import gym
import logging
import gym_mtd
import numpy as np
import gym
import os
import sys
import traceback

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from rl.agents.dqn import *
from rl.policy import *
from rl.memory import *
from rl.processors import *
from rl.callbacks import *

from tqdm import tqdm
from BaseDefender import BaseDefender
from Defenders import *
from keras_rl_util.AttackerProcessor import AttackerProcessor
from keras_rl_util.EpsDecayGreedyQPolicy import EpsDecayGreedyQPolicy

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

np.random.seed(1)
random.seed(1)

debug = False
m = 10
steps = 1000
episodes = 100

rootLogger.setLevel(logging.INFO if debug is False else logging.DEBUG)

env = gym.make('MTD-v0', m=m, time_limit=sys.maxsize, utenv=0, setting=1, ca=0.2, defender=BaseDefender())

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
model.add(Dense(256, activation='sigmoid'))
# model.add(Activation())
model.add(Dense(64, activation='elu'))
# model.add(Activation('elu'))
model.add(Dense(nb_actions))
logging.info(model.summary())

memory = SequentialMemory(limit=128,
                          window_length=1,
                          ignore_episode_boundaries=False)


# policy = EpsDecayGreedyQPolicy(eps=1., decay=.98, step_decay=steps)
# policy = GreedyQPolicy()
policy = EpsGreedyQPolicy(eps=.1)

processor = AttackerProcessor(m=m)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               # nb_steps_warmup=1 * steps,
               nb_steps_warmup=128,
               enable_dueling_network=False,
               enable_double_dqn=False,
               dueling_type='avg',
               target_model_update=1,
               train_interval=1,
               policy=policy,
               gamma=.97,
               processor=processor,
               batch_size=32)

dqn.compile(Adam(lr=1e-3),
            metrics=['mse'])

if os.path.isfile(f'duel_dqn_attacker_weights.h5f'):
    dqn.load_weights(f'duel_dqn_attacker_weights.h5f')

try:
    dqn.fit(env,
            nb_steps=steps * episodes,
            nb_max_episode_steps=steps,
            log_interval=steps,
            visualize=debug,
            verbose=2)

except Exception:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5, file=sys.stderr)

finally:
    dqn.save_weights(f'duel_dqn_attacker_weights.h5f', overwrite=True)
    dqn.test(env,
             nb_episodes=4,
             nb_max_episode_steps=steps,
             verbose=2,
             visualize=debug)
