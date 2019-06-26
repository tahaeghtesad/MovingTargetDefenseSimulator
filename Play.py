import sys
import os
import traceback

import gym
import gym_mtd
import stable_baselines
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from Defenders import *

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

debug = False
m = 10
steps = 200
episodes = 1000

rootLogger.setLevel(logging.INFO if debug is False else logging.DEBUG)

env = DummyVecEnv([lambda: gym.make('MTDAtt-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.0, alpha=.05,
               defender=UniformDefender())])

weight_path = 'attacker.pkl'


class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         act_fun=tf.nn.sigmoid,
                         layers=[256, 128],
                         layer_norm=False,
                         dueling=False,
                         feature_extraction="mlp"
                         )


model = stable_baselines.DQN(
    policy=CustomPolicy,
    env=env,
    # gamma=.9,
    # learning_rate=5e-3,
    # buffer_size=steps,
    # exploration_fraction=.3,
    # exploration_final_eps=.5,
    # train_freq=steps,
    # batch_size=steps,
    # checkpoint_freq=10000,
    # checkpoint_path=None,
    # learning_starts=1000,
    # target_network_update_freq=steps,
    # prioritized_replay=True,
    # param_noise=False,
    verbose=2,
    tensorboard_log='tb_logs',
    full_tensorboard_log=True
)

# if os.path.isfile(weight_path):
#     print('Loading weight file...')
#     model.load(weight_path, env)

try:
    model.learn(
        total_timesteps=episodes * steps,
        callback=None,
        log_interval=1,
        tb_log_name=f'DQN_{model.gamma:.2f}_{model.exploration_fraction:.2f}_{model.exploration_final_eps:.2f}_{model.learning_rate:.5f}_{model.prioritized_replay}_{False}'
    )

except KeyboardInterrupt:
    pass

except Exception:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

finally:
    print('Saving weight file...')
    model.save(weight_path)
