import sys
import os
import traceback

import gym
import gym_mtd
import logging
import stable_baselines
import tensorflow as tf
from stable_baselines.deepq.policies import MlpPolicy

from agents.defenders import UniformDefender
from agents.attackers import UniformAttacker

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
steps = 1000
episodes = 300

rootLogger.setLevel(logging.INFO if debug is False else logging.DEBUG)

attacker_env = gym.make('MTDAtt-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.0, alpha=.05,
               defender=UniformDefender())

defender_env = gym.make('MTDDef-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.0, alpha=.05,
               attacker=UniformAttacker())

weight_path = 'weights.pkl'


class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         act_fun=tf.nn.sigmoid,
                         layers=[256, 128],
                         dueling=False,
                         )


attacker_model = stable_baselines.DQN(
    policy=CustomPolicy,
    env=attacker_env,
    verbose=2,
    tensorboard_log='tb_logs',
    full_tensorboard_log=True
)

defender_model = stable_baselines.DQN(
    policy=CustomPolicy,
    env=defender_env,
    verbose=2,
    tensorboard_log='tb_logs',
    full_tensorboard_log=True
)

if os.path.isfile(f'defender_{weight_path}') and os.path.isfile(f'attacker_{weight_path}'):
    print('Loading weight file...')
    attacker_model.load(f'attacker_{weight_path}', attacker_env)
    defender_model.load(f'defender_{weight_path}', defender_env)

try:
    defender_model.learn(
        total_timesteps=episodes * steps,
        callback=None,
        log_interval=1,
        tb_log_name=f'DQN_defender'
    )

    attacker_model.learn(
        total_timesteps=episodes * steps,
        callback=None,
        log_interval=1,
        tb_log_name=f'DQN_attacker'
    )

except KeyboardInterrupt:
    pass

except Exception:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

finally:
    print('Saving weight file...')
    attacker_model.save(f'attacker_{weight_path}')
    defender_model.save(f'defender_{weight_path}')
