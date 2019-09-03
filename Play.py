import sys
import os
import traceback
import uuid
import csv

import gym
import gym_mtd
import logging
import stable_baselines
import tensorflow as tf
from stable_baselines.deepq.policies import *
from stable_baselines.common.vec_env import *

from agents.defenders import *
from agents.attackers import *

id = uuid.uuid4()

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)


fileHandler = logging.FileHandler(f'logs/log_{str(id).replace("-","_")}.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

training_mode = sys.argv[1] == 'attacker'
print(f'Training Mode: {"Attacker" if training_mode else "Defender"}')

csv_row = [f'log_{str(id).replace("-","_")}.log', 'Attacker' if training_mode else 'Defender']

debug = False
m = 10
steps = 5000
episodes = 1000

rootLogger.setLevel(logging.INFO if debug is False else logging.DEBUG)

if training_mode:
    env = gym.make('MTDAtt-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2,
                   defender=PCPDefender())

else:
    env = gym.make('MTDDef-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2,
                   attacker=UniformAttacker())

params = env.config
csv_row += [params['opponent'], '', episodes, params['m'], params['delta'], params['alpha'], params['T'], params['nu'], params['utenv'], params['setting'], params['c_a'], '']

weight_path = 'weights.pkl'


def get_params_dqn(dqn_model: stable_baselines.DQN):
    return {
        "checkpoint_path": dqn_model.checkpoint_path,
        "param_noise": dqn_model.param_noise,
        "learning_starts": dqn_model.learning_starts,
        "train_freq": dqn_model.train_freq,
        "prioritized_replay": dqn_model.prioritized_replay,
        "prioritized_replay_eps": dqn_model.prioritized_replay_eps,
        "batch_size": dqn_model.batch_size,
        "target_network_update_freq": dqn_model.target_network_update_freq,
        "checkpoint_freq": dqn_model.checkpoint_freq,
        "prioritized_replay_alpha": dqn_model.prioritized_replay_alpha,
        "prioritized_replay_beta0": dqn_model.prioritized_replay_beta0,
        "prioritized_replay_beta_iters": dqn_model.prioritized_replay_beta_iters,
        "exploration_final_eps": dqn_model.exploration_final_eps,
        "exploration_fraction": dqn_model.exploration_fraction,
        "learning_rate": dqn_model.learning_rate,
        "gamma": dqn_model.gamma,
        "verbose": dqn_model.verbose,
        "observation_space": dqn_model.observation_space,
        "action_space": dqn_model.action_space,
        "policy": dqn_model.policy,
        "n_envs": dqn_model.n_envs,
        "_vectorize_action": dqn_model._vectorize_action,
        "policy_kwargs": dqn_model.policy_kwargs
    }

attacker_policy = {
    'activation': tf.nn.tanh,
    'layers': [256, 128],
    'dueling': False
}

defender_policy = {
    'activation': tf.nn.tanh,
    'layers': [256, 128],
    'dueling': False
}

if training_mode:
    csv_row += [attacker_policy['layers'], attacker_policy['activation'], attacker_policy['dueling'], '']
else:
    csv_row += [defender_policy['layers'], defender_policy['activation'], defender_policy['dueling'], '']


class CustomAttackerPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         act_fun=attacker_policy['activation'],
                         layers=attacker_policy['layers'],
                         dueling=attacker_policy['dueling'],
                         )

        rootLogger.info(attacker_policy)


class CustomDefenderPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs,
                         act_fun=defender_policy['activation'],
                         layers=defender_policy['layers'],
                         dueling=defender_policy['dueling'],
                         )

        rootLogger.info(defender_policy)

if training_mode:
    attacker_model = stable_baselines.DQN(
        policy=CustomAttackerPolicy,
        env=DummyVecEnv([lambda: env]),
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=2,
        tensorboard_log='tb_logs',
        full_tensorboard_log=True
    )
    dqn_params = get_params_dqn(attacker_model)
    rootLogger.info(f'Initializing Attack Learner:\n' +
                    f'{dqn_params}')
else:
    defender_model = stable_baselines.DQN(
        policy=CustomDefenderPolicy,
        env=DummyVecEnv([lambda: env]),
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=2,
        tensorboard_log='tb_logs',
        full_tensorboard_log=True
    )
    dqn_params = get_params_dqn(defender_model)
    rootLogger.info(f'Initializing Defense Learner:\n' +
                    f'{dqn_params}')

csv_row += [dqn_params['prioritized_replay'], dqn_params['exploration_final_eps'],
            dqn_params['exploration_fraction'], '', dqn_params['learning_rate'], dqn_params['gamma']]

# if os.path.isfile(f'defender_{weight_path}') and os.path.isfile(f'attacker_{weight_path}'):
#     print('Loading weight file...')
#     attacker_model.load(f'attacker_{weight_path}', attacker_env)
#     defender_model.load(f'defender_{weight_path}', defender_env)

with open('Reports.csv', 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(csv_row)

def callback(locals_, globals_):
    self_ = locals_['self']

    if 'action' in locals_:
        summary = tf.Summary(value=[tf.Summary.Value(tag='game/actions', simple_value=locals_['action'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

    if 'update_eps' in locals_:
        summary = tf.Summary(value=[tf.Summary.Value(tag='input_info/eps', simple_value=locals_['update_eps'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

    if 'info' in locals_:
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/attacker_reward', simple_value=locals_['info']['rewards']['att'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/defender_reward', simple_value=locals_['info']['rewards']['def'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

    return True


try:
    rootLogger.info(f'Begin training for {episodes} episodes/{steps} steps')

    if training_mode:
        env.learn(
            total_timesteps=episodes * steps,
            callback=callback,
            log_interval=1,
            tb_log_name=f'DQN_attacker_{str(id).split("-")[0]}'
        )
    else:
        env.learn(
            total_timesteps=episodes * steps,
            callback=callback,
            log_interval=1,
            tb_log_name=f'DQN_defender_{str(id).split("-")[0]}'
        )

except KeyboardInterrupt:
    print('Gracefully exiting...')

except Exception:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

finally:
    print('Saving weight file...')
    if training_mode:
        env.save(f'weights/attacker_{str(id).split("-")[0]}_{weight_path}')
    else:
        env.save(f'weights/defender_{str(id).split("-")[0]}_{weight_path}')
