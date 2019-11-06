import gym
import gym_mtd
from agents.attackers import *
from agents.defenders import *
from tqdm import tqdm
from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import Pool

steps = 1000
episodes = 10
m = 10


def play(attacker, defender):
    env = gym.make('MTD-v0', m=m, time_limit=steps, utenv=2, setting=0, ca=0.05)

    ap = AttackerProcessor(m=m)
    dp = DefenderProcessor(m=m)

    au = 0
    du = 0

    aul = list()
    dul = list()
    iter = 0

    for i in range(episodes):

        done = False
        obs = env.reset()

        a_obs = ap.process_observation(obs)
        d_obs = dp.process_observation(obs)

        while not done:
            obs, reward, done, info = env.step((
                ap.process_action(attacker.predict(a_obs)),
                dp.process_action(defender.predict(d_obs))
            ))

            a_obs, a_r, a_d, a_i = ap.process_step(obs, reward, done, info)
            d_obs, d_r, d_d, d_i = dp.process_step(obs, reward, done, info)

            au += a_r
            du += d_r

            # obs, reward, done, info = env.step(attacker.predict(obs))
            aul.append((iter, a_r))
            dul.append((iter, d_r))
            iter += 1

    # print(f'{attacker.__class__.__name__}/{defender.__class__.__name__}: {au / steps / episodes:.4f}/{du / steps / episodes:.4f}')
    return au / steps / episodes, du / steps / episodes


def is_equilibrium(arr, x, y) -> bool:
    if np.max([p[1] for p in arr[x, :]]) == arr[x][y][1]:
        if np.max([p[0] for p in arr[:, y]]) == arr[x][y][0]:
            return True
    return False


attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
len_attackers = sum([sum(1 for _ in p) for p in [g.gen_configurations() for g in [a for a in attackers]]])
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]
len_defenders = sum([sum(1 for _ in p) for p in [g.gen_configurations() for g in [d for d in defenders]]])

print(f'Attackers: {len_attackers}/Defenders: {len_defenders}')

payoff_table = np.zeros((len_attackers, len_defenders, 2))

for i, a in enumerate(attackers):
    for j, d in enumerate(defenders):
        payoff_table[i, j] = play(a(), d())


def run(params):
    i, j, a, d = params
    return play(a, d)

def gen_games():
    i = 0
    for a in attackers:
        for ag in a.gen_configurations():
            j = 0
            for d in defenders:
                for dg in d.gen_configurations():
                    # print(f'{i}/{j}', end=' ')
                    yield i, j, ag, dg
                    j += 1
            i += 1
            # print()


responses = Pool(8).map(run, gen_games())
index = 0
for i in range(len_attackers):
    for j in range(len_defenders):
        payoff_table[i, j] = responses[index]
        index += 1

np.save('payoff', payoff_table)