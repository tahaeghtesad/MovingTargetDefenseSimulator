import gym
import gym_mtd
from agents.attackers import *
from agents.defenders import *
from tqdm import tqdm
from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor
import matplotlib.pyplot as plt
import numpy as np

steps = 1000
episodes = 10
m = 10


def play(attacker, defender):
    env = gym.make('MTD-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2)

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
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]

payoff_table = np.zeros((len(attackers), len(defenders), 2))

for i, a in enumerate(attackers):
    for j, d in enumerate(defenders):
        payoff_table[i, j] = play(a(), d())


print('\\backslashbox{Attackers}{Defenders} ', end='')
for d in defenders:
    print(f'& {d.__name__.replace("Defender", "")} ', end='')
print('\\\\\n\\hline')


for i, a in enumerate(attackers):
    print(f'{a.__name__.replace("Attacker", "")} ', end='')
    for j, d in enumerate(defenders):
        ar, dr = payoff_table[i, j]
        print('& ', end='')
        if is_equilibrium(payoff_table, i, j):
            print('\\cellcolor[gray]{0.8} ', end='')
        print(f'\\backslashbox{{{ar:.4f}}}{{{dr:.4f}}} ', end='')
    print('\\\\\n\\hline')
