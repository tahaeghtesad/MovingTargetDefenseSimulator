import gym
import gym_mtd
from util.DQNWrapper import DQNWrapper
from agents.attackers import *
from agents.defenders import *
from tqdm import tqdm
from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor
import matplotlib.pyplot as plt

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


attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]


print('\\backslashbox{Attackers}{Defenders} ', end='')
for d in defenders:
    print(f'& {d.__name__.replace("Defender", "")} ', end='')
print('\\\\\n\\hline')


for a in attackers:
    print(f'{a.__name__.replace("Attacker", "")} ', end='')
    for d in defenders:
        ar, dr = play(a(), d())
        print(f'& \\backslashbox{{{ar:.4f}}}{{{dr:.4f}}} ', end='')
    print('\\\\\n\\hline')
