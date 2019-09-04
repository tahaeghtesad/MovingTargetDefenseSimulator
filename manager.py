import gym
import gym_mtd
from agents.attackers import *
from agents.defenders import *
from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor

steps = 100000
m=10

env = gym.make('MTD-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2)

ap = AttackerProcessor(m=m)
dp = DefenderProcessor(m=m)

attacker = UniformAttacker(m=m)
defender = PCPDefender(m=m)

done = False
obs = env.reset()

a_obs = ap.process_observation(obs)
d_obs = dp.process_observation(obs)

au = 0
du = 0

while not done:

    obs, reward, done, info = env.step((attacker.predict(a_obs), defender.predict(d_obs)))

    a_obs, a_r, a_d, a_i = ap.process_step(obs, reward, done, info)
    d_obs, d_r, d_d, d_i = dp.process_step(obs, reward, done, info)

    au += a_r
    du += d_r

print(f'Attacker/Defender: {au/steps:.4f}/{du/steps:.4f}')