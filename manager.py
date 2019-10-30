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
episodes = 1
m = 10

plt.figure(figsize=(16, 12))


def smooth(series, weight):
    ret = [sum(series[0:200])/200]
    smoothed = sum(series[0:200])/200
    for i in range(1, len(series)):
        smoothed = smoothed * weight + (1-weight) * series[i]
        ret.append(smoothed)
    return ret


def add_plots(attacker, defender, which, smoothing_weight=(1-50/(episodes * steps))):
    global smooth
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

    print(f'Attacker/Defender: {au / steps / episodes:.4f}/{du / steps / episodes:.4f}')

    if which:
        plt.plot([0, steps * episodes], [au/steps/episodes, au/steps/episodes], alpha=0.5,
                 label=f'{attacker.__class__.__name__} Average Reward', linewidth=2.0)
        rewards = sorted(random.sample(aul, 500), key=lambda i: i[0])
        plt.plot([c[0] for c in rewards], [c[1] for c in rewards], alpha=0.3, linestyle='-', label=f'{attacker.__class__.__name__} Reward')
        plt.plot([c[0] for c in aul], smooth([c[1] for c in aul], smoothing_weight), label=f'{attacker.__class__.__name__} Smoothed Reward', linewidth=3.0)

    else:
        plt.plot([0, steps * episodes], [du / steps / episodes, du / steps / episodes], alpha=0.5,
                 label=f'{defender.__class__.__name__} Average Reward', linewidth=2.0)
        # rewards = sorted(random.sample(dul, 500), key=lambda i: i[0])
        rewards = dul
        plt.plot([c[0] for c in rewards], [c[1] for c in rewards], alpha=0.3, linestyle='-', label=f'{defender.__class__.__name__} Reward')

        plt.plot([c[0] for c in dul], smooth([c[1] for c in dul], smoothing_weight), label=f'{defender.__class__.__name__} Smoothed Reward', linewidth=3.0)

    plt.legend()
    return au / steps / episodes, du / steps / episodes

# attacker = UniformAttacker(m=m)
# attacker = DQNWrapper.load('weights/attacker_63be0de9_weights.pkl')
# defender = UniformDefender(m=m)

add_plots(UniformAttacker(), DQNWrapper.load('weights/defender_61e4024a_weights.pkl'), 0)
add_plots(UniformAttacker(), UniformDefender(), 0)
add_plots(UniformAttacker(), PCPDefender(), 0)
# add_plots(DQNWrapper.load('weights/attacker_f0419153_weights.pkl'), UniformDefender(), 1)
# add_plots(UniformAttacker(), UniformDefender(), 1)
#
plt.savefig('defender_reward_plot.png', dpi=300)
# plt.savefig('attacker_reward_plot.png', dpi=300)
# plt.show()
