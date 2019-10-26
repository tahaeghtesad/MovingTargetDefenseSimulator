import gym
import gym_mtd
import matplotlib.pyplot as plt
from agents.defenders import *
from agents.attackers import *
from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor

episodes = 1
steps = 500
m = 10


def smooth(series, weight=.99):
    ret = [series[0]]
    smoothed = series[0]
    for i in range(1, len(series)):
        smoothed = smoothed * weight + (1-weight) * series[i]
        ret.append(smoothed)
    return ret

def compare(env, ap):
    if len(env) != len(ap):
        return False
    for i in range(len(env)):
        if env[i]['control'].value != ap[i]['control']:
            return False
    return True


def run_game(attacker, defender):
    env = gym.make('MTD-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2)

    ap = AttackerProcessor(m=m)
    dp = DefenderProcessor(m=m)

    au = 0
    du = 0

    aul = list()
    dul = list()

    attacker_action = list()
    defender_action = list()

    attacker_control = list()
    defender_control = list()
    down = list()

    for i in range(episodes):

        done = False
        obs = env.reset()
        iter = 0

        a_obs = ap.process_observation(obs)
        d_obs = dp.process_observation(obs)

        while not done:

            attacker_prediction = attacker.predict(a_obs)

            a_action = ap.process_action(attacker_prediction)
            d_action = dp.process_action(defender.predict(d_obs))

            obs, reward, done, info = env.step((
                a_action, d_action
            ))

            attacker_action.append(a_action)
            defender_action.append(d_action)

            attacker_control.append(env.nca)
            defender_control.append(env.ncd)
            down.append(env.nd)

            a_obs, a_r, a_d, a_i = ap.process_step(obs, reward, done, info)
            d_obs, d_r, d_d, d_i = dp.process_step(obs, reward, done, info)

            assert compare(env.servers, ap.servers) == True

            au += a_r
            du += d_r

            # obs, reward, done, info = env.step(attacker.predict(obs))
            aul.append(a_r)
            dul.append(d_r)
            iter += 1

    print(f'{attacker.__class__.__name__}/{defender.__class__.__name__}: {au / steps / episodes:.4f}/{du / steps / episodes:.4f}')
    return aul, dul, attacker_action, defender_action, attacker_control, defender_control, down


def create_plots(attacker, defender):
    aul, dul, attacker_action, defender_action, attacker_control, defender_control, down = run_game(attacker, defender)

    plt.figure(figsize=(20, 10))
    plt.title(f'{attacker.__class__.__name__} vs {defender.__class__.__name__} Utility')
    plt.plot(smooth(aul), label='SmoothedAttackerUtility')
    plt.plot(smooth(dul), label='SmoothedDefenderUtility')
    plt.plot(aul, alpha=0.2, linestyle='-', label=f'AttackerUtility')
    plt.plot(dul, alpha=0.2, linestyle='-', label=f'DefenderUtility')
    plt.grid()
    plt.legend()
    plt.savefig(f'debug_plots/{attacker.__class__.__name__}_{defender.__class__.__name__}_util.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title(f'{attacker.__class__.__name__} vs {defender.__class__.__name__} Action')
    plt.plot([c - 0.05 for c in attacker_action], '.', label='AttackerAction')
    plt.plot([c + 0.05 for c in defender_action], '.', label='DefenderAction')
    plt.grid()
    plt.legend()
    plt.savefig(f'debug_plots/{attacker.__class__.__name__}_{defender.__class__.__name__}_action.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title(f'{attacker.__class__.__name__} vs {defender.__class__.__name__} Control')
    plt.plot(smooth(attacker_control), label='SmoothedAttackerControl')
    plt.plot(smooth(defender_control), label='SmoothedDefenderControl')
    plt.plot(attacker_control, alpha=0.2, label='AttackerControl')
    plt.plot(defender_control, alpha=0.2, label='DefenderControl')

    plt.plot(smooth(down), label='SmoothedDown')
    plt.plot(down, alpha=0.2, label='Down')
    plt.grid()
    plt.legend()
    plt.savefig(f'debug_plots/{attacker.__class__.__name__}_{defender.__class__.__name__}_control.png')
    plt.close()


attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]

for a in attackers:
    for d in defenders:
        create_plots(a(), d())

create_plots(ControlThresholdAttacker(), ControlThresholdDefender())

