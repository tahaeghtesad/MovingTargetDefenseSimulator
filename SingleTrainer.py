import gym
import gym_mtd
import sys
import traceback

from AttackerNNExperience import AttackerNNExperience
from Attackers import *
from Defenders import *
from AttackLearner import AttackLearner
from keras_rl_util.AttackerProcessor import AttackerProcessor

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)

m = 10
episodes = 200
steps = 1000

attack_exp = AttackerNNExperience('attacker', m=m, max_memory_size=steps)


def main():
    try:
        processor = AttackerProcessor(m=m)
        env = gym.make('MTD-v0', m=m, time_limit=steps, utenv=0, setting=1, ca=0.2, alpha=.05,
                       defender=UniformDefender())

        learner = AttackLearner(attack_exp, env, processor)

        learner.fit(steps, episodes)

    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5, file=sys.stderr)

    finally:
        learner.finalize(True)
        learner.test(steps, 3)


if __name__ == '__main__':
    main()
