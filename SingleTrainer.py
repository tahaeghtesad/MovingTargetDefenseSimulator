from tqdm import tqdm

from AttackLearner import AttackLearner
from DefenseLearner import DefenseLearner
from AttackerNNExperience import AttackerNNExperience
from DefenderNNExperience import DefenderNNExperience
from Defenders import *
from Attackers import *
from Game import Game
from enum import Enum

import datetime
import multiprocessing
import numpy as np

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)

number_of_servers = 10
episodes = 10000
steps = 1000

attacker_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=steps)
defender_exp = DefenderNNExperience('defender', m=number_of_servers, max_memory_size=steps)


def generate_samples(i):

    # ca = i/episodes*.05
    # epsilon = (episodes-i)/episodes
    # delta = int(i/episodes*7/10*number_of_servers)

    # print(f'Running game {i}...')

    ca = 0.2
    epsilon = .97**i
    delta = 7

    attacker_exp.reset_exp()
    defender_exp.reset_exp()

    game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=ca, downtime=delta)

    if mode == Mode.Attacker:
        attacker = AttackLearner(attacker_exp, m=number_of_servers, epsilon=epsilon, downtime=delta)
        defender = UniformDefender(m=number_of_servers, p=4, downtime=delta)
    else:
        attacker = MaxProbeAttacker(m=number_of_servers, downtime=delta)
        defender = DefenseLearner(defender_exp, m=number_of_servers, epsilon=epsilon, downtime=delta)

    game.play(attacker, defender)

    rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Defender: {attacker.utility/steps:.4f}/{defender.utility/steps:.4f}')


# def train():
#     if mode == Mode.Attacker:
#         attacker_exp.train_on_samples()
#     else:
#         defender_exp.train_on_samples()


def evaluate_attacker(attackerT):
    au = 0
    du = 0

    episodes = 10
    steps = 1000

    for i in range(episodes):
        game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = AttackLearner(attacker_exp, m=number_of_servers, train=False) if attackerT == AttackLearner else attackerT(m=number_of_servers)
        defender = UniformDefender(m=number_of_servers, p=4)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'{attackerT.__name__}/Defender: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')
    print(f'{attackerT.__name__}/Defender: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')


def evaluate_defender(defenderT):
    au = 0
    du = 0

    episodes = 10
    steps = 1000

    for i in range(episodes):
        game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = MaxProbeAttacker(m=number_of_servers)
        defender = DefenseLearner(defender_exp, m=number_of_servers, train=False) if defenderT == DefenseLearner else defenderT(m=number_of_servers)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'Attacker/{defenderT.__name__}: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')
    print(f'Attacker/{defenderT.__name__}: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')


class Mode(Enum):
    Attacker = 1
    Defender = 2


mode = Mode.Defender


def main():

    begin = datetime.datetime.now()

    print('Running Simulations')

    # with multiprocessing.Pool(int(multiprocessing.cpu_count()/2)) as pool:
    #     pool.map(generate_samples, range(0, episodes))

    # train()

    for i in tqdm(range(episodes)):
        generate_samples(i)


    if mode == Mode.Attacker:
        evaluate_attacker(AttackLearner)
        evaluate_attacker(BaseAttacker)
        evaluate_attacker(MaxProbeAttacker)
        evaluate_attacker(UniformAttacker)
    else:
        evaluate_defender(DefenseLearner)
        evaluate_defender(BaseDefender)
        evaluate_defender(UniformDefender)
        evaluate_defender(PCPDefender)

    rootLogger.info(f'Training took {(datetime.datetime.now() - begin).total_seconds():.2f} seconds.')
    print(f'Training took {(datetime.datetime.now() - begin).total_seconds():.2f} seconds.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if mode == Mode.Attacker:
            evaluate_attacker(AttackLearner)
            evaluate_attacker(BaseAttacker)
            evaluate_attacker(MaxProbeAttacker)
            evaluate_attacker(UniformAttacker)
        else:
            evaluate_defender(DefenseLearner)
            evaluate_defender(BaseDefender)
            evaluate_defender(UniformDefender)
            evaluate_defender(PCPDefender)
        exit(-1)
