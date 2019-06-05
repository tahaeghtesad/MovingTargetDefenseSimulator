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
episodes = 200
steps = 1000

attack_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=steps)
defender_exp = DefenderNNExperience('defender', m=number_of_servers, max_memory_size=steps)


def train(i, mode):

    ca = i/episodes*.05
    epsilon = (episodes-i)/episodes

    game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=steps, ca=ca)

    if mode == Mode.Attacker:
        attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=epsilon)
        defender = UniformDefender(m=number_of_servers, p=4)
    else:
        attacker = MaxProbeAttacker(m=number_of_servers)
        defender = DefenseLearner(defender_exp, m=number_of_servers, epsilon=(episodes-i)/episodes)

    game.play(attacker, defender)

    attacker.finalize(True)
    defender.finalize(True)

    rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Defender: {attacker.utility/steps:.4f}/{defender.utility/steps:.4f}')



def evaluate_attacker(attackerT):
    au = 0
    du = 0

    episodes = 10
    steps = 1000

    for i in range(episodes):
        game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = AttackLearner(attack_exp, m=number_of_servers, train=False) if attackerT == AttackLearner else attackerT()
        defender = UniformDefender(m=number_of_servers, p=4)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'{attackerT.__name__}/Defender: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')

def evaluate_defender(defenderT):
    au = 0
    du = 0

    episodes = 10
    steps = 1000

    for i in range(episodes):
        game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = MaxProbeAttacker()
        defender = DefenseLearner(defender_exp, m=number_of_servers, train=False) if defenderT == DefenseLearner else defenderT()

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'Attacker/{defenderT.__name__}: {au/steps/episodes:.4f}/{du/steps/episodes:.4f}')

class Mode(Enum):
    Attacker = 1
    Defender = 2

mode = Mode.Attacker

def main():

    begin = datetime.datetime.now()

    for i in tqdm(range(episodes)):
        train(i, mode)

    rootLogger.info(f'Training took {(datetime.datetime.now() - begin).total_seconds():.2f} seconds.')

    if mode == Mode.Attacker:
        evaluate_attacker(AttackLearner)
        evaluate_attacker(BaseAttacker)
        evaluate_attacker(MaxProbeAttacker)
        evaluate_attacker(UniformAttacker)
    else:
        evaluate_defender(DefenseLearner)
        evaluate_defender(BaseDefender)
        evaluate_defender(UniformDefender)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(-1)
