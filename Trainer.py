from tqdm import tqdm

from Attackers import *
from Defenders import *
from AttackLearner import AttackLearner
from DefenceLearner import DefenceLearner

from multiprocessing import Pool
from Game import Game

import logging
import time
import sys
import math

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

# defend_model = DefenceLearner.create_neural_model()
attack_model = AttackLearner.create_neural_model(m=number_of_servers)

# defend_model = DefenceLearner.create_q_table(number_of_servers)
# attack_model = AttackLearner.create_q_table()


episodes = 100000
try:

    for i in tqdm(range(episodes)):

        game = Game(utenv=2, setting=1, m=number_of_servers)
        attacker = AttackLearner(m=number_of_servers, epsilon=(episodes-i)/episodes, model=attack_model)
        defender = MaxProbeDefender(pi=3, m=number_of_servers)

        # attacker = UniformAttacker()
        # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

        game.play(attacker, defender)

        attacker.finalize(i % 1000 == 0)
        defender.finalize(i % 1000 == 0)
        rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Defender:{int(attacker.utility)}/{int(defender.utility)}')

except KeyboardInterrupt:
    exit(-1)
