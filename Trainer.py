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

defend_model = DefenceLearner.create_model(m=2)
attack_model = AttackLearner.create_model(m=2)

episodes = 1000
try:

    for i in tqdm(range(episodes)):
        game = Game(utenv=2, setting=1, m=2)
        attacker = AttackLearner(epsilon=(episodes-i)/episodes, model=attack_model, m=2)
        defender = MaxProbeDefender(pi=2)

        # attacker = UniformAttacker()
        # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

        game.play(attacker, defender)

        attacker.finalize(i == episodes-1)
        defender.finalize(i == episodes-1)
        rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Defender:{int(attacker.utility)}/{int(defender.utility)}')

except KeyboardInterrupt:
    exit(-1)
