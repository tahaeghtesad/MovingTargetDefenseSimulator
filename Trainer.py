from tqdm import tqdm

from Attackers import *
from DefendLearner import DefendLearner
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

model = DefendLearner.create_model(10)


episodes = 1000
try:

    for i in tqdm(range(episodes)):
        game = Game()
        attacker = UniformAttacker()
        defender = DefendLearner(model=model, epsilon=(episodes - i) / episodes)
        game.play(attacker, defender)
        rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Deffender:{int(attacker.utility)}/{int(defender.utility)}')

except KeyboardInterrupt:
    exit(-1)
