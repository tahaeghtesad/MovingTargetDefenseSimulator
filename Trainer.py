from tqdm import tqdm

from Attackers import ControlThresholdAttacker
from DefendLearner import DefendLearner
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

episodes = 1000
try:
    for epoch in tqdm(range(episodes)):
        game = Game(utenv=2)
        attacker = ControlThresholdAttacker()
        defender = DefendLearner(epsilon=(episodes - epoch) / episodes)
        game.play(attacker, defender)
        time.sleep(.01)
        rootLogger.info(f'Attacker utility: {int(attacker.utility)}')
        rootLogger.info(f'Defender utility: {int(defender.utility)}')

except KeyboardInterrupt:
    exit(-1)
