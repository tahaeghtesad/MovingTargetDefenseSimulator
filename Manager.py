from Attacker import BaseAttacker
from Defender import BaseDefender
from Game import Game

import logging
import sys

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.ERROR)


game = Game(time_limit=2 ** 256)
attacker = BaseAttacker()
defender = BaseDefender()

try:
    game.play(attacker, defender)
except KeyboardInterrupt:
    exit(-1)
