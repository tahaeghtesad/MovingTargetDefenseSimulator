from Game import Game
from Attackers import *
from Defenders import *
import time

import logging
import sys

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.ERROR)


game = Game()
# attacker = MaxProbeAttacker()
# defender = PCPDefender()
# attacker = UniformAttacker()
# defender = ControlThresholdDefender()
attacker = ControlThresholdAttacker()
defender = UniformDefender()

try:
    game.play(attacker, defender)
    time.sleep(.1)
    rootLogger.error(f'Attacker utility: {attacker.utility}')
    rootLogger.error(f'Defender utility: {defender.utility}')
except KeyboardInterrupt:
    exit(-1)
