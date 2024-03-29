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

rootLogger.setLevel(logging.INFO)


# game = Game()
# attacker = MaxProbeAttacker()
# attacker = UniformAttacker()
# attacker = ControlThresholdAttacker()
# defender = ControlThresholdDefender()
# defender = PCPDefender()
# defender = UniformDefender()
# defender = ControlTargetDefender()

attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]

try:
    for attackerT in attackers:
        for defenderT in defenders:
            rootLogger.info(f'{attackerT} VS. {defenderT}')
            game = Game(utenv=0, setting=1)
            attacker = attackerT()
            defender = defenderT()
            game.play(attacker, defender)
            # time.sleep(.1)
            rootLogger.info(f'Attacker utility: {attacker.utility/1000:.4f}')
            rootLogger.info(f'Defender utility: {defender.utility/1000:.4f}')
except KeyboardInterrupt:
    exit(-1)
