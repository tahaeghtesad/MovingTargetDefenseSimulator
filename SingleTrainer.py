from tqdm import tqdm

from AttackLearner import AttackLearner
from DefenseLearner import DefenseLearner
from AttackerNNExperience import AttackerNNExperience
from DefenderNNExperience import DefenderNNExperience
from Defenders import *
from Attackers import *
from Game import Game

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
episodes = 1000
steps = 1000

attack_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=steps)
defender_exp = DefenderNNExperience('defender', m=number_of_servers, max_memory_size=steps)


def train(i):

    attack_exp.erase_memory()
    defender_exp.erase_memory()

    game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=0.20)
    # attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=(episodes-i)/episodes)
    # defender = UniformDefender(m=number_of_servers, p=4)

    attacker = BaseAttacker(m=number_of_servers)

    # attacker = MaxProbeAttacker(m=number_of_servers)
    defender = DefenseLearner(defender_exp, m=number_of_servers, epsilon=0)

    # defender = BaseDefender()

    game.play(attacker, defender)

    attacker.finalize(True)
    defender.finalize(True)

    rootLogger.info(f'Game {i+1}/{episodes}: Attacker/Defender: {attacker.utility/steps:.4f}/{defender.utility/steps:.4f}')



def evaluate_attacker(attackerT):
    au = 0
    du = 0

    episodes = 100
    steps = 150

    for i in range(episodes):
        game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = AttackLearner(attack_exp, m=number_of_servers, train=False) if attackerT == AttackLearner else attackerT()
        defender = UniformDefender(m=number_of_servers, p=4)
        # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'{attackerT.__name__}/Defender: {au/steps/episodes}/{du/steps/episodes}')

def evaluate_defender(defenderT):
    au = 0
    du = 0

    episodes = 100
    steps = 150

    for i in range(episodes):
        game = Game(utenv=0, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
        attacker = BaseAttacker()
        defender = DefenseLearner(defender_exp, m=number_of_servers, train=False) if defenderT == DefenseLearner else defenderT()
        # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'Attacker/{defenderT.__name__}: {au/steps/episodes}/{du/steps/episodes}')



def main():
    for i in tqdm(range(episodes)):
        train(i)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # evaluate_attacker(AttackLearner)
        # evaluate_attacker(BaseAttacker)
        # evaluate_attacker(MaxProbeAttacker)
        # evaluate_attacker(UniformAttacker)

        evaluate_defender(DefenseLearner)
        evaluate_defender(BaseDefender)
        evaluate_defender(UniformDefender)
        evaluate_defender(PCPDefender)

