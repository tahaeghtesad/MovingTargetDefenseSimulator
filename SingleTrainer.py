from tqdm import tqdm

from AttackLearner import AttackLearner
from AttackerNNExperience import AttackerNNExperience
from Defenders import *
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

number_of_servers = 3
episodes = 10000
rounds = 100000

attack_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=rounds)


def train(i):
    game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=rounds)
    attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=1/math.sqrt(i + 1))
    defender = UniformDefender(m=number_of_servers, p=10)

    # attacker = MaxProbeAttacker(m=number_of_servers)
    # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

    game.play(attacker, defender)

    attacker.finalize(i != 0 and i % 10 == 0)
    defender.finalize(i != 0 and i % 10 == 0)
    rootLogger.info(f'Game {i + 1}/{episodes}: Attacker/Defender:{int(attacker.utility)}/{int(defender.utility)}')


def main():
    for i in tqdm(range(episodes)):
        train(i)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(-1)
