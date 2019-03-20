from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from AttackLearner import AttackLearner
from AttackerQExperience import AttackerQExperience
from Defenders import *
from Game import Game
import copy

rootLogger = logging.getLogger()

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler('log.log', mode='w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)

number_of_servers = 2
episodes = 1000

whole_exp = AttackerQExperience('attacker-total', m=number_of_servers)


def train(i):
    attack_exp = AttackerQExperience('attacker', m=number_of_servers)
    game = Game(utenv=1, setting=1, m=number_of_servers, ca=0)
    attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=1 / math.sqrt(i + 1))
    defender = UniformDefender(m=number_of_servers, p=14)

    # attacker = UniformAttacker(m=number_of_servers)
    # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

    game.play(attacker, defender)

    # attacker.finalize(i != 0 and i % 100 == 0)
    # defender.finalize(i != 0 and i % 100 == 0)
    rootLogger.info(f'Game {i + 1}/{episodes}: Attacker/Defender:{int(attacker.utility)}/{int(defender.utility)}')
    return attack_exp.retrieve_exp()


def main():
    pool = Pool(cpu_count())
    for i in tqdm(range(episodes)):
        exps = pool.map(train, [i] * cpu_count())
        for exp in exps:
            whole_exp.store_exp(exp)
            whole_exp.train_model(1000)
        whole_exp.store()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(-1)
