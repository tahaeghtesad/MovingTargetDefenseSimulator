from tqdm import tqdm

from AttackLearner import AttackLearner
from AttackerNNExperience import AttackerNNExperience
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
steps = 40000

attack_exp = AttackerNNExperience('attacker', m=number_of_servers, max_memory_size=steps)

attacker_util = 0
defender_util = 0


def train(i):
    game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=steps, ca=.2)
    attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=.99) #(episodes-i)/episodes)
    defender = UniformDefender(m=number_of_servers, p=4)

    # attacker = BaseAttacker(m=number_of_servers)

    # attacker = MaxProbeAttacker(m=number_of_servers)
    # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

    game.play(attacker, defender)

    attacker.finalize(True)
    defender.finalize(True)


    evaluate_attacker(AttackLearner(attack_exp, m=number_of_servers, train=False))
    evaluate_attacker(BaseAttacker())
    evaluate_attacker(MaxProbeAttacker())



def evaluate_attacker(attacker):
    au = 0
    du = 0
    for i in range(100):
        game = Game(utenv=1, setting=1, m=number_of_servers, time_limit=150, ca=.2)
        # attacker = AttackLearner(attack_exp, m=number_of_servers, epsilon=.80, train=False)  # (episodes-i)/episodes)
        defender = UniformDefender(m=number_of_servers, p=4)

        # attacker = BaseAttacker(m=number_of_servers)

        # attacker = MaxProbeAttacker(m=number_of_servers)
        # defender = DefendLearner(epsilon=(episodes-i)/episodes, model=defend_model)

        game.play(attacker, defender)

        au += attacker.utility
        du += defender.utility

    rootLogger.info(f'{attacker.__class__.__name__}/Defender: {au/100}/{du/100}')



def main():
    for i in tqdm(range(episodes)):
        train(i)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(-1)
