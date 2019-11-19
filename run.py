import subprocess
from subprocess import PIPE


def run(params):

    run_params = ['sbatch', 'run.srun.sh'] + params

    print(f"Running... {run_params}")
    completed = subprocess.run(run_params, stdout=PIPE)
    job_id = int(completed.stdout.decode("ascii").split('\n')[0].split(' ')[-1])
    return job_id


episodes = ['250', '500']
efs = ['0.2']  # 0.2
evs = ['0.02']  # 0.02
layers = ['64', '64,64']  # 25, 25
gammas = ['0.99']  # 0.99
duelings = ['True', 'False']
doubles = ['True', 'False']
prioritized_replays = ['False']
normalizations = ['True']

attacker_opponents = ['PCPDefender']
defender_opponents = ['UniformAttacker']

# runs = []
# for ep in episodes:
#     for ef in efs:
#         for ev in evs:
#             for layer in layers:
#                 for gamma in gammas:
#                     for dueling in duelings:
#                         for double in doubles:
#                             for prioritized_replay in prioritized_replays:
#                                 for normalization in normalizations:
#                                     for opponent in attacker_opponents:
#                                         runs.append(('attacker', ep, opponent, ef, ev, layer, gamma, dueling, double, prioritized_replay, normalization))
                                    # for opponent in defender_opponents:
                                    #     runs.append(('defender', ep, opponent, ef, ev, layer, gamma, dueling, double, prioritized_replay, normalization))

runs = [
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64', '0.99', 'False', 'False', 'False', 'True', '0', 'Vanilla'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64', '0.99', 'False', 'True', 'False', 'True', '0', 'Double-$Q$'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64', '0.99', 'True', 'False', 'False', 'True', '0', 'Dueling-$Q'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.20', '64', '0.99', 'False', 'False', 'False', 'True', '0', '$\\epsilon_f = 0.2$'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64,64', '0.99', 'False', 'False', 'False', 'True', '0', '2 Hidden Layers'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64', '0.99', 'False', 'False', 'True', 'True', '0', 'Prioritized Replay'],
    ['attacker', '500', 'PCPDefender', '0.2', '0.02', '64', '0.99', 'False', 'False', 'False', 'True', '0.5', '$\\nu = 0.5$'],

    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64', '0.99', 'False', 'False', 'False', 'True', '0', 'Vanilla'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64', '0.99', 'False', 'True', 'False', 'True', '0', 'Double-$Q$'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64', '0.99', 'True', 'False', 'False', 'True', '0', 'Dueling-$Q'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.20', '64', '0.99', 'False', 'False', 'False', 'True', '0', '$\\epsilon_f = 0.2$'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64,64', '0.99', 'False', 'False', 'False', 'True', '0', '2 Hidden Layers'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64', '0.99', 'False', 'False', 'True', 'True', '0', 'Prioritized Replay'],
    ['defender', '500', 'UniformAttacker', '0.2', '0.02', '64', '0.99', 'False', 'False', 'False', 'True', '0.5', '$\\nu = 0.5$'],
]

for r in runs:
    id = run(r)
    print(f'Job submitted with id: {id}')

# p = multiprocessing.Pool(int(multiprocessing.cpu_count()/2))
#
# p.map(run, runs)
