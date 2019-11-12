import subprocess
from subprocess import PIPE


def run(params):
    player, episodes, opponent, ef, ev, layers, gamma, dueling, double, prioritized_replay, normalization = params
    print(f"Running... {['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma, dueling, double, prioritized_replay, normalization]}")
    completed = subprocess.run(['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma, dueling, double, prioritized_replay, normalization], stdout=PIPE)
    job_id = int(completed.stdout.decode("ascii").split('\n')[1].split(' ')[-1])
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

runs = []
for ep in episodes:
    for ef in efs:
        for ev in evs:
            for layer in layers:
                for gamma in gammas:
                    for dueling in duelings:
                        for double in doubles:
                            for prioritized_replay in prioritized_replays:
                                for normalization in normalizations:
                                    for opponent in attacker_opponents:
                                        runs.append(('attacker', ep, opponent, ef, ev, layer, gamma, dueling, double, prioritized_replay, normalization))
                                    # for opponent in defender_opponents:
                                    #     runs.append(('defender', ep, opponent, ef, ev, layer, gamma, dueling, double, prioritized_replay, normalization))

for r in runs:
    id = run(r)
    print(f'Job submitted with id: {id}')

# p = multiprocessing.Pool(int(multiprocessing.cpu_count()/2))
#
# p.map(run, runs)
