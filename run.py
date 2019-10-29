import subprocess


def run(params):
    player, episodes, opponent, ef, ev, layers, gamma = params
    print(f"Running... {['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma]}")
    subprocess.run(['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma])
    return 0


episodes = ['100', '200', '500']
efs = ['0.2']  # 0.2
evs = ['0.02']  # 0.02
layers = ['x', '25,25', '256']  # 25, 25
gammas = ['.99', '0.97', '0.999']  # 0.99

attacker_opponents = ['UniformDefender']
defender_opponents = ['UniformAttacker']

runs = []
for ep in episodes:
    for opponent in attacker_opponents:
        for ef in efs:
            for ev in evs:
                for layer in layers:
                    for gamma in gammas:
                        runs.append(('attacker', ep, opponent, ef, ev, layer, gamma))

for ep in episodes:
    for opponent in defender_opponents:
        for ef in efs:
            for ev in evs:
                for layer in layers:
                    for gamma in gammas:
                        runs.append(('defender', ep, opponent, ef, ev, layer, gamma))

for r in runs:
    run(r)

# p = multiprocessing.Pool(int(multiprocessing.cpu_count()/2))
#
# p.map(run, runs)
