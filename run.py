import subprocess


def run(params):
    player, episodes, opponent, ef, ev, layers = params
    print(f"Running... {['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers]}")
    subprocess.run(['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers])
    return 0


episodes = ['500']
efs = ['0.2']
evs = ['0.02']
layers = ['x', '256']

attacker_opponents = ['UniformDefender']
defender_opponents = ['UniformAttacker']

runs = []
for ep in episodes:
    for opponent in attacker_opponents:
        for ef in efs:
            for ev in evs:
                for layer in layers:
                    runs.append(('attacker', ep, opponent, ef, ev, layer))

for ep in episodes:
    for opponent in defender_opponents:
        for ef in efs:
            for ev in evs:
                for layer in layers:
                    runs.append(('defender', ep, opponent, ef, ev, layer))

for r in runs:
    run(r)

# p = multiprocessing.Pool(int(multiprocessing.cpu_count()/2))
#
# p.map(run, runs)
