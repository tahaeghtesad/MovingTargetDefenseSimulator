import subprocess
import multiprocessing


def run(params):
    player, episodes, opponent, ef, ev, layers = params
    subprocess.run(['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers])
    return 0


episodes = ['50', '100']
efs = ['0.5', '0.2', '0.1']
evs = ['0.02', '0.1', '0.2']
layers = ['x', '25', '25,25', '256', '256,128']

attacker_opponents = ['UniformDefender', 'PCPDefender']
defender_opponents = ['UniformAttacker', 'MaxProbeAttacker']

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
