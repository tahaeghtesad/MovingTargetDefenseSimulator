import subprocess


def run(params):
    player, episodes, opponent, ef, ev, layers, gamma = params
    print(f"Running... {['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma]}")
    # completed = \
    subprocess.run(['sbatch', 'run.srun.sh', player, episodes, opponent, ef, ev, layers, gamma])
    # job_id = int(completed.stdout.decode("ascii").split('\n')[1].split(' ')[-1])
    # return job_id


episodes = ['100', '200']
efs = ['0.1', '0.2', '0.3']  # 0.2
evs = ['0.01', '0.02', '0.1', '0.05']  # 0.02
layers = ['x', '25', '25,25']  # 25, 25
gammas = ['0.9', '0.95', '0.97', '.999']  # 0.99

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

# for ep in episodes:
#     for opponent in defender_opponents:
#         for ef in efs:
#             for ev in evs:
#                 for layer in layers:
#                     for gamma in gammas:
#                         runs.append(('defender', ep, opponent, ef, ev, layer, gamma))

for r in runs:
    # id = run(r)
    # print(f'Job submitted with id: {id}')
    run(r)

# p = multiprocessing.Pool(int(multiprocessing.cpu_count()/2))
#
# p.map(run, runs)
