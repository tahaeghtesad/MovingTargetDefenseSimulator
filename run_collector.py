import os
import subprocess


def run(params):
    dir, name = params
    print(f"Running... {['sbatch', 'collect.srun.sh', dir, name]}")
    subprocess.run(['sbatch', 'collect.srun.sh', dir, name])
    return 0


if __name__ == '__main__':
    paths = []

    for subdir, dirs, files in os.walk('tb_logs'):
        for dir in dirs:
            for s, ds, fs in os.walk(f'tb_logs/{dir}'):
                paths.append((dir, fs[0]))
        break

    for p in paths:
        run(p)