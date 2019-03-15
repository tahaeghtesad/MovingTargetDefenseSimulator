#!/bin/bash

#SBATCH -J MTD
#SBATCH -t 6:00:00
#SBATCH -N 1 -n 4
#SBATCH -p volta
#SBATCH --gres=gpu:1

#SBATCH -A laszka


module load python/3.6
cd MovingTargetDefenceSimulator

source venv/bin/activate

python Trainer.py

deactivate