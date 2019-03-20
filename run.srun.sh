#!/bin/bash

#SBATCH -J MTD
#SBATCH -t 6:00:00
#SBATCH -N 1 -n 4
#SBATCH -p volta
#SBATCH --gres=gpu:1

#SBATCH -A laszka

module load GCC/7.2.0-2.29
module load Anaconda3/python-3.6
module load cuDNN/7.5.0-CUDA-10.0.130

cd MovingTargetDefenceSimulator

#python -m virtualenv venv

#source venv/bin/activate
#pip install --user tqdm tensorflow-gpu==1.12.1 keras

python Trainer.py

#deactivate
#rm -rf venv