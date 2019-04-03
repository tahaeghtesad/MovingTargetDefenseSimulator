#!/bin/bash

#SBATCH -J MTD
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 16
#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -A laszka

module load GCC/7.2.0-2.29
module load Anaconda3/python-3.6
module load cuDNN/7.5.0-CUDA-10.0.130

cd MovingTargetDefenceSimulator

python Play.py
