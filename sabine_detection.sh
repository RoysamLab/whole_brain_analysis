#!/bin/bash

#SBATCH -J G3_14_det 
#SBATCH -o G3_14_det
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 28
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem 240g
#SBATCH -A roysam

module load CUDA/.10.0.130
module load cuDNN/7.5.0-CUDA-10.0.130

source activate /brazos/roysam/shared/miniconda/envs/tf

python main_detection.py
