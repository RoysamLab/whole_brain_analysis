#!/bin/bash

#SBATCH -J G2_BR_22
#SBATCH -o G2_BR_22
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 28
#SBATCH --mem 240gb
#SBATCH -A roysam

module load Anaconda3
source activate /brazos/roysam/shared/miniconda/envs/brain

python main_reconstruction_lin.py