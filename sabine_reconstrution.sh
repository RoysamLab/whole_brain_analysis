#!/bin/bash

#SBATCH -J G4_BR_19
#SBATCH -o G4_BR_19
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 28
#SBATCH --mem 240gb
#SBATCH -A roysam

source activate /brazos/roysam/shared/miniconda/envs/brain

python main_reconstruction_linux.py

