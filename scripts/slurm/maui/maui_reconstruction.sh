#!/bin/bash

#SBATCH -J G3_BR_18
#SBATCH -o G3_BR_18
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 40
#SBATCH --mem 185gb

source /project/ece/roysam/jahandar/software/miniconda/bin/activate /project/ece/roysam/jahandar/software/miniconda/envs/brain


python main_reconstruction.py \
       --INPUT_DIR=/project/hnguyen/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L/original \
       --OUTPUT_DIR=/project/hnguyen/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L \
       --MODE=supervised \
       --SCRIPT=scripts/reconstruction/20_plex.csv
