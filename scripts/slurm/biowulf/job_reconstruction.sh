#!/bin/bash

#SBATCH -J AD1
#SBATCH -o AD1
#SBATCH -t 48:00:00
#SBATCH -p largemem
#SBATCH -n 40
#SBATCH --mem 500g

source deactivate
conda  activate /data/jahanipourj2/software/miniconda/envs/brain

python /data/jahanipourj2/codes/whole_brain_analysis/main_reconstruction.py \
       --INPUT_DIR=/data/jahanipourj2/datasets/50PLX/S3/original \
       --OUTPUT_DIR=/data/jahanipourj2/datasets/50PLX/S3 \
       --MODE=supervised \
       --SCRIPT=/data/jahanipourj2/datasets/50PLX/S3/script.csv
