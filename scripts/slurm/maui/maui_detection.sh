#!/bin/bash

#SBATCH -J G2_BR_22_det
#SBATCH -o G2_BR_22_det
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem 50g

module load CUDA/9.0.176 

module load Anaconda3
source activate /project/hnguyen/jahandar/software/miniconda/envs/brain

python main_detection.py \
       --INPUT_DIR=/project/hnguyen/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L/detection_results \
       --OUTPUT_DIR=/project/hnguyen/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L/detection_results \
       --DAPI=R2C1.tif \
