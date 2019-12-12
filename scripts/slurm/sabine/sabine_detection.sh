#!/bin/bash

#SBATCH -J G2_BR_22_det
#SBATCH -o G2_BR_22_det
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem 50g
#SBATCH -A roysam

module load CUDA/.10.0.130
module load cuDNN/7.4.2.24-CUDA-10.0.130

module load Anaconda3
source activate /brazos/roysam/shared/miniconda/envs/brain

python main_detection.py \
       --INPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/detection_results \
       --OUTPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/detection_results \
       --DAPI=R2C1.tif \
