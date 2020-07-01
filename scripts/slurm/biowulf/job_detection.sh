#!/bin/bash

#SBATCH -J S3_det
#SBATCH -o S3_det.log
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -n 20
#SBATCH --gres gpu:v100x:1
#SBATCH --mem 50g

module load CUDA/9.0
module load cuDNN/7.0

source deactivate
conda activate /data/jahanipourj2/software/miniconda/envs/brain

python main_detection.py \
       --INPUT_DIR=/data/jahanipourj2/datasets/50PLX/S3/detection_results \
       --OUTPUT_DIR=/data/jahanipourj2/datasets/50PLX/S3/detection_results \
       --DAPI=S3_R2C1.tif \
       --HISTONES=S3_R2C2.tif
