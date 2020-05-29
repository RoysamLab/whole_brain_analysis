#!/bin/bash

#SBATCH -J G4_19
#SBATCH -o log_det
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 5 
#SBATCH --gres gpu:1
#SBATCH --mem 50g

module use /project/ece/roysam/jahandar/software/modulefiles/all
module load CUDA/9.0.176 

source /project/ece/roysam/jahandar/software/miniconda/bin/activate /project/ece/roysam/jahandar/software/miniconda/envs/brain

cd /project/ece/roysam/jahandar/whole_brain_analysis

python main_detection.py \
       --INPUT_DIR=/project/ece/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L/detection_results \
       --OUTPUT_DIR=/project/ece/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#18_HC_14L/detection_results \
       --DAPI=R2C1.tif \
