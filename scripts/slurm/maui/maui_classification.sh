#!/bin/bash

#SBATCH -J G4_19
#SBATCH -o log_class
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 10 
#SBATCH --gres gpu:1
#SBATCH --mem 100g

module use /project/ece/roysam/jahandar/software/modulefiles/all
module load CUDA/9.0.176 

source /project/ece/roysam/jahandar/software/miniconda/bin/activate /project/ece/roysam/jahandar/software/miniconda/envs/brain

cd /project/ece/roysam/jahandar/whole_brain_analysis

python main_classification.py \
       --INPUT_DIR /project/ece/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#19_HC_12R/final \
       --OUTPUT_DIR /project/ece/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#19_HC_12R/classification_results \
       --BBXS_FILE /project/ece/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#19_HC_12R/detection_results/bbxs_detection.txt \
       --DAPI R1C1.tif \
       --NEUN R1C4.tif \
       --S100 R2C6.tif \
       --IBA1 R1C7.tif \
       --RECA1 R1C6.tif \
       --test_mode first \
       --thresholds .5 .5 1 .93 .5