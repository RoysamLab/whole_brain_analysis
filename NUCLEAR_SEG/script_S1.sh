#!/bin/sh
#SBATCH -o script_S1.outfile
#SBATCH -e script_S1.errfile
#SBATCH -N 1
#SBATCH -t 120:00:00 
#SBATCH --mem-per-cpu=100GB
#SBATCH -n 1
#SBATCH --gres=gpu:1

module load cudatoolkit/10.1
conda activate BrainCellSeg

usr_root="/project/ece/roysam/xiaoyang/exps"
project_root=$usr_root"/whole_brain_analysis"
data_root="/project/ece/roysam/datasets/50_plex/S1/final"
cd $project_root


# Prepare image
#python NUCLEAR_SEG/main_prepare_images.py \
#--INPUT_DIR $data_root \
#--OUTPUT_DIR NUCLEAR_SEG/data \
#--DAPI S1_R2C1.tif \
#--HISTONES S1_R2C2.tif 


# Run segmentation/Detection

python3 main_nucleiSeg.py detect \
--dataset=NUCLEAR_SEG/data/multiplex.tif  \
--results=NUCLEAR_SEG/output 
