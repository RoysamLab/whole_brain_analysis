
#!/bin/sh
#SBATCH -o script_demo.outfile
#SBATCH -e script_demo.errfile
#SBATCH -N 1
#SBATCH -t 120:00:00 
#SBATCH --mem-per-cpu=100GB
#SBATCH -n 1
#SBATCH --gres=gpu:1

module load cudatoolkit/10.1
conda activate BrainCellSeg

usr_root="/project/ece/roysam/xiaoyang/exps" 
project_root=$usr_root"/SegmentationPipeline"
data_root=$usr_root"/Data/50_plex/jj_final"

validation_set=$data_root"/atlas/multiplex_atlas"
dataset_path=$data_root"/images_stacked_multiplex/gray_adjusted.tif"
multiplex_path=$data_root"/images_stacked_multiplex/multiplex_adjusted.tif"





python main_prepare_images.py \
--INPUT_DIR=/path/to/input/dir \
--OUTPUT_DIR=NUCLEAR_SEG/data \
--DAPI R2C1.tif \
--HISTONES R2C2.tif 




cd ..
python3 main_nucleiSeg.py detect \
--dataset=NUCLEAR_SEG/data/multiplex.tif  \
--results=/path/to/output/dir \
