#!/bin/sh
#SBATCH -p medium --mem=125GB --exclusive
#SBATCH -t 50:00:00 
#SBATCH -o run-ICE-generate_temp.outfile
#SBATCH -e run-ICE-generate_temp.errfile

cd /brazos/roysam/xli63/exps/SegmentationPipeline/DataAnalysis

# Gerate ICE [From bbox]  

#pip install fcswrite --user
data_dir="/brazos/roysam/50_plex/Set#1_S1"
python GenerateICE_FCS_script.py \
--inputDir="$data_dir"/final \
--maskType=b \   # bbbox
--maskDir="$data_dir"/detection_results/bbxs_detection.txt \
--outputDir="$data_dir"/ICE_FCS_files_temp \
--xmlDir=$xmldir
--saveVis=1 \
--downscaleRate=4 \
--seedSize=3
