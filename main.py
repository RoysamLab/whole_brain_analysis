import os
import time
import subprocess


###################################################################
# NAMING PROTOCOL: RxCy (x = round number | y = channel number)
###################################################################


# REGISTRATION
input_dir = r'/brazos/roysam/asingh42/DashData_Raw/G3_mFPI_Vehicle/G3_BR#14_HC_11L'
output_dir = r'/project/roysam/datasets/TBI/G2_mFPI_Vehicle/G3_BR#14_HC_11L'
command = """python 1_PREPROCESSING/registration_multiRds.py \
             --input_dir={0} \
             --output_dir={1} \
             --maxtrials=100 \
             --tiling="800,1600" \
             --imadjust=True \
             --demo=False \
             -nk 100000 \
           """.format(input_dir, output_dir)
start = time.time()
p = subprocess.call(command)
print('Registration pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# INTRA-CHANNEL CORRECTION
input_dir = r'E:\jahandar\DashData\Injury\original'
output_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
disk_size = [20, 40]
command = """matlab -minimize -nosplash -nodesktop -wait -r "intra_channel_correction('{}','{}',{}); quit"\
          """.format(input_dir, output_dir, disk_size)
start = time.time()
p = subprocess.call(command)
print('Intra-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# INTRA-CHANNEL CORRECTION UNSUPERVISED

input_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
output_dir = r'E:\jahandar\DashData\Injury\unmixed'
brightfield = 11
command = """python 1_PREPROCESSING/inter_channel_correction_unsupervised.py \
             --input_dir {0} \
             --output_dir {1} \
             --brightfield {2} \
          """.format(input_dir, output_dir, brightfield)
p = subprocess.call(command)
print('Inter-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# INTER-CHANNEL CORRECTION SUPERVISED
input_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
output_dir = r'E:\jahandar\DashData\Injury\unmixed'
script = r'E:\jahandar\DashData\Injury\unmixed\supervised.csv'
command = """python inter_channel_correction_supervised.py \
             --input_dir {0} \
             --output_dir {1} \
             --script_file {2} """.format(input_dir, output_dir, script)
p = subprocess.call(command)
print('Inter-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# DETECTION
