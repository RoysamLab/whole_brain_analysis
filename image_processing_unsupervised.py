import os
import time
import subprocess


###################################################################
# NAMING PROTOCOL: RxCy (x = round number | y = channel number)
###################################################################

INPUT_DIR = r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L\original'
OUTPUT_DIR = r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L\unsupervised'
BRIGHTFIELD = 11

# REGISTRATION
input_dir = INPUT_DIR
output_dir = os.path.join(OUTPUT_DIR, 'registered')
command = ' '.join([r"python 1_PREPROCESSING/registration.py",
                   "--input_dir={}".format(input_dir),
                   "--output_dir={}".format(output_dir),
                   "--maxtrials=100",
                   "--tiling='400,800'",
                   "--imadjust=True",
                   "--demo=True",
                   "-nk 30000"])
start = time.time()
p = subprocess.call(command, shell=True)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Registration pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))

# INTRA-CHANNEL CORRECTION
input_dir = os.path.join(OUTPUT_DIR, 'registered')
output_dir = os.path.join(OUTPUT_DIR, 'IL_corrected')
disk_size = [20, 40]
command = ' '.join(["matlab -nojvm -nosplash -nodesktop -wait -r",
                    "\"addpath(fullfile(pwd, '1_PREPROCESSING'));",
                    "intra_channel_correction('{}','{}',{}, {}); quit\"".format(input_dir, output_dir, disk_size, BRIGHTFIELD)])
start = time.time()
p = subprocess.call(command, shell=True)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Intra-channel fluorescence correction pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))

# INTER-CHANNEL CORRECTION UNSUPERVISED
input_dir = os.path.join(OUTPUT_DIR, 'IL_corrected')
output_dir = os.path.join(OUTPUT_DIR, 'unmixed')
command = ' '.join(["python 1_PREPROCESSING/inter_channel_correction_unsupervised.py",
                    "--input_dir={}".format(input_dir),
                    "--output_dir={}".format(output_dir),
                     "--brightfield={}".format(BRIGHTFIELD)])
start = time.time()
p = subprocess.call(command, shell=True)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Inter-channel fluorescence correction pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))