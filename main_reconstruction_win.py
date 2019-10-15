import os
import time
import subprocess

###################################################################
# NAMING PROTOCOL: RxCy (x = round number | y = channel number)
###################################################################


INPUT_DIR = r'/brazos/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#14_HC_11L/original'
OUTPUT_DIR = r'/brazos/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#14_HC_11L'
MODE = 'supervised'

# for supervised
SCRIPT = r'scripts/20_plex.csv'

# for unsupervised
DEFAULT_CROP = [34000, 8000, 44000, 15000]
BRIGHTFIELD = 11


# create script if not specified
if MODE is 'unsupervised':
    from RECONSTRUCTION.prepare_script import create_script
    create_script(os.path.join(OUTPUT_DIR, 'script.csv'), INPUT_DIR, DEFAULT_CROP, brightfield=BRIGHTFIELD)
    SCRIPT = os.path.join(OUTPUT_DIR, 'script.csv')

# REGSITRATION
input_dir = INPUT_DIR
output_dir = os.path.join(OUTPUT_DIR, 'registered')
command = ' '.join([r"python RECONSTRUCTION/registration.py",
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


# INTRA CHANNEL CORRECTION
ipnut_dir = os.path.join(OUTPUT_DIR, 'registered')
output_dir = os.path.join(OUTPUT_DIR, 'intra_corrected')
disk_size = [20, 40]
command = ' '.join(["matlab -nojvm -nosplash -nodesktop -wait -r",
                    "\"addpath(fullfile(pwd, '1_PREPROCESSING'));",
                    "intra_channel_correction('{}','{}',{}, {}, '{}'); quit\"".format(input_dir, output_dir, disk_size,
                                                                                      BRIGHTFIELD, SCRIPT)])
start = time.time()
p = subprocess.call(command, shell=True)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Intra-channel fluorescence correction pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))


# INTER CHANNEL CORRECTION
input_dir = os.path.join(OUTPUT_DIR, 'intra_corrected')
output_dir = os.path.join(OUTPUT_DIR, 'inter_corrected')
start = time.time()
if MODE == 'unsupervised':
    from RECONSTRUCTION.inter_channel_correction import inter_channel_correct_unsupervised
    inter_channel_correct_unsupervised(input_dir, output_dir, SCRIPT)
elif MODE == 'supervised':
    from RECONSTRUCTION.inter_channel_correction import inter_channel_correct_supervised
    inter_channel_correct_supervised(input_dir, output_dir, SCRIPT)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Inter channel correction finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
