import os
import time
import subprocess


###################################################################
# NAMING PROTOCOL: RxCy (x = round number | y = channel number)
###################################################################


# REGISTRATION

# INTRA-CHANNEL CORRECTION
input_dir = r'E:\jahandar\DashData\Injury\original'
output_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
disk_size = [20, 40]
start = time.time()
command = """matlab -minimize -nosplash -nodesktop -wait -r "intra_channel_correction('{}','{}',{}); quit" """.format(input_dir, output_dir, disk_size)
p = subprocess.call(command)
print('Intra-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# INTRA-CHANNEL CORRECTION UNSUPERVISED
input_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
output_dir = r'E:\jahandar\DashData\Injury\unmixed'
brightfield = 1
command = """python inter_channel_correction_unsupervised.py --input_dir {} --output_dir {} --brightfield {} """.format(input_dir, output_dir, brightfield)
p = subprocess.call(command)
print('Inter-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# INTER-CHANNEL CORRECTION SUPERVISED
input_dir = r'E:\jahandar\DashData\Injury\IL_corrected'
output_dir = r'E:\jahandar\DashData\Injury\unmixed'
script = r'E:\jahandar\DashData\Injury\unmixed\unsupervised.csv'
command = """python inter_channel_correction_unsupervised.py --input_dir {} --output_dir {} --brightfield {} """.format(input_dir, output_dir, brightfield)
p = subprocess.call(command)
print('Inter-channel fluorescence correction pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))

# DETECTION
