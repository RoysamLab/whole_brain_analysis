import os
import subprocess

# MATLABROOT = "C:\Program Files\MATLAB\R2018b"

# create a new conda environment
# os.system('conda create -n brain python=3.6 --yes')

# install required libraries
cmds = ['conda activate brain',
        'pip install pandas',
        'conda install --yes scikit-learn',
        'conda install --yes -c conda-forge scikit-image=0.16.1',
        'conda install --yes -c conda-forge tifffile',
        'pip install tensorflow-gpu==1.9.0',
        'pip install cython',
        'pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"',
        'pip install progressbar2',
        'pip install requests',
        'pip install opencv-python',
        'pip install keras==2.2.0',
        'pip install h5py',
        'pip install tqdm',
        'conda deactivate']
subprocess.call(' && '.join(cmds), shell=True)

# # install Matlab engine
# if os.path.isdir(MATLABROOT):
#     os.chdir(os.path.join(MATLABROOT, 'extern', 'engines', 'python'))
#     cmds = ['conda activate brain',
#             'python setup.py install',
#             'conda deactivate']
#     subprocess.call(' && '.join(cmds), shell=True)

