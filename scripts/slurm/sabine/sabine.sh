#!/bin/bash


module load Anaconda3
module load matlab


srun -t 1-00:00:00 -N 1 -p medium --mem=240gb --x11=first --pty /bin/bash -l
