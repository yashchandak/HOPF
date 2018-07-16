#!/usr/bin/env bash
echo "Ensure \n(a) cython_all_setup.py was executed"

mkdir ./stdoutput
sbatch --array=0-25 job0.sh

#mkdir ./stdoutput_1
#sbatch --array=0-300 job1.sh

#mkdir ./stdoutput_2
#sbatch --array=0-600 job2.sh