#!/usr/bin/env bash

mkdir ./stdoutput
sbatch --array=0-2 job_cora.sh

#mkdir ./stdoutput_1
#sbatch --array=0-300 job1.sh

#mkdir ./stdoutput_2
#sbatch --array=0-600 job2.sh