#!/usr/bin/env bash

mkdir -p ./stdoutput
sbatch --array=0-2 job_cora.sh

#mkdir -p ./stdoutput_1
#sbatch --array=0-300 job1.sh

#mkdir -p ./stdoutput_2
#sbatch --array=0-600 job2.sh