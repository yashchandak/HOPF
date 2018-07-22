#!/usr/bin/env bash

#mkdir -p ./stdoutput
#sbatch --array=0-2 job_cora.sh

mkdir -p ./stdoutput_1
sbatch --array=0-1000 job1.sh

#mkdir -p ./stdoutput_2
#sbatch --array=0-1000 job2.sh

#mkdir -p ./stdoutput_3
#sbatch --array=0-1000 job3.sh
#
#mkdir -p ./stdoutput_4
#sbatch --array=0-1000 job4.sh