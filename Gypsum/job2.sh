#!/bin/bash
#
#SBATCH --job-name=highD_AFE
#SBATCH --output=./stdoutput_2/highD_2_%A_%a.out # output file
#SBATCH --error=./stdoutput_2/highD_2.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=0-11:59       # Maximum runtime in D-HH:MM
#SBATCH --cpus-per-task=1    # CPU cores per process
#SBATCH --mem-per-cpu=500    # Memory in MB per cpu allocated
#SBATCH --mail-type=END
#SBATCH --mail-user=ychandak@cs.umass.edu

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export PATH="/home/ychandak/miniconda3/envs/pytorch/bin:$PATH"
export PYTHONPATH="/home/ychandak/RL:$PYTHONPATH"
source activate pytorch

srun python /home/ychandak/RL/src/run.py --inc $SLURM_ARRAY_TASK_ID --base 0 --hyper 2

sleep 1
exit