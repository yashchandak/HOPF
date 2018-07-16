#!/bin/bash
#
#SBATCH --job-name=py
#SBATCH --error=./stdoutput/highD_0.err        # File to which STDERR will be written
#SBATCH --partition=titanx-short    # Partition to submit to
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=ychandak@cs.umass.edu

export PATH="/home/ychandak/miniconda3/envs/tf/bin:$PATH"
export PYTHONPATH="/home/ychandak/HOPF:$PYTHONPATH"
source activate tf

srun python /home/ychandak/HOPF/Swarm/Hyper/hyperparam_sweep0.py --inc $SLURM_ARRAY_TASK_ID --base 0

sleep 1
exit