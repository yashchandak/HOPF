#!/bin/bash
#
#SBATCH --job-name=py
#SBATCH --error=./stdoutput/HOPF.err        # File to which STDERR will be written
#SBATCH --partition=titanx-short    # Partition to submit to
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=ychandak@cs.umass.edu

export PATH="/home/ychandak/miniconda3/envs/tf/bin:$PATH"
export PYTHONPATH="/home/ychandak/HOPF:$PYTHONPATH"
source activate tf-1.3

srun python /home/ychandak/HOPF/Swarm/Hyper/cora_hyper.py --inc $SLURM_ARRAY_TASK_ID --base 0

sleep 1
exit