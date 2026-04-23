#!/bin/bash

#SBATCH --job-name=2_5_fusion
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/2_5_fusion_%j_%a.out

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"

python 2_5_fusion.py --job-index $SLURM_ARRAY_TASK_ID --workers 4

echo "Job completed: $(date)"
