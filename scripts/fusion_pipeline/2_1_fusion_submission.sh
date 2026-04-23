#!/bin/bash

#SBATCH --job-name=2_1_fusion
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/2_1_fusion_%j_%a.out

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"

python 2_1_fusion.py --job-index $SLURM_ARRAY_TASK_ID

echo "Job completed: $(date)"
