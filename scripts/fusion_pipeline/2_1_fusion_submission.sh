#!/bin/bash

#SBATCH --job-name=2_1_fusion
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/2_1_fusion_%j_%a.out

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

JOB_INDEX_OFFSET=3  # ADAPT HERE

python 2_1_fusion.py --job-index $SLURM_ARRAY_TASK_ID --job-index-offset $JOB_INDEX_OFFSET

echo "Job completed: $(date)"
