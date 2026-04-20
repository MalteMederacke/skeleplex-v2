#!/bin/bash

#SBATCH --job-name=2_4_fusion
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
#SBATCH --output=logs/2_4_fusion_%j_%a.out

cd "$(dirname "${BASH_SOURCE[0]}")"

source ../env.sh

JOB_INDEX_OFFSET=3  # ADAPT HERE

python 2_4_fusion.py --job-index $SLURM_ARRAY_TASK_ID --job-index-offset $JOB_INDEX_OFFSET

echo "Job completed: $(date)"
