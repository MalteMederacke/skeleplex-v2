#!/bin/bash

#SBATCH --job-name=2_3_fusion
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/2_3_fusion_%j_%a.out

cd "$(dirname "${BASH_SOURCE[0]}")"

source ../env.sh

JOB_INDEX_OFFSET=3  # ADAPT HERE

python 2_3_fusion.py --job-index $SLURM_ARRAY_TASK_ID --job-index-offset $JOB_INDEX_OFFSET --workers 4

echo "Job completed: $(date)"
