#!/bin/bash

#SBATCH --job-name=2_2_fusion
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/2_2_fusion_%j_%a.out

module load stack/2024-06 cuda/12.8.0
module load stack/2024-06 python/3.11.6
source skeleplexenv/bin/activate

JOB_INDEX_OFFSET=4

python 2_2_fusion.py --job-index $SLURM_ARRAY_TASK_ID --job-index-offset $JOB_INDEX_OFFSET

echo "Job completed: $(date)"
