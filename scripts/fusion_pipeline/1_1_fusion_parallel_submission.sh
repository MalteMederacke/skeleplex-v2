#!/bin/bash

#SBATCH --job-name=1_1_fusion_p
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/1_1_fusion_p_%j_%a.out

module load stack/2024-06 cuda/12.8.0
module load stack/2024-06 python/3.11.6
source skeleplexenv/bin/activate

CHUNKS_PER_TASK=50  # ADAPT HERE

python 1_1_fusion_worker.py csvs/step_1_1.csv $CHUNKS_PER_TASK

echo "Job completed: $(date)"
