#!/bin/bash

#SBATCH --job-name=1_3_fusion_p
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/1_3_fusion_p_%j_%a.out

source "$SLURM_SUBMIT_DIR/env.sh"

CSV=${1:-csvs/step_1_3.csv}
CHUNKS_PER_TASK=${CHUNKS_PER_TASK:-50}

python 1_3_fusion_worker.py $CSV $CHUNKS_PER_TASK

echo "Job completed: $(date)"
