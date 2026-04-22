#!/bin/bash

#SBATCH --job-name=2_3_fusion_p
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/2_3_fusion_p_%j_%a.out

source "$SLURM_SUBMIT_DIR/env.sh"

CSV=${1:-csvs/step_2_3.csv}
CHUNKS_PER_TASK=${CHUNKS_PER_TASK:-10}  # fewer chunks — inference is slow per chunk

python 2_3_fusion_worker.py $CSV $CHUNKS_PER_TASK

echo "Job completed: $(date)"
