#!/bin/bash

#SBATCH --job-name=2_4_5_repair
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/2_4_5_repair_%j_%a.out

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"

python 2_4_5_repair_breaks.py --job-index $SLURM_ARRAY_TASK_ID --workers 8

echo "Job completed: $(date)"
