#!/bin/bash

#SBATCH --job-name=1_2_fusion
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=55G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/1_2_fusion_%j.out

cd "$SLURM_SUBMIT_DIR"

source ../env.sh

python 1_2_fusion.py

echo "Job completed: $(date)"
