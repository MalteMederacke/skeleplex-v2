#!/bin/bash

#SBATCH --job-name=1_3_fusion
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/1_3_fusion_%j.out

cd "$SLURM_SUBMIT_DIR"

source ../env.sh

python 1_3_fusion.py

echo "Job completed: $(date)"
