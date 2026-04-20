#!/bin/bash

#SBATCH --job-name=3_fusion
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=55G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/3_fusion_%j.out

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

python 3_fusion.py

echo "Job completed: $(date)"
