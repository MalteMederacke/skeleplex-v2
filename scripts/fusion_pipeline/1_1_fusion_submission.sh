#!/bin/bash

#SBATCH --job-name=1_1_fusion
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=1_1_fusion_%j.out

module load stack/2024-06 cuda/12.8.0
module load stack/2024-06 python/3.11.6
source skeleplexenv/bin/activate

python 1_1_fusion.py

echo "Job completed: $(date)"
