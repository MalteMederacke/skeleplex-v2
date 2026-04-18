#!/bin/bash

#SBATCH --job-name=2_4_fusion_p
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/2_4_fusion_p_%j_%a.out

module load stack/2024-06 python/3.11.6
source skeleplexenv/bin/activate

python 2_4_fusion_worker.py csvs/step_2_4.csv

echo "Job completed: $(date)"
