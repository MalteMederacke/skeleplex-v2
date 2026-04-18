#!/bin/bash

#SBATCH --job-name=prepare_phase2
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/prepare_phase2_%j.out

module load stack/2024-06 python/3.11.6
source skeleplexenv/bin/activate

python prepare_parallel_fusion.py --phase 2

echo "Phase 2 preparation completed: $(date)"
