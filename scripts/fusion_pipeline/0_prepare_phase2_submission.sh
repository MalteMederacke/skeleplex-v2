#!/bin/bash

#SBATCH --job-name=prepare_phase2
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/prepare_phase2_%j.out

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"

python prepare_parallel_fusion.py --phase 2

echo "Phase 2 preparation completed: $(date)"
