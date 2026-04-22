#!/bin/bash
# Submit one fusion step as a SLURM array job.
#
# Usage:
#   ./submit_step.sh <step> [csv] [chunks_per_task] [max_concurrent]
#
# Examples:
#   ./submit_step.sh 1_2
#   ./submit_step.sh 1_2 csvs/step_1_2_retry.csv 75 10
#
# Arguments:
#   step             One of: 1_1 1_2 1_3 2_2 2_3 2_4
#   csv              CSV file to process (default: csvs/step_<STEP>.csv)
#   chunks_per_task  Chunks per SLURM task       (default: 50)
#   max_concurrent   Max simultaneous array tasks (default: 20)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STEP=${1:?Usage: $0 <step> [csv] [chunks_per_task] [max_concurrent]}
CSV=${2:-csvs/step_${STEP}.csv}
CPT=${3:-50}
MAX_CONCURRENT=${4:-20}

if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV" >&2
    exit 1
fi

# Count data rows (subtract header)
N_CHUNKS=$(( $(wc -l < "$CSV") - 1 ))
N_TASKS=$(( (N_CHUNKS + CPT - 1) / CPT ))
ARRAY_END=$(( N_TASKS - 1 ))

echo "Step            : $STEP"
echo "CSV             : $CSV"
echo "Chunks          : $N_CHUNKS"
echo "Chunks per task : $CPT"
echo "Array tasks     : $N_TASKS  (0-${ARRAY_END}%${MAX_CONCURRENT})"
echo ""

sbatch \
    --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
    --export="ALL,CHUNKS_PER_TASK=${CPT}" \
    "${STEP}_fusion_parallel_submission.sh" "$CSV"
