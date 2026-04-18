#!/bin/bash
# Parallel fusion pipeline — all chunk-based steps run as SLURM array jobs
# capped at 5 concurrent tasks per step.
#
# USAGE (run from the fusion_pipeline directory on the login node):
#   1. python prepare_parallel_fusion.py --phase 1   # fast, run locally first
#   2. bash submit_all_parallel.sh
#
# ADAPT: set ARRAY_RANGE to match your number of scales (e.g. "0-3" for 4 scales).
# ADAPT: set JOB_INDEX_OFFSET inside each *_submission.sh (default: 3).

ARRAY_RANGE="0-3"   # ADAPT HERE — for scale-level jobs (2_1, 2_5)
CONCURRENCY=5       # max parallel chunk workers per step

mkdir -p logs

# -----------------------------------------------------------------------
# Phase 1 CSV sizes (prepared locally before this script runs)
# -----------------------------------------------------------------------
N_1_1=$(( $(wc -l < csvs/step_1_1.csv) - 1 ))
N_1_2=$(( $(wc -l < csvs/step_1_2.csv) - 1 ))
N_1_3=$(( $(wc -l < csvs/step_1_3.csv) - 1 ))

echo "Chunk counts — 1_1: $N_1_1  1_2: $N_1_2  1_3: $N_1_3"
echo "Submitting parallel fusion pipeline..."

# -----------------------------------------------------------------------
# Part 1 — parallel chunk arrays (all scales in one shot, no scale loop)
# -----------------------------------------------------------------------
JOB_1_1=$(sbatch --parsable \
    --array=0-$((N_1_1-1))%${CONCURRENCY} \
    1_1_fusion_parallel_submission.sh)
echo "  1_1_fusion_p submitted: $JOB_1_1  (${N_1_1} chunks)"

JOB_1_2=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_1 \
    --array=0-$((N_1_2-1))%${CONCURRENCY} \
    1_2_fusion_parallel_submission.sh)
echo "  1_2_fusion_p submitted: $JOB_1_2  (${N_1_2} chunks)"

JOB_1_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_2 \
    --array=0-$((N_1_3-1))%${CONCURRENCY} \
    1_3_fusion_parallel_submission.sh)
echo "  1_3_fusion_p submitted: $JOB_1_3  (${N_1_3} chunks)"

# -----------------------------------------------------------------------
# Part 2.1 — scale the image (one task per scale, unchanged)
# -----------------------------------------------------------------------
JOB_2_1=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_3 \
    --array=$ARRAY_RANGE \
    2_1_fusion_submission.sh)
echo "  2_1_fusion submitted:   $JOB_2_1  (array $ARRAY_RANGE)"

# -----------------------------------------------------------------------
# Phase 2 preparation — needs scaled images from 2_1
# -----------------------------------------------------------------------
JOB_PREP2=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_1 \
    0_prepare_phase2_submission.sh)
echo "  prepare_phase2 submitted: $JOB_PREP2"

# -----------------------------------------------------------------------
# Parts 2.2 – 2.4 — parallel chunk arrays across all scales combined.
# The array upper bound is set conservatively; workers exit immediately
# if their chunk_id exceeds the actual CSV row count.
# ADAPT MAX_CHUNKS if your dataset has more than 9999 chunks total.
# -----------------------------------------------------------------------
MAX_CHUNKS=9999

JOB_2_2=$(sbatch --parsable \
    --dependency=afterok:$JOB_PREP2 \
    --array=0-${MAX_CHUNKS}%${CONCURRENCY} \
    2_2_fusion_parallel_submission.sh)
echo "  2_2_fusion_p submitted: $JOB_2_2"

JOB_2_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_2 \
    --array=0-${MAX_CHUNKS}%${CONCURRENCY} \
    2_3_fusion_parallel_submission.sh)
echo "  2_3_fusion_p submitted: $JOB_2_3"

JOB_2_4=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_3 \
    --array=0-${MAX_CHUNKS}%${CONCURRENCY} \
    2_4_fusion_parallel_submission.sh)
echo "  2_4_fusion_p submitted: $JOB_2_4"

# -----------------------------------------------------------------------
# Part 2.5 — upscale (one task per scale, unchanged)
# -----------------------------------------------------------------------
JOB_2_5=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_4 \
    --array=$ARRAY_RANGE \
    2_5_fusion_submission.sh)
echo "  2_5_fusion submitted:   $JOB_2_5  (array $ARRAY_RANGE)"

# -----------------------------------------------------------------------
# Part 3 — fuse + repair + final skeletonize (single job, unchanged)
# -----------------------------------------------------------------------
JOB_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_5 \
    3_fusion_submission.sh)
echo "  3_fusion submitted:     $JOB_3"

echo ""
echo "All jobs submitted. Summary:"
echo "  1_1_fusion_p   : $JOB_1_1  (${N_1_1} chunks, max ${CONCURRENCY} concurrent)"
echo "  1_2_fusion_p   : $JOB_1_2  (${N_1_2} chunks, max ${CONCURRENCY} concurrent)"
echo "  1_3_fusion_p   : $JOB_1_3  (${N_1_3} chunks, max ${CONCURRENCY} concurrent)"
echo "  2_1_fusion     : $JOB_2_1  (array $ARRAY_RANGE)"
echo "  prepare_phase2 : $JOB_PREP2"
echo "  2_2_fusion_p   : $JOB_2_2  (max ${CONCURRENCY} concurrent)"
echo "  2_3_fusion_p   : $JOB_2_3  (max ${CONCURRENCY} concurrent)"
echo "  2_4_fusion_p   : $JOB_2_4  (max ${CONCURRENCY} concurrent)"
echo "  2_5_fusion     : $JOB_2_5  (array $ARRAY_RANGE)"
echo "  3_fusion       : $JOB_3"
