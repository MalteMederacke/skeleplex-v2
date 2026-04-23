#!/bin/bash
# Parallel fusion pipeline — chunk-based steps run as SLURM array jobs,
# each task processing CHUNKS_PER_TASK chunks before finishing.
#
# USAGE (run from the fusion_pipeline directory on the login node):
#   1. python prepare_parallel_fusion.py --phase 1   # fast, run locally first
#   2. bash submit_all_parallel.sh
#
# ADAPT: ARRAY_RANGE  — number of scales (e.g. "0-3" for 4 scales)
# ADAPT: CHUNKS_PER_TASK — chunks processed per SLURM task
# ADAPT: CONCURRENCY — max tasks running at once per step

ARRAY_RANGE="0-3"       # ADAPT HERE — for scale-level jobs (2_1, 2_5)
CHUNKS_PER_TASK=50      # ADAPT HERE — must match value in *_submission.sh scripts
CHUNKS_PER_TASK_23=10   # ADAPT HERE — for step 2_3 (ML inference, slower per chunk)
CONCURRENCY=5           # ADAPT HERE — max parallel tasks per step

mkdir -p logs

# -----------------------------------------------------------------------
# Helper: compute number of SLURM tasks needed given CSV row count
# -----------------------------------------------------------------------
n_tasks() {
    local n_chunks=$1
    local cpt=$2
    echo $(( (n_chunks + cpt - 1) / cpt ))
}

# -----------------------------------------------------------------------
# Phase 1 CSV sizes (prepared locally before this script runs)
# -----------------------------------------------------------------------
N_1_1=$(( $(wc -l < csvs/step_1_1.csv) - 1 ))
N_1_2=$(( $(wc -l < csvs/step_1_2.csv) - 1 ))
N_1_3=$(( $(wc -l < csvs/step_1_3.csv) - 1 ))

T_1_1=$(n_tasks $N_1_1 $CHUNKS_PER_TASK)
T_1_2=$(n_tasks $N_1_2 $CHUNKS_PER_TASK)
T_1_3=$(n_tasks $N_1_3 $CHUNKS_PER_TASK)

echo "Chunk counts  — 1_1: $N_1_1  1_2: $N_1_2  1_3: $N_1_3"
echo "Task counts   — 1_1: $T_1_1  1_2: $T_1_2  1_3: $T_1_3  (${CHUNKS_PER_TASK} chunks/task)"
echo "Submitting parallel fusion pipeline..."

# -----------------------------------------------------------------------
# Part 1 — parallel chunk arrays
# -----------------------------------------------------------------------
JOB_1_1=$(sbatch --parsable \
    --array=0-$((T_1_1-1))%${CONCURRENCY} \
    1_1_fusion_parallel_submission.sh)
echo "  1_1_fusion_p submitted: $JOB_1_1"

JOB_1_2=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_1 \
    --array=0-$((T_1_2-1))%${CONCURRENCY} \
    1_2_fusion_parallel_submission.sh)
echo "  1_2_fusion_p submitted: $JOB_1_2"

JOB_1_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_2 \
    --array=0-$((T_1_3-1))%${CONCURRENCY} \
    1_3_fusion_parallel_submission.sh)
echo "  1_3_fusion_p submitted: $JOB_1_3"

# -----------------------------------------------------------------------
# Part 2.1 — scale the image (one task per scale, unchanged)
# -----------------------------------------------------------------------
JOB_2_1=$(sbatch --parsable \
    --dependency=afterok:$JOB_1_3 \
    --array=$ARRAY_RANGE \
    2_1_fusion_submission.sh)
echo "  2_1_fusion submitted:   $JOB_2_1  (array $ARRAY_RANGE)"

# -----------------------------------------------------------------------
# Phase 2 preparation — runs after 2_1, creates CSVs + output zarrs
# -----------------------------------------------------------------------
JOB_PREP2=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_1 \
    0_prepare_phase2_submission.sh)
echo "  prepare_phase2 submitted: $JOB_PREP2"

# -----------------------------------------------------------------------
# Parts 2.2 – 2.4 — conservative upper bound on task count.
# Workers exit immediately if their batch_id is out of range.
# ADAPT MAX_CHUNKS if your dataset is very large.
# -----------------------------------------------------------------------
MAX_CHUNKS=9999
MAX_TASKS=$(( (MAX_CHUNKS + CHUNKS_PER_TASK - 1) / CHUNKS_PER_TASK ))
MAX_TASKS_23=$(( (MAX_CHUNKS + CHUNKS_PER_TASK_23 - 1) / CHUNKS_PER_TASK_23 ))

JOB_2_2=$(sbatch --parsable \
    --dependency=afterok:$JOB_PREP2 \
    --array=0-${MAX_TASKS}%${CONCURRENCY} \
    2_2_fusion_parallel_submission.sh)
echo "  2_2_fusion_p submitted: $JOB_2_2  (max ${MAX_TASKS} tasks)"

JOB_2_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_2 \
    --array=0-${MAX_TASKS_23}%${CONCURRENCY} \
    2_3_fusion_parallel_submission.sh)
echo "  2_3_fusion_p submitted: $JOB_2_3  (max ${MAX_TASKS_23} tasks)"

JOB_2_4=$(sbatch --parsable \
    --dependency=afterok:$JOB_2_3 \
    --array=0-${MAX_TASKS}%${CONCURRENCY} \
    2_4_fusion_parallel_submission.sh)
echo "  2_4_fusion_p submitted: $JOB_2_4  (max ${MAX_TASKS} tasks)"

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
echo "All jobs submitted."
echo "  1_1_fusion_p   : $JOB_1_1"
echo "  1_2_fusion_p   : $JOB_1_2"
echo "  1_3_fusion_p   : $JOB_1_3"
echo "  2_1_fusion     : $JOB_2_1"
echo "  prepare_phase2 : $JOB_PREP2"
echo "  2_2_fusion_p   : $JOB_2_2"
echo "  2_3_fusion_p   : $JOB_2_3"
echo "  2_4_fusion_p   : $JOB_2_4"
echo "  2_5_fusion     : $JOB_2_5"
echo "  3_fusion       : $JOB_3"
