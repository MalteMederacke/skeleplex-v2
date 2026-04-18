#!/bin/bash
# Master submission script for the full fusion pipeline.
# Jobs are chained via SLURM dependencies so each step waits for the previous to finish.
#
# ADAPT: Set ARRAY_RANGE to match your number of scales (e.g. "0-3" for 4 scales).
# ADAPT: Set JOB_INDEX_OFFSET inside each *_submission.sh (default: 3).
#        With ARRAY_RANGE="0-3" and JOB_INDEX_OFFSET=3, the processed scales are -3,-2,-1,0.

ARRAY_RANGE="0-3"  # ADAPT HERE

mkdir -p logs
echo "Submitting Fusion Pipeline (array range: ${ARRAY_RANGE})..."

# --- Part I ---
JOB_1_1=$(sbatch --parsable 1_1_fusion_submission.sh)
echo "  1_1_fusion submitted: $JOB_1_1"

JOB_1_2=$(sbatch --parsable --dependency=afterok:$JOB_1_1 1_2_fusion_submission.sh)
echo "  1_2_fusion submitted: $JOB_1_2"

JOB_1_3=$(sbatch --parsable --dependency=afterok:$JOB_1_2 1_3_fusion_submission.sh)
echo "  1_3_fusion submitted: $JOB_1_3"

# --- Part II (array jobs, one task per scale) ---
JOB_2_1=$(sbatch --parsable --dependency=afterok:$JOB_1_3 --array=$ARRAY_RANGE 2_1_fusion_submission.sh)
echo "  2_1_fusion array submitted: $JOB_2_1"

JOB_2_2=$(sbatch --parsable --dependency=afterok:$JOB_2_1 --array=$ARRAY_RANGE 2_2_fusion_submission.sh)
echo "  2_2_fusion array submitted: $JOB_2_2"

JOB_2_3=$(sbatch --parsable --dependency=afterok:$JOB_2_2 --array=$ARRAY_RANGE 2_3_fusion_submission.sh)
echo "  2_3_fusion array submitted: $JOB_2_3"

JOB_2_4=$(sbatch --parsable --dependency=afterok:$JOB_2_3 --array=$ARRAY_RANGE 2_4_fusion_submission.sh)
echo "  2_4_fusion array submitted: $JOB_2_4"

JOB_2_5=$(sbatch --parsable --dependency=afterok:$JOB_2_4 --array=$ARRAY_RANGE 2_5_fusion_submission.sh)
echo "  2_5_fusion array submitted: $JOB_2_5"

# --- Part III ---
JOB_3=$(sbatch --parsable --dependency=afterok:$JOB_2_5 3_fusion_submission.sh)
echo "  3_fusion submitted: $JOB_3"

echo ""
echo "All jobs submitted. Summary:"
echo "  1_1_fusion : $JOB_1_1"
echo "  1_2_fusion : $JOB_1_2"
echo "  1_3_fusion : $JOB_1_3"
echo "  2_1_fusion : $JOB_2_1  (array $ARRAY_RANGE)"
echo "  2_2_fusion : $JOB_2_2  (array $ARRAY_RANGE)"
echo "  2_3_fusion : $JOB_2_3  (array $ARRAY_RANGE)"
echo "  2_4_fusion : $JOB_2_4  (array $ARRAY_RANGE)"
echo "  2_5_fusion : $JOB_2_5  (array $ARRAY_RANGE)"
echo "  3_fusion   : $JOB_3"
