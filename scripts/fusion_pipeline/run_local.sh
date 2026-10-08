#!/bin/bash
# Run the fusion pipeline on a single machine, without SLURM.
#
# USAGE (from this directory, with the SkelePlex environment active):
#   1. adapt _constants.py
#   2. bash run_local.sh [workers]
#
# Every step runs once over all chunks. Completed steps are skipped when the
# script is started again; delete logs/<step>.done to repeat a step, or the
# logs/ directory to start from scratch. Wall-clock times are appended to
# logs/timings.csv.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WORKERS=${1:-4}  # CPU processes for labelling and upscaling

mkdir -p logs
TIMINGS=logs/timings.csv
[ -f "$TIMINGS" ] || echo "step,seconds" > "$TIMINGS"

N_SCALES=$(python -c "from _constants import SCALE_RANGES_MANUAL as s; print(len(s))")

# run <name> <command...> — skipped if logs/<name>.done exists
run() {
    local name=$1
    shift
    if [ -f "logs/$name.done" ]; then
        echo "[skip] $name"
        return
    fi
    echo "[run ] $name"
    local t0
    t0=$(date +%s)
    if ! "$@" > "logs/$name.log" 2>&1; then
        echo "FAILED: $name — last lines of logs/$name.log:" >&2
        tail -n 20 "logs/$name.log" >&2
        exit 1
    fi
    echo "$name,$(( $(date +%s) - t0 ))" >> "$TIMINGS"
    touch "logs/$name.done"
}

# chunks <step> — process all chunks of a chunk-based step in one batch
chunks() {
    local csv=csvs/step_$1.csv
    SLURM_ARRAY_TASK_ID=0 python "$1_fusion_worker.py" "$csv" $(( $(wc -l < "$csv") - 1 ))
}

# per_scale <name> <command...> — run a command once per scale
per_scale() {
    local name=$1
    shift
    for i in $(seq 0 $(( N_SCALES - 1 ))); do
        run "${name}_scale_index_$i" "$@" --job-index "$i"
    done
}

# --- Part 1: radius map and scale map at native resolution ---
run prepare_phase1 python prepare_parallel_fusion.py --phase 1 --force
run 1_1 chunks 1_1
run 1_2 chunks 1_2
run 1_3 chunks 1_3

# --- Part 2: predict the skeleton on every scale ---
per_scale 2_1 python 2_1_fusion.py
run prepare_phase2 python prepare_parallel_fusion.py --phase 2 --force
run 2_2 chunks 2_2
run 2_3 chunks 2_3
run 2_4 chunks 2_4
per_scale 2_4_5 python 2_4_5_repair_breaks.py --workers "$WORKERS"
per_scale 2_5 python 2_5_fusion.py --workers "$WORKERS"

# --- Part 3: fuse the scales, repair breaks, thin ---
run 3 python 3_fusion.py

echo ""
echo "Fusion finished. Timings (s):"
cat "$TIMINGS"
