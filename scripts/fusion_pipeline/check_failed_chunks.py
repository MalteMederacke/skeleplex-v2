"""Check which chunks from a fusion step were never written and emit a retry CSV.

Usage:
    python check_failed_chunks.py --step 1_1
    python check_failed_chunks.py --step 2_3
    python check_failed_chunks.py --step 2_3 --csv csvs/step_2_3_retry.csv

The script checks the output zarr store for each row in the CSV. Zarr only creates
chunk files when data is actually written, so a missing chunk file means that
processing tile was never completed.

The retry CSV is written to csvs/step_<X>_retry.csv with job_id re-indexed from 0.
The script also prints the exact sbatch command to resubmit with more time.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import zarr

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    DISTANCE_FIELD_ZARR,
    RADIUS_MAP_PATH,
    SCALE_MAP_PATH,
    SCALE_MAP_PROCESSED_PATH,
    SKELETON_PREDICTIONS_ZARR,
    SKELETONIZED_ON_SCALES_ZARR,
)
from _parallel_utils import parse_slice_string

CSV_DIR = Path(__file__).parent / "csvs"


def _output_path(step: str, scale: int | None) -> str:
    if step == "1_1":
        return RADIUS_MAP_PATH
    if step == "1_2":
        return SCALE_MAP_PATH
    if step == "1_3":
        return SCALE_MAP_PROCESSED_PATH
    if step == "2_2":
        return f"{DISTANCE_FIELD_ZARR}/scale{scale}_maxball_2"
    if step == "2_3":
        return f"{SKELETON_PREDICTIONS_ZARR}/scale{scale}"
    if step == "2_4":
        return f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale}"
    raise ValueError(f"Unknown step: {step}")


def _chunk_written(z: zarr.Array, core_slices: tuple) -> bool:
    """Return True if the first zarr chunk covering core_slices exists on disk.

    Zarr v2 DirectoryStore only creates chunk files on write.  Checking the
    first chunk of the core region is sufficient — if the job completed the
    entire tile was flushed together.
    """
    chunk_coords = tuple(
        s.start // c for s, c in zip(core_slices[:3], z.chunks[:3])
    )
    key = ".".join(str(c) for c in chunk_coords)
    return key in z.store


def find_failed(step: str, csv_path: Path) -> list[int]:
    df = pd.read_csv(csv_path)
    zarr_cache: dict[str, zarr.Array] = {}
    failed: list[int] = []

    for idx, row in df.iterrows():
        scale = int(row["scale_number"]) if "scale_number" in row.index else None
        out_path = _output_path(step, scale)

        if out_path not in zarr_cache:
            try:
                zarr_cache[out_path] = zarr.open_array(out_path, mode="r")
            except Exception as e:
                print(f"  WARNING: cannot open {out_path}: {e}")
                print("  Marking all remaining rows as failed.")
                failed.extend(range(idx, len(df)))
                break

        core_slices = parse_slice_string(row["core_in_result_slices_extended"])
        if not _chunk_written(zarr_cache[out_path], core_slices):
            failed.append(idx)

    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["1_1", "1_2", "1_3", "2_2", "2_3", "2_4"],
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input CSV to check (default: csvs/step_<STEP>.csv)",
    )
    parser.add_argument(
        "--chunks-per-task",
        type=int,
        default=50,
        help="Chunks per SLURM task for the retry (used in printed sbatch command)",
    )
    args = parser.parse_args()

    csv_path = args.csv or CSV_DIR / f"step_{args.step}.csv"
    retry_path = CSV_DIR / f"step_{args.step}_retry.csv"

    print(f"Checking step {args.step} from {csv_path} ...")
    df_full = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df_full)}")

    failed = find_failed(args.step, csv_path)

    if not failed:
        print("  All chunks written — nothing to retry.")
        sys.exit(0)

    retry_df = df_full.iloc[failed].copy().reset_index(drop=True)
    retry_df["job_id"] = retry_df.index
    retry_df.to_csv(retry_path, index=False)

    cpt = args.chunks_per_task
    n_tasks = (len(retry_df) + cpt - 1) // cpt

    print(f"\n  Failed / total : {len(failed)} / {len(df_full)}")
    print(f"  Retry CSV      : {retry_path}")
    print(f"\nResubmit with more time (adapt --time and CHUNKS_PER_TASK as needed):")
    print(
        f"  sbatch --array=0-{n_tasks - 1}%5 --time=08:00:00 "
        f"--export=ALL,CHUNKS_PER_TASK={cpt} "
        f"{args.step}_fusion_parallel_submission.sh {retry_path}"
    )
