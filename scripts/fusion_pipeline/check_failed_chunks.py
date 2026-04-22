"""Check which chunks from a fusion step were never written and emit a retry CSV.

Usage:
    python check_failed_chunks.py --step 1_1
    python check_failed_chunks.py --step 2_3
    python check_failed_chunks.py --step 2_3 --csv csvs/step_2_3_retry.csv

    # Diagnose path/layout issues before a full run:
    python check_failed_chunks.py --step 1_2 --diagnose

The script checks the output zarr store for each row in the CSV. Zarr only creates
chunk files when data is actually written, so a missing chunk file means that
processing tile was never completed.

The retry CSV is written to csvs/step_<X>_retry.csv with job_id re-indexed from 0.
The script also prints the exact sbatch command to resubmit with more time.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

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


def _read_zarr_meta(array_path: str) -> tuple[str, tuple[int, ...]]:
    """Return (layout, chunk_shape) by reading zarr metadata directly.

    Tries zarr v2 (.zarray), zarr v3 (zarr.json), and n5 (attributes.json).
    Falls back to (192,192,192) if nothing is found.
    """
    # zarr v2
    v2_meta = os.path.join(array_path, ".zarray")
    if os.path.exists(v2_meta):
        with open(v2_meta) as f:
            meta = json.load(f)
        return "v2", tuple(meta["chunks"][:3])

    # zarr v3
    v3_meta = os.path.join(array_path, "zarr.json")
    if os.path.exists(v3_meta):
        with open(v3_meta) as f:
            meta = json.load(f)
        chunks = (
            meta.get("chunk_grid", {})
            .get("configuration", {})
            .get("chunk_shape", None)
        )
        if chunks:
            return "v3", tuple(chunks[:3])
        return "v3", (192, 192, 192)

    # n5
    n5_meta = os.path.join(array_path, "attributes.json")
    if os.path.exists(n5_meta):
        with open(n5_meta) as f:
            meta = json.load(f)
        chunks = meta.get("blockSize", None)
        if chunks:
            return "n5", tuple(chunks[:3])
        return "n5", (192, 192, 192)

    return "unknown", (192, 192, 192)


def _chunk_exists(array_path: str, chunk_coords: tuple, layout: str) -> bool:
    """Check one chunk by the correct path for the detected layout."""
    if layout == "v2":
        return os.path.exists(
            os.path.join(array_path, ".".join(str(c) for c in chunk_coords))
        )
    if layout == "v3":
        return os.path.exists(os.path.join(array_path, "c", *map(str, chunk_coords)))
    if layout == "n5":
        # n5: reversed dimension order, slash-separated directories
        return os.path.exists(
            os.path.join(array_path, *map(str, reversed(chunk_coords)))
        )
    # unknown — try all three
    dot = os.path.join(array_path, ".".join(str(c) for c in chunk_coords))
    c_slash = os.path.join(array_path, "c", *map(str, chunk_coords))
    n5_slash = os.path.join(array_path, *map(str, reversed(chunk_coords)))
    return os.path.exists(dot) or os.path.exists(c_slash) or os.path.exists(n5_slash)



def _diagnose(step: str, csv_path: Path, n: int = 5) -> None:
    """Print metadata, first n chunk paths checked, and actual directory listing."""
    df = pd.read_csv(csv_path, nrows=1)
    row = df.iloc[0]
    scale = int(row["scale_number"]) if "scale_number" in row.index else None
    out_path = _output_path(step, scale)
    layout, chunk_size = _read_zarr_meta(out_path)

    print(f"\nArray path : {out_path}")
    print(f"Layout     : {layout}")
    print(f"Chunk size : {chunk_size}")
    print("\nTop-level entries in zarr directory:")
    try:
        entries = sorted(os.listdir(out_path))
        for e in entries:
            print(f"  {e}")
    except Exception as exc:
        print(f"  (could not list directory: {exc})")

    # For v3, show contents of c/ and count total chunk files
    if layout == "v3":
        c_dir = os.path.join(out_path, "c")
        try:
            sub = sorted(os.listdir(c_dir))
            print(f"\nFirst entries in c/ ({len(sub)} total):")
            for e in sub[:10]:
                c_sub = os.path.join(c_dir, e)
                try:
                    sub2 = sorted(os.listdir(c_sub))
                    print(f"  c/{e}/  ({len(sub2)} entries: {sub2[:5]}...)")
                except Exception:
                    print(f"  c/{e}")
            total = sum(
                len(files)
                for _, _, files in os.walk(c_dir)
            )
            print(f"\nTotal chunk files under c/ : {total}")
        except Exception as exc:
            print(f"  (could not list c/: {exc})")

    df_full = pd.read_csv(csv_path, nrows=n)
    print(f"\nPaths checked for first {n} rows:")
    for _, row in df_full.iterrows():
        core_slices = parse_slice_string(row["core_in_result_slices_extended"])
        chunk_coords = tuple(
            s.start // c for s, c in zip(core_slices[:3], chunk_size, strict=False)
        )
        found = _chunk_exists(out_path, chunk_coords, layout)
        if layout == "v2":
            path = os.path.join(out_path, ".".join(str(c) for c in chunk_coords))
        elif layout == "v3":
            path = os.path.join(out_path, "c", *map(str, chunk_coords))
        else:
            path = f"{out_path}/<{layout}>/{chunk_coords}"
        print(f"  {'OK  ' if found else 'MISS'} {path}")


def find_failed_batches(csv_path: Path, chunks_per_task: int) -> list[int]:
    """Return batch IDs whose completion marker is missing."""
    from _parallel_utils import find_incomplete_batches
    return find_incomplete_batches(str(csv_path), chunks_per_task)


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
        default=75,
        help="Chunks per SLURM task (must match original submission)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print detected zarr layout, chunk paths, and directory listing then exit",
    )
    args = parser.parse_args()

    csv_path = args.csv or CSV_DIR / f"step_{args.step}.csv"

    if args.diagnose:
        _diagnose(args.step, csv_path)
        sys.exit(0)

    cpt = args.chunks_per_task
    retry_path = CSV_DIR / f"step_{args.step}_retry.csv"

    print(f"Checking step {args.step} from {csv_path} ...")
    df_full = pd.read_csv(csv_path)
    n_batches = (len(df_full) + cpt - 1) // cpt
    print(f"  Total chunks: {len(df_full)}  |  Batches: {n_batches}  |  CPT: {cpt}")

    failed_batches = find_failed_batches(csv_path, cpt)

    if not failed_batches:
        print("  All batches completed — nothing to retry.")
        sys.exit(0)

    # Collect all chunk rows belonging to failed batches
    failed_rows = []
    for b in failed_batches:
        start = b * cpt
        end = min(start + cpt, len(df_full))
        failed_rows.extend(range(start, end))

    retry_df = df_full.iloc[failed_rows].copy().reset_index(drop=True)
    retry_df["job_id"] = retry_df.index
    retry_df.to_csv(retry_path, index=False)

    n_tasks = (len(retry_df) + cpt - 1) // cpt

    print(f"\n  Failed batches / total : {len(failed_batches)} / {n_batches}")
    print(f"  Failed chunks  / total : {len(failed_rows)} / {len(df_full)}")
    print(f"  Retry CSV              : {retry_path}")
    print(f"\nResubmit with more time (adapt --time and CHUNKS_PER_TASK as needed):")
    print(
        f"  sbatch --array=0-{n_tasks - 1}%5 --time=08:00:00 "
        f"--export=ALL,CHUNKS_PER_TASK={cpt} "
        f"{args.step}_fusion_parallel_submission.sh {retry_path}"
    )
