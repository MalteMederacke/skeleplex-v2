"""Prepare chunk CSVs and output zarr stores for the parallel fusion pipeline.

Run phase 1 once (after INPUT_IMAGE exists, before submitting SLURM jobs):
    python prepare_parallel_fusion.py --phase 1

Run phase 2 after 2_1_fusion has finished (scaled images must exist):
    python prepare_parallel_fusion.py --phase 2
"""

import argparse
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    DISTANCE_FIELD_TYPE,
    DISTANCE_FIELD_ZARR,
    INPUT_IMAGE_PATH,
    RADIUS_MAP_PATH,
    SCALE_MAP_PATH,
    SCALE_MAP_PROCESSED_PATH,
    SCALE_RANGES_MANUAL,
    SCALED_IMAGE_ZARR,
    SKELETON_PREDICTIONS_ZARR,
    SKELETONIZED_ON_SCALES_ZARR,
)
from _parallel_utils import get_chunking_df

CSV_DIR = Path(__file__).parent / "csvs"

# Storage chunk size — all zarr arrays are stored at this resolution.
# Do NOT change unless you rebuild all zarr stores from scratch.
STORAGE_CHUNK = (192, 192, 192)

# Processing tile sizes per step — must be multiples of STORAGE_CHUNK so that
# no two concurrent tasks ever write to the same zarr chunk file on disk.
# Larger tiles = fewer SLURM tasks, slower per task but less scheduler overhead.
# Smaller tasks = more parallelism, but each task has more startup cost.
# ADAPT per step based on dataset size and cluster throughput.
PROC_CHUNK = {
    "1_1": (192, 192, 192),   # radius map: high variance, keep small for parallelism
    "1_2": (192, 192, 192),   # scale map
    "1_3": (192, 192, 192),   # processed scale map
    "2_2": (192, 192, 192),   # distance field
    "2_3": (192, 192, 192),   # skeleton prediction (already slow per chunk)
    "2_4": (384, 384, 384),   # mask+threshold+skel: fast, use larger tiles
}


def _save_df(df: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"  CSV exists, skipping: {path}")
        return
    df.to_csv(path, index=False)
    print(f"  Written {len(df)} rows → {path}")

def _open_or_create_zarr(path: str, shape, chunks, dtype) -> None:
    try:
        z = zarr.open_array(path, mode="r")
        if z.shape == shape:
            print(f"  Zarr exists with correct shape, skipping: {path}")
            return
    except (FileNotFoundError, KeyError, zarr.errors.NodeTypeValidationError):
        pass

    zarr.open_array(path, mode="w", shape=shape, chunks=chunks, dtype=dtype)
    print(f"  Created zarr {shape} @ {path}")

def prepare_phase1(force: bool) -> None:
    print("=== Phase 1: steps 1_1, 1_2, 1_3 ===")
    CSV_DIR.mkdir(exist_ok=True)

    img = da.from_zarr(INPUT_IMAGE_PATH)
    shape = img.shape
    print(f"Input image shape: {shape}")

    # --- 1_1: radius map ---
    df = get_chunking_df(img, shape, PROC_CHUNK["1_1"], (30, 30, 30))
    _save_df(df, CSV_DIR / "step_1_1.csv", force)
    _open_or_create_zarr(RADIUS_MAP_PATH, shape, STORAGE_CHUNK, np.float32)

    # --- 1_2: scale map (input = radius map, same shape) ---
    df = get_chunking_df(img, shape, PROC_CHUNK["1_2"], (10, 10, 10))
    _save_df(df, CSV_DIR / "step_1_2.csv", force)
    _open_or_create_zarr(SCALE_MAP_PATH, shape, STORAGE_CHUNK, np.float32)

    # --- 1_3: processed scale map ---
    df = get_chunking_df(img, shape, PROC_CHUNK["1_3"], (20, 20, 20))
    _save_df(df, CSV_DIR / "step_1_3.csv", force)
    _open_or_create_zarr(SCALE_MAP_PROCESSED_PATH, shape, STORAGE_CHUNK, np.float32)


def prepare_phase2(force: bool) -> None:
    print("=== Phase 2: steps 2_2, 2_3, 2_4 ===")
    CSV_DIR.mkdir(exist_ok=True)

    scales = list(SCALE_RANGES_MANUAL.keys())
    dfs_22, dfs_23, dfs_24 = [], [], []

    for scale in scales:
        scaled_path = f"{SCALED_IMAGE_ZARR}/scale{scale}"
        try:
            scaled = da.from_zarr(scaled_path)
        except Exception as e:
            print(f"  WARNING: could not load {scaled_path}: {e}")
            print("  Run 2_1_fusion first, then re-run --phase 2")
            continue

        s = scaled.shape
        print(f"  scale={scale}, shape={s}")

        # 2_2: distance / normal field — border (10,10,10)
        _is_normal = DISTANCE_FIELD_TYPE == "normal_field"
        chunk_22 = (3, *STORAGE_CHUNK) if _is_normal else STORAGE_CHUNK
        out_shape_22 = (3, *s) if _is_normal else s
        df = get_chunking_df(
            scaled, out_shape_22, PROC_CHUNK["2_2"], (10, 10, 10), scale_number=scale
        )
        dfs_22.append(df)
        _open_or_create_zarr(
            f"{DISTANCE_FIELD_ZARR}/scale{scale}_{DISTANCE_FIELD_TYPE}",
            out_shape_22, chunk_22, np.float32,
        )

        # 2_3: skeleton prediction — border (60,60,60)
        df = get_chunking_df(
            scaled, s, PROC_CHUNK["2_3"], (60, 60, 60), scale_number=scale
        )
        dfs_23.append(df)
        _open_or_create_zarr(
            f"{SKELETON_PREDICTIONS_ZARR}/scale{scale}", s, STORAGE_CHUNK, np.float32
        )

        # 2_4: mask + threshold + skeletonize — use skeletonize border (10,10,10)
        df = get_chunking_df(
            scaled, s, PROC_CHUNK["2_4"], (10, 10, 10), scale_number=scale
        )
        dfs_24.append(df)
        _open_or_create_zarr(
            f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale}", s, STORAGE_CHUNK, bool
        )

    def _concat_save(dfs, name):
        if not dfs:
            return
        combined = pd.concat(dfs, ignore_index=True)
        combined["job_id"] = combined.index
        _save_df(combined, CSV_DIR / f"step_{name}.csv", force)

    _concat_save(dfs_22, "2_2")
    _concat_save(dfs_23, "2_3")
    _concat_save(dfs_24, "2_4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing CSVs"
    )
    args = parser.parse_args()

    if args.phase == 1:
        prepare_phase1(args.force)
    else:
        prepare_phase2(args.force)

    print("Done.")
