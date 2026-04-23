"""Shared utilities for parallel chunk-based fusion processing."""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def write_batch_marker(csv_path: str, batch_id: int) -> None:
    """Write a sentinel file marking this batch as successfully completed."""
    marker_dir = Path(csv_path).parent / "markers"
    marker_dir.mkdir(exist_ok=True)
    stem = Path(csv_path).stem
    (marker_dir / f"{stem}_batch_{batch_id}.done").touch()


def find_incomplete_batches(csv_path: str, chunks_per_task: int) -> list[int]:
    """Return batch IDs that have no completion marker."""
    df = pd.read_csv(csv_path)
    n_batches = (len(df) + chunks_per_task - 1) // chunks_per_task
    marker_dir = Path(csv_path).parent / "markers"
    stem = Path(csv_path).stem
    return [
        b for b in range(n_batches)
        if not (marker_dir / f"{stem}_batch_{b}.done").exists()
    ]


def get_chunking_df(
    input_array,
    output_shape: tuple,
    chunk_shape: tuple[int, int, int],
    extra_border: tuple[int, int, int],
    scale_number: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per chunk describing all relevant slices.

    Parameters
    ----------
    input_array : array-like with .shape
        Used only to read the 3-D spatial shape.
    output_shape : tuple
        Full shape of the output array (may have extra leading dims for channels).
    chunk_shape : tuple[int, int, int]
        Tile size for processing.
    extra_border : tuple[int, int, int]
        Overlap border added around each tile when reading.
    scale_number : int or None
        If provided, a ``scale_number`` column is added to every row.
    """
    if len(chunk_shape) != 3 or len(extra_border) != 3:
        raise ValueError("chunk_shape and extra_border must be 3-tuples")

    array_shape = input_array.shape
    if len(array_shape) < 3:
        raise ValueError("Input array must be at least 3-D")

    n_chunks = tuple(int(np.ceil(array_shape[i] / chunk_shape[i])) for i in range(3))

    rows = []
    for i in range(n_chunks[0]):
        for j in range(n_chunks[1]):
            for k in range(n_chunks[2]):
                core_start = (
                    i * chunk_shape[0],
                    j * chunk_shape[1],
                    k * chunk_shape[2],
                )
                core_end = (
                    min((i + 1) * chunk_shape[0], array_shape[0]),
                    min((j + 1) * chunk_shape[1], array_shape[1]),
                    min((k + 1) * chunk_shape[2], array_shape[2]),
                )
                core_slice = tuple(slice(core_start[d], core_end[d]) for d in range(3))

                expanded_start = tuple(
                    max(0, core_start[d] - extra_border[d]) for d in range(3)
                )
                expanded_end = tuple(
                    min(array_shape[d], core_end[d] + extra_border[d]) for d in range(3)
                )
                expanded_slice = tuple(
                    slice(expanded_start[d], expanded_end[d]) for d in range(3)
                )

                actual_border_before = tuple(
                    core_start[d] - expanded_start[d] for d in range(3)
                )
                core_in_result = [
                    slice(
                        actual_border_before[d],
                        actual_border_before[d] + (core_end[d] - core_start[d]),
                    )
                    for d in range(3)
                ]

                n_extra_dims = len(output_shape) - 3
                if n_extra_dims > 0:
                    extra_slices = [slice(0, output_shape[idx]) for idx in range(n_extra_dims)]
                    core_in_result = extra_slices + core_in_result
                    core_slice_extended = extra_slices + list(core_slice)
                else:
                    core_slice_extended = list(core_slice)

                row = {
                    "input_slices": str(tuple(core_slice)),
                    "expanded_slices": str(tuple(expanded_slice)),
                    "core_in_result_slices": str(tuple(core_in_result)),
                    "core_in_result_slices_extended": str(tuple(core_slice_extended)),
                }
                if scale_number is not None:
                    row["scale_number"] = scale_number
                rows.append(row)

    df = pd.DataFrame(rows)
    df["job_id"] = df.index
    return df


def parse_slice_string(slice_str: str) -> tuple:
    """Convert a string like ``(slice(0, 192, None), ...)`` back to slice objects."""
    slice_pattern = r"slice\(([^)]+)\)"
    matches = re.findall(slice_pattern, slice_str)
    slices = []
    for match in matches:
        args = [a.strip() for a in match.split(",")]
        parsed = [None if a == "None" else int(a) for a in args]
        if len(parsed) == 2:
            slices.append(slice(parsed[0], parsed[1]))
        elif len(parsed) == 3:
            slices.append(slice(parsed[0], parsed[1], parsed[2]))
        else:
            slices.append(slice(parsed[0]))
    return tuple(slices)


def get_slices_for_chunk(job_df: pd.DataFrame, chunk_id: int):
    """Return the four slice tuples for a given chunk row.

    Returns
    -------
    (expanded_slices, core_in_result_slices, core_in_result_slices_extended,
     scale_number_or_None)
    """
    row = job_df.iloc[chunk_id]
    expanded = parse_slice_string(row["expanded_slices"])
    core_in_result = parse_slice_string(row["core_in_result_slices"])
    core_out = parse_slice_string(row["core_in_result_slices_extended"])
    scale = int(row["scale_number"]) if "scale_number" in row.index else None
    return expanded, core_in_result, core_out, scale
