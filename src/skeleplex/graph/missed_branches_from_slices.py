"""Detect potentially missed daughter branches from orthogonal cross-section slices.

Along a well-reconstructed branch the segmentation cross-section profile follows:
  complex shape (parent junction) → simple elliptic/round → complex (daughter junctions)

Irregularities in the *interior* portion of that profile suggest that a daughter branch
was missed during skeletonisation. Three independent signals are checked:

  - **low solidity**: the cross-section deviates from a convex shape (branching point)
  - **multiple components**: the cross-section captures disjoint objects
  - **area jump**: the cross-section area suddenly increases (extra branch in the plane)
"""

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


def compute_slice_metrics(segmentation_slice: np.ndarray) -> dict:
    """Compute shape metrics for a single 2D segmentation cross-section.

    Parameters
    ----------
    segmentation_slice : np.ndarray
        2D array (binary or interpolated float). Values > 0.5 are foreground.

    Returns
    -------
    dict
        Keys: area, n_components, solidity, eccentricity, circularity.
        NaN values indicate the slice was empty or the metric could not be computed.
    """
    binary = (segmentation_slice > 0.5).astype(np.uint8)
    area = int(binary.sum())

    if area == 0:
        return {
            "area": 0,
            "n_components": 0,
            "solidity": np.nan,
            "eccentricity": np.nan,
            "circularity": np.nan,
        }

    labeled = label(binary)
    n_components = int(labeled.max())

    largest = max(regionprops(labeled), key=lambda p: p.area)
    perimeter = largest.perimeter
    circularity = (4 * np.pi * largest.area / perimeter**2) if perimeter > 0 else np.nan

    return {
        "area": area,
        "n_components": n_components,
        "solidity": largest.solidity,
        "eccentricity": largest.eccentricity,
        "circularity": circularity,
    }


def compute_edge_slice_metrics(
    h5_path: str | Path,
    segmentation_key: str = "segmentation",
) -> pd.DataFrame:
    """Load a branch H5 file and compute per-slice shape metrics.

    Parameters
    ----------
    h5_path : str or Path
        Path to an H5 file produced by write_slices_to_h5.
    segmentation_key : str
        Dataset key for the segmentation data.

    Returns
    -------
    pd.DataFrame
        One row per slice. Columns: slice_idx, area, n_components, solidity,
        eccentricity, circularity.
    """
    with h5py.File(Path(h5_path), "r") as f:
        segmentation = f[segmentation_key][:]  # (n_slices, H, W)

    records = [
        {"slice_idx": i, **compute_slice_metrics(segmentation[i])}
        for i in range(segmentation.shape[0])
    ]
    return pd.DataFrame(records)


def flag_edge_for_missed_branch(
    metrics_df: pd.DataFrame,
    n_boundary_slices: int = 3,
    solidity_threshold: float = 0.85,
    area_jump_factor: float = 1.5,
    min_interior_slices: int = 3,
) -> tuple[bool, list[str]]:
    """Check whether per-slice metrics indicate a potentially missed daughter branch.

    Only interior slices (excluding the first and last ``n_boundary_slices``) are
    examined, since complexity near the branch ends is expected from parent/daughter
    junctions.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-slice metrics as returned by compute_edge_slice_metrics.
    n_boundary_slices : int
        Slices at each end to skip.
    solidity_threshold : float
        Interior slices with solidity below this are flagged.
        Convex shapes (circles, ellipses) score 1.0; branching-point cross-sections
        typically fall below ~0.85.
    area_jump_factor : float
        Interior slices whose area exceeds this multiple of the median interior area
        are flagged as potentially capturing an additional branch.
    min_interior_slices : int
        Branches with fewer interior slices are skipped (returned as not flagged with
        reason "too_short").

    Returns
    -------
    flagged : bool
    reasons : list[str]
        Human-readable description of each triggered flag.
    """
    n_slices = len(metrics_df)
    interior = metrics_df.iloc[n_boundary_slices : n_slices - n_boundary_slices]

    if len(interior) < min_interior_slices:
        return False, ["too_short"]

    reasons = []

    # --- low solidity ---
    low_sol = interior[interior["solidity"] < solidity_threshold]
    if not low_sol.empty:
        idxs = low_sol["slice_idx"].tolist()
        reasons.append(
            f"low_solidity (slices {idxs}, min={low_sol['solidity'].min():.3f})"
        )

    # --- multiple connected components ---
    multi = interior[interior["n_components"] > 1]
    if not multi.empty:
        idxs = multi["slice_idx"].tolist()
        reasons.append(f"multiple_components (slices {idxs})")

    # --- area jump relative to median interior area ---
    valid_areas = interior["area"].replace(0, np.nan).dropna()
    if not valid_areas.empty:
        median_area = valid_areas.median()
        if median_area > 0:
            large = interior[interior["area"] > area_jump_factor * median_area]
            if not large.empty:
                idxs = large["slice_idx"].tolist()
                reasons.append(
                    f"area_jump (slices {idxs},"
                    f" max={large['area'].max() / median_area:.2f}x median)"
                )

    return len(reasons) > 0, reasons


def analyze_missed_branches_from_slices(
    h5_dir: str | Path,
    file_base: str,
    segmentation_key: str = "segmentation",
    n_boundary_slices: int = 3,
    solidity_threshold: float = 0.85,
    area_jump_factor: float = 1.5,
    min_interior_slices: int = 3,
) -> pd.DataFrame:
    """Scan a directory of branch H5 files and flag edges with potential missed daughters.

    Parameters
    ----------
    h5_dir : str or Path
        Directory containing H5 files produced by write_slices_to_h5.
    file_base : str
        The filename prefix used when the H5 files were written.
    segmentation_key : str
        Dataset key for the segmentation data inside the H5 files.
    n_boundary_slices : int
        Number of slices at each branch end to exclude from the analysis.
    solidity_threshold : float
        Interior slices with solidity below this value are flagged.
    area_jump_factor : float
        Interior slices whose area exceeds this multiple of the median interior area
        are flagged.
    min_interior_slices : int
        Branches with fewer interior slices than this are skipped.

    Returns
    -------
    pd.DataFrame
        One row per edge. Columns:
        - start_node, end_node : graph edge identifiers parsed from the filename
        - n_slices             : total number of cross-sections
        - n_interior_slices   : number of interior slices that were analysed
        - flagged             : True if any anomaly was detected
        - flags               : semicolon-separated list of triggered flag descriptions
    """
    h5_dir = Path(h5_dir)
    pattern = re.compile(rf"{re.escape(file_base)}_sn_(\d+)_en_(\d+)\.h5")

    records = []
    for h5_file in sorted(h5_dir.glob(f"{file_base}_sn_*_en_*.h5")):
        match = pattern.match(h5_file.name)
        if match is None:
            continue
        start_node = int(match.group(1))
        end_node = int(match.group(2))

        metrics_df = compute_edge_slice_metrics(h5_file, segmentation_key=segmentation_key)
        flagged, reasons = flag_edge_for_missed_branch(
            metrics_df,
            n_boundary_slices=n_boundary_slices,
            solidity_threshold=solidity_threshold,
            area_jump_factor=area_jump_factor,
            min_interior_slices=min_interior_slices,
        )

        n_interior = max(0, len(metrics_df) - 2 * n_boundary_slices)
        records.append(
            {
                "start_node": start_node,
                "end_node": end_node,
                "n_slices": len(metrics_df),
                "n_interior_slices": n_interior,
                "flagged": flagged,
                "flags": "; ".join(reasons),
            }
        )

    return pd.DataFrame(records)
