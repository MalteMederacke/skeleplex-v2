"""Batch analysis pipeline: curated graphs -> measurements -> CSV.

For each graph in GRAPHS_FIXED_DIR (falling back to GRAPHS_DIR if absent):
  1. Load graph; skip if origin not set
  2. Prune degree-2 nodes (iterative)
  3. Length-prune short leaf edges
  4. get_all_graph_properties (structural)
  5. sample_volume_slices_from_spline_parallel -> graphs_final/slices/<stem>/
  6. filter_and_segment_lumen(find_lumen=False) -> graphs_final/slices_filt/<stem>/
  7. add_measurements_from_h5_to_graph
  8. get_all_graph_properties (again, now with lumen data)
  9. run_all_angle_metrics
  10. Save final graph -> graphs_final/<stem>_graph_final.json
  11. graph_attributes_to_df + metadata columns -> csvs/<stem>_final.csv

Graphs are built in the segmentation's coordinate space, so slices are sampled
from the matching image / segmentation arrays (the cropped variants when
present) to stay aligned.

Skip samples whose final graph JSON already exists (unless OVERWRITE).
"""

import argparse
import gc
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from skeleplex.graph.modify_graph import length_pruning, prune_degree_2_nodes
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.utils import write_slices_to_h5
from skeleplex.measurements.angles import run_all_angle_metrics
from skeleplex.measurements.branches import (
    add_measurements_from_h5_to_graph,
    filter_and_segment_lumen,
)
from skeleplex.measurements.graph_properties import get_all_graph_properties
from skeleplex.measurements.utils import graph_attributes_to_df

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    APPROX,
    CIRCULARITY_THRESHOLD,
    CSVS_DIR,
    ECCENTRICITY_THRESHOLD,
    GRAPHS_DIR,
    GRAPHS_FINAL_DIR,
    GRAPHS_FIXED_DIR,
    LENGTH_PRUNE_THRESHOLD,
    NUM_WORKERS,
    PROJECT_ROOT,
    SAMPLE_GRID_SPACING_UM,
    SAMPLE_POSITIONS,
    SLICE_SIZE_UM,
    SLICE_SPACING,
    ZARR_DIR,
)


# ─── Metadata parsing ────────────────────────────────────────────────────────

def sample_metadata(stem: str) -> dict:
    """Metadata columns to attach to every row of a sample's CSV.

    Override this to parse experiment metadata (genotype, condition, timepoint,
    ...) out of the sample name. The default just records the sample name.

    Example — parse a Wnt11-KO genotype axis::

        s = stem.lower()
        genotype = (
            "het" if "het" in s else
            "MUT" if "mut" in s else
            "WT" if "wt" in s else "unknown"
        )
        return {"sample": stem, "genotype": genotype}
    """
    return {"sample": stem}  # ADAPT HERE


# ─── Graph discovery ─────────────────────────────────────────────────────────

def find_graphs() -> list[Path]:
    """Collect graphs from GRAPHS_DIR, preferring the curated GRAPHS_FIXED_DIR copy."""
    found = []
    for gp in sorted(GRAPHS_DIR.glob("*_graph.json")):
        fixed = GRAPHS_FIXED_DIR / gp.name
        found.append(fixed if fixed.exists() else gp)
    return found


# ─── Core processing ─────────────────────────────────────────────────────────

def process_graph(graph_path: Path, overwrite: bool) -> None:
    stem = graph_path.stem[: -len("_graph")]  # strip trailing _graph
    metadata = sample_metadata(stem)

    print(f"  sample    : {stem}")
    print(f"  metadata  : {metadata}")

    # Output paths
    GRAPHS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    pruned_graph_path = GRAPHS_FINAL_DIR / f"{stem}_graph_pruned.json"
    final_graph_path = GRAPHS_FINAL_DIR / f"{stem}_graph_final.json"

    slices_dir = GRAPHS_FINAL_DIR / "slices" / stem
    slices_filt_dir = GRAPHS_FINAL_DIR / "slices_filt" / stem

    CSVS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSVS_DIR / f"{stem}_final.csv"

    if overwrite:
        for d in (slices_dir, slices_filt_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"  [overwrite] removed {d.relative_to(PROJECT_ROOT)}")

    if final_graph_path.exists() and not overwrite:
        print(f"  [skip] final graph exists: {final_graph_path.name}")
        return

    # ── 1. Load ──────────────────────────────────────────────────────────────
    graph = SkeletonGraph.from_json_file(str(graph_path))
    origin = getattr(graph, "origin", None)
    if origin is None:
        print("  [skip] origin not set — inspect in review_graphs.py first")
        return

    voxel_size_um = graph.voxel_size_um
    prefix = f"{stem}_"
    print(f"  origin    : {origin}")
    print(f"  voxel_size: {voxel_size_um}")

    # ── 2. Prune degree-2 nodes ───────────────────────────────────────────────
    print("  pruning degree-2 nodes ...")
    num_nodes_after = graph.graph.number_of_nodes()
    c = 0
    while True:
        num_nodes = num_nodes_after
        prune_degree_2_nodes(graph, do_all=False)
        num_nodes_after = graph.graph.number_of_nodes()
        c += 1
        print(f"    iteration {c}: {num_nodes_after} nodes")
        if num_nodes_after >= num_nodes:
            break

    # ── 2b. Length-prune short leaf edges ─────────────────────────────────────
    print(f"  length pruning leaf edges < {LENGTH_PRUNE_THRESHOLD} ...")
    num_nodes_after = graph.graph.number_of_nodes()
    c = 0
    while True:
        num_nodes = num_nodes_after
        length_pruning(graph, length_threshold=LENGTH_PRUNE_THRESHOLD)
        num_nodes_after = graph.graph.number_of_nodes()
        c += 1
        print(f"    iteration {c}: {num_nodes_after} nodes")
        if num_nodes_after >= num_nodes:
            break

    # ── 3. Graph properties (structural) ─────────────────────────────────────
    print("  get_all_graph_properties (structural) ...")
    graph.graph, _ = get_all_graph_properties(
        graph.graph,
        prefix=prefix,
        voxel_size_um=voxel_size_um,
        origin=origin,
        approx=APPROX,
    )

    # Save pruned graph so add_measurements_from_h5_to_graph can load it
    graph.to_json_file(str(pruned_graph_path))

    # ── 4. Find zarr for image + segmentation (cropped, to match graph) ───────
    zarr_path = ZARR_DIR / f"{stem}.zarr"
    if not zarr_path.exists():
        print(f"  ERROR: no zarr found for {stem}")
        return
    img_key = "image_cropped" if (zarr_path / "image_cropped").exists() else "image"
    seg_key = (
        "segmentation_cropped"
        if (zarr_path / "segmentation_cropped").exists()
        else "segmentation"
    )
    volume_path = str(zarr_path / img_key)
    segmentation_path = str(zarr_path / seg_key)

    # ── 5. Sample slices ─────────────────────────────────────────────────────
    if slices_dir.exists() and any(slices_dir.iterdir()):
        print(f"  slices already sampled, skipping ({slices_dir})")
    else:
        slices_dir.mkdir(parents=True, exist_ok=True)
        print(f"  sampling slices → {slices_dir}")
        slice_dict, seg_slice_dict = graph.sample_volume_slices_from_spline_parallel(
            volume_path=volume_path,
            segmentation_path=segmentation_path,
            slice_spacing=SLICE_SPACING,
            sample_grid_spacing_um=SAMPLE_GRID_SPACING_UM,
            slice_size_um=SLICE_SIZE_UM,
            num_workers=NUM_WORKERS,
            approx=APPROX,
        )
        write_slices_to_h5(
            str(slices_dir) + "/",
            stem,
            slice_dict,
            seg_slice_dict,
            sample_grid_spacing_um=SAMPLE_GRID_SPACING_UM,
        )
        del slice_dict, seg_slice_dict
        gc.collect()

    # ── 6. Filter slices ──────────────────────────────────────────────────────
    if slices_filt_dir.exists() and any(slices_filt_dir.iterdir()):
        print(f"  filtered slices already exist, skipping ({slices_filt_dir})")
    else:
        slices_filt_dir.mkdir(parents=True, exist_ok=True)
        print(f"  filter_and_segment_lumen → {slices_filt_dir}")
        filter_and_segment_lumen(
            data_path=str(slices_dir),
            save_path=str(slices_filt_dir),
            find_lumen=False,
            eccentricity_thresh=ECCENTRICITY_THRESHOLD,
            circularity_thresh=CIRCULARITY_THRESHOLD,
        )

    # ── 7. Add lumen measurements to graph ───────────────────────────────────
    print("  add_measurements_from_h5_to_graph ...")
    graph = add_measurements_from_h5_to_graph(
        graph_path=str(pruned_graph_path),
        input_path=str(slices_filt_dir),
    )

    # ── 8. Graph properties (with lumen data) ────────────────────────────────
    print("  get_all_graph_properties (with lumen) ...")
    graph.graph, _ = get_all_graph_properties(
        graph.graph,
        prefix=prefix,
        voxel_size_um=voxel_size_um,
        origin=origin,
        approx=APPROX,
    )

    # ── 9. Angle metrics ─────────────────────────────────────────────────────
    print("  run_all_angle_metrics ...")
    graph.graph, _ = run_all_angle_metrics(
        graph.graph,
        origin=origin,
        approx=APPROX,
        sample_positions=np.asarray(SAMPLE_POSITIONS),
    )

    # ── 10. Save final graph ──────────────────────────────────────────────────
    graph.to_json_file(str(final_graph_path))
    print(f"  saved graph → {final_graph_path.relative_to(PROJECT_ROOT)}")

    # ── 11. CSV ───────────────────────────────────────────────────────────────
    df = graph_attributes_to_df(graph.graph)
    for col, value in metadata.items():
        df[col] = value
    df.to_csv(str(csv_path), index=False)
    print(f"  saved CSV  → {csv_path.relative_to(PROJECT_ROOT)}")


def main(overwrite: bool = False) -> None:
    """Measure every curated graph and write per-sample + combined CSVs.

    Parameters
    ----------
    overwrite : bool
        Re-process samples whose final graph already exists, deleting their
        cached ``slices/`` and ``slices_filt/`` directories first.
    """
    graph_paths = find_graphs()
    if not graph_paths:
        print(f"No graphs found in {GRAPHS_DIR}. Build skeleton graphs first.")
        sys.exit(1)

    print(f"Found {len(graph_paths)} graphs.\n")

    errors = []
    for graph_path in graph_paths:
        print(graph_path.name)
        try:
            process_graph(graph_path, overwrite=overwrite)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((graph_path, e))
        print()

    # Merge all per-sample CSVs into one combined file
    csv_files = sorted(CSVS_DIR.glob("*_final.csv"))
    if csv_files:
        combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        combined_path = CSVS_DIR / "all_samples_combined.csv"
        combined.to_csv(str(combined_path), index=False)
        print(
            f"Combined CSV ({len(combined)} rows) → "
            f"{combined_path.relative_to(PROJECT_ROOT)}"
        )

    n = len(graph_paths)
    print(f"\nDone. {n - len(errors)}/{n} processed.")
    if errors:
        print("Errors:")
        for p, e in errors:
            print(f"  {p.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process samples whose final graph already exists.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite)
