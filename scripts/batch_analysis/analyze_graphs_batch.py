"""Batch analysis pipeline: curated graphs -> measurements -> CSV.

For each graph found by the layout module (the curated copy in graphs_fixed/
when there is one, else the raw graph):
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

A sample segmented on several channels is measured ONCE, on the best channel
available; anything curated out is skipped (see ``exclusions.py``).

Skips samples whose final graph JSON already exists (unless --overwrite).
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
    ECCENTRICITY_THRESHOLD,
    LENGTH_PRUNE_THRESHOLD,
    NUM_WORKERS,
    SAMPLE_GRID_SPACING_UM,
    SAMPLE_METADATA,
    SAMPLE_POSITIONS,
    SLICE_SIZE_UM,
    SLICE_SPACING,
)
from _layout import (  # noqa: E402
    csvs_dir,
    find_graphs,
    graphs_final_dir,
    image_key_in,
    rel,
)


# ─── Metadata parsing ────────────────────────────────────────────────────────

def sample_metadata(stem: str, sample) -> dict:
    """Metadata columns to attach to every row of a sample's CSV.

    Set SAMPLE_METADATA in your config to a function of (stem, sample) to parse
    experiment metadata out of the sample name or its directory. The default
    just records the sample name.

    ``sample`` is the :class:`_layout.Sample`, so ``sample.sample_dir.name`` is
    the condition directory in the nested layout and ``sample.channel`` is the
    channel the segmentation came from.

    Example — a treatment screen whose condition directories are named
    ``Ctrl``, ``LatA_1uM``, ``CNF100ng``::

        import re

        def SAMPLE_METADATA(stem, sample):
            condition = sample.sample_dir.name
            if "_" in condition:
                treatment, concentration = condition.split("_", 1)
            else:
                m = re.match(r"([A-Za-z]+)(\\d+.*)?", condition)
                treatment = m.group(1) if m else condition
                concentration = (m.group(2) or "") if m else ""
            m = re.search(r"sample(\\d+)", stem)
            return {
                "sample": stem,
                "condition": condition,
                "treatment": treatment,
                "concentration": concentration,
                "sample_number": int(m.group(1)) if m else 0,
            }
    """
    if SAMPLE_METADATA is not None:
        return SAMPLE_METADATA(stem, sample)
    return {"sample": stem}


def output_keys(refs):
    """Map each graph ref to a filename-safe key, unique across the whole run.

    Stems are normally unique acquisition names, but in the nested layout
    nothing stops two conditions from holding the same stem — and the CSV
    directory is shared, so equal stems would silently overwrite each other's
    table and drop a sample from the combined CSV. Only the colliding stems get
    qualified with their condition, so ordinary runs keep their plain names.
    """
    counts = {}
    for ref in refs:
        counts[ref.sample.stem] = counts.get(ref.sample.stem, 0) + 1

    keys = {}
    for ref in refs:
        stem = ref.sample.stem
        if counts[stem] > 1:
            keys[ref.path] = f"{ref.sample.sample_dir.name}_{stem}"
        else:
            keys[ref.path] = stem
    return keys


# ─── Core processing ─────────────────────────────────────────────────────────

def process_graph(ref, key: str, overwrite: bool) -> None:
    """Measure one graph and write its final graph JSON and CSV."""
    graph_path = ref.path
    sample = ref.sample
    stem = sample.stem
    metadata = sample_metadata(stem, sample)

    print(f"  sample    : {stem}")
    print(f"  channel   : {sample.channel}")
    print(f"  metadata  : {metadata}")

    # Output paths
    final_dir = graphs_final_dir(sample)
    final_dir.mkdir(parents=True, exist_ok=True)
    pruned_graph_path = final_dir / f"{stem}_graph_pruned.json"
    final_graph_path = final_dir / f"{stem}_graph_final.json"

    slices_dir = final_dir / "slices" / stem
    slices_filt_dir = final_dir / "slices_filt" / stem

    out_csv_dir = csvs_dir()
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_csv_dir / f"{key}_final.csv"

    if overwrite:
        for d in (slices_dir, slices_filt_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"  [overwrite] removed {rel(d)}")

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
    prefix = f"{key}_"
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

    # ── 4. Image + segmentation arrays (cropped, to match the graph) ─────────
    zarr_path = sample.zarr_path
    if not zarr_path.exists():
        print(f"  ERROR: no zarr found for {stem} at {zarr_path}")
        return

    import zarr as _zarr
    store = _zarr.open(str(zarr_path), mode="r")
    img_key = (
        "image_cropped" if "image_cropped" in store
        else image_key_in(store, sample.channel)
    )
    seg_key = (
        "segmentation_cropped" if "segmentation_cropped" in store
        else "segmentation"
    )
    volume_path = str(zarr_path / img_key)
    segmentation_path = str(zarr_path / seg_key)
    print(f"  arrays    : {img_key} / {seg_key}")

    # ── 5. Sample slices ─────────────────────────────────────────────────────
    if slices_dir.exists() and any(slices_dir.iterdir()):
        print(f"  slices already sampled, skipping ({rel(slices_dir)})")
    else:
        slices_dir.mkdir(parents=True, exist_ok=True)
        print(f"  sampling slices → {rel(slices_dir)}")
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
        print(f"  filtered slices already exist, skipping ({rel(slices_filt_dir)})")
    else:
        slices_filt_dir.mkdir(parents=True, exist_ok=True)
        print(f"  filter_and_segment_lumen → {rel(slices_filt_dir)}")
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
    print(f"  saved graph → {rel(final_graph_path)}")

    # ── 11. CSV ───────────────────────────────────────────────────────────────
    df = graph_attributes_to_df(graph.graph)
    for col, value in metadata.items():
        df[col] = value
    df.to_csv(str(csv_path), index=False)
    print(f"  saved CSV  → {rel(csv_path)}")


def main(overwrite: bool = False, channels=None) -> None:
    """Measure every curated graph and write per-sample + combined CSVs.

    Parameters
    ----------
    overwrite : bool
        Re-process samples whose final graph already exists, deleting their
        cached ``slices/`` and ``slices_filt/`` directories first.
    channels : sequence of str, optional
        Channel preference order. Defaults to CHANNEL_PRIORITY.
    """
    refs = find_graphs(channels=channels)
    if not refs:
        print("No graphs found. Build skeleton graphs first.")
        sys.exit(1)

    keys = output_keys(refs)
    n_qualified = sum(1 for r in refs if keys[r.path] != r.sample.stem)
    n_curated = sum(1 for r in refs if r.curated)
    print(f"Found {len(refs)} graphs ({n_curated} curated).")
    if n_qualified:
        print(
            f"{n_qualified} share a stem with another sample and were "
            "qualified with their condition name in the CSV output."
        )
    print()

    errors = []
    for ref in refs:
        print(rel(ref.path))
        try:
            process_graph(ref, keys[ref.path], overwrite=overwrite)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((ref, e))
        print()

    # Merge all per-sample CSVs into one combined file
    out_csv_dir = csvs_dir()
    csv_files = sorted(out_csv_dir.glob("*_final.csv"))
    if csv_files:
        combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        combined_path = out_csv_dir / "all_samples_combined.csv"
        combined.to_csv(str(combined_path), index=False)
        print(f"Combined CSV ({len(combined)} rows) → {rel(combined_path)}")

    n = len(refs)
    print(f"\nDone. {n - len(errors)}/{n} processed.")
    if errors:
        print("Errors:")
        for ref, e in errors:
            print(f"  {rel(ref.path)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process samples whose final graph already exists.",
    )
    parser.add_argument(
        "--channel", action="append", dest="channels",
        help="Channel preference order (repeatable). Defaults to CHANNEL_PRIORITY.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite, channels=args.channels)
