"""Batch script: segmentation zarr -> skeleton prediction -> skeleton graph.

For each segmented sample found by ``_layout.find_segmented``:
  1. Load the (cropped) segmentation, fill holes
  2. Compute the inward unit normal field
  3. Run the normal-field skeletonize model
  4. Binarize and save the normal-field prediction + binary skeleton to the zarr
  5. prune_and_fix_skeleton + image_to_graph_skan -> SkeletonGraph
  6. Save the graph JSON next to the sample (see ``_layout.graph_path``)

The skeleton is built on the CROPPED segmentation when present (much smaller
than the full volume, which is mostly background). Graph coordinates are then in
cropped-volume space; branching angles are translation-invariant so this is fine
for downstream analysis. Skeleton arrays are stored with a matching '_cropped'
suffix so they stay aligned with segmentation_cropped.

A sample segmented on several channels is processed ONCE, on the best channel
available (CHANNEL_PRIORITY) — without that, a sample with both an ssh and a
dapi segmentation would contribute two graphs of the same acquisition. Samples
under a curated-out directory are skipped entirely (see ``exclusions.py``).

Skips samples whose graph JSON already exists. Manual steps (origin, directing,
curation) stay in review_graphs.py.
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize as ski_skeletonize

from skeleplex.graph.break_detection import prune_and_fix_skeleton
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.skeleton._skeletonize import load_normal_field_model, skeletonize
from skeleplex.skeleton.distance_field import (
    inward_unit_normal_field_cpu,
    inward_unit_normal_field_gpu,
)

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    BRANCH_TRIMMING_LEN,
    BREAK_DISTANCE,
    CHANNELS,
    DEFAULT_VOXEL_SIZE_UM,
    NORMAL_FIELD_BACKEND,
    SKELETON_CHECKPOINT,
    SKELETON_THRESHOLD,
    SKELETONIZE_KWARGS,
)
from _layout import find_segmented, graph_path, rel  # noqa: E402

# prefer the cropped segmentation (smaller -> faster EDT/skeletonize)
PREFER_CROPPED = True


def _chunks(shape, base=(64, 256, 256)):
    return tuple(min(b, s) for b, s in zip(base, shape))


def seg_key_for(store):
    """Segmentation array to skeletonize, and the suffix for output arrays."""
    if PREFER_CROPPED and "segmentation_cropped" in store:
        return "segmentation_cropped", "_cropped"
    return "segmentation", ""


def compute_normal_field(segmentation, backend):
    # EDT is a global operation — must run on the full volume, cannot be chunked.
    if backend == "gpu":
        return inward_unit_normal_field_gpu(segmentation)
    return inward_unit_normal_field_cpu(segmentation)


def process_sample(sample, model, backend):
    """Build and save the skeleton graph for one segmented sample."""
    out_path = graph_path(sample)

    if out_path.exists():
        print(f"  [skip] graph already exists: {out_path.name}")
        return

    store = zarr.open(str(sample.zarr_path), mode="r+")
    voxel_size_um = list(store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM)))

    seg_key, suffix = seg_key_for(store)
    if seg_key not in store:
        print(f"  [skip] no '{seg_key}' array")
        return
    segmentation = np.asarray(store[seg_key][:], dtype=bool)

    print("  fill holes ...")
    segmentation = binary_fill_holes(segmentation)

    print(f"  normal field on shape {segmentation.shape} ...")
    normal_field = compute_normal_field(segmentation, backend)

    print("  skeletonize ...")
    skeleton_prediction = skeletonize(normal_field, model=model, **SKELETONIZE_KWARGS)
    skeleton_prediction = skeleton_prediction * segmentation
    skeleton_bin = (skeleton_prediction > SKELETON_THRESHOLD) * segmentation

    # Save prediction and binary skeleton into the same zarr (matching suffix)
    for key, arr, dtype in [
        (f"normal_field_prediction{suffix}", skeleton_prediction, np.float32),
        (f"skeleton{suffix}", skeleton_bin.astype(np.uint8), np.uint8),
    ]:
        if key in store:
            del store[key]
        store.create_array(key, data=arr.astype(dtype), chunks=_chunks(arr.shape))

    del normal_field, skeleton_prediction
    gc.collect()

    print("  prune and fix skeleton ...")
    skeleton_morph = ski_skeletonize(skeleton_bin)
    skeleton_morph = prune_and_fix_skeleton(
        skeleton_morph,
        segmentation=segmentation,
        break_distance=BREAK_DISTANCE,
        branch_trimming_len=BRANCH_TRIMMING_LEN,
    )

    print("  build graph ...")
    nx_graph = image_to_graph_skan(skeleton_morph, image_voxel_size_um=voxel_size_um)
    graph = SkeletonGraph(nx_graph, voxel_size_um=voxel_size_um)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_json_file(str(out_path))
    print(f"  saved graph -> {rel(out_path)}  (from '{seg_key}')")


def main(
    checkpoint: Path = None,
    backend: str = NORMAL_FIELD_BACKEND,
    channels=None,
) -> None:
    """Build a skeleton graph for every segmented sample.

    Parameters
    ----------
    checkpoint : Path, optional
        Normal-field skeleton-prediction model checkpoint (``.ckpt``).
        Defaults to SKELETON_CHECKPOINT.
    backend : str
        ``"cpu"`` or ``"gpu"`` for the inward unit normal field computation.
    channels : sequence of str, optional
        Channel preference order. Defaults to CHANNEL_PRIORITY.
    """
    checkpoint = Path(checkpoint) if checkpoint else Path(SKELETON_CHECKPOINT)

    samples = find_segmented(channels=channels)
    if not samples:
        print("No segmentation zarrs found. Run segment_batch.py first.")
        sys.exit(1)

    todo = [s for s in samples if not graph_path(s).exists()]
    print(
        f"Found {len(samples)} segmented samples, {len(todo)} without a graph.\n"
    )
    if not todo:
        print("Nothing to do.")
        return

    print("Loading normal field model ...")
    model = load_normal_field_model(str(checkpoint))
    print("Model loaded.\n")

    errors = []
    for i, sample in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {rel(sample.zarr_path)}  ({sample.channel})")
        try:
            process_sample(sample, model, backend)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((sample, e))

    print(f"\nDone. {len(todo) - len(errors)}/{len(todo)} processed.")
    if errors:
        print("Errors:")
        for s, e in errors:
            print(f"  {rel(s.zarr_path)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SKELETON_CHECKPOINT)
    parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default=NORMAL_FIELD_BACKEND,
        help="Backend for the inward unit normal field computation.",
    )
    parser.add_argument(
        "--channel", action="append", dest="channels", choices=list(CHANNELS),
        help="Channel preference order (repeatable). Defaults to CHANNEL_PRIORITY.",
    )
    args = parser.parse_args()

    main(checkpoint=args.checkpoint, backend=args.backend, channels=args.channels)
