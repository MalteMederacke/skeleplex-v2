"""Batch script: segmentation zarr -> skeleton prediction -> skeleton graph.

For each <ZARR_DIR>/<name>.zarr group written by segment_batch.py:
  1. Load the (cropped) segmentation, fill holes
  2. Compute the inward unit normal field
  3. Run the normal-field skeletonize model
  4. Binarize and save the normal-field prediction + binary skeleton to the zarr
  5. prune_and_fix_skeleton + image_to_graph_skan -> SkeletonGraph
  6. Save the graph JSON to <GRAPHS_DIR>/<name>_graph.json

The skeleton is built on the CROPPED segmentation when present (much smaller
than the full volume, which is mostly background). Graph coordinates are then in
cropped-volume space; branching angles are translation-invariant so this is fine
for downstream analysis. Skeleton arrays are stored with a matching '_cropped'
suffix so they stay aligned with segmentation_cropped.

Skips samples where the graph JSON already exists. Manual steps (origin,
directing, curation) stay in review_graphs.py.
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
    DEFAULT_VOXEL_SIZE_UM,
    GRAPHS_DIR,
    NORMAL_FIELD_BACKEND,
    SKELETON_CHECKPOINT,
    SKELETON_THRESHOLD,
    SKELETONIZE_KWARGS,
    ZARR_DIR,
)

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


def process_zarr(zarr_path, model, backend):
    stem = zarr_path.stem
    graph_path = GRAPHS_DIR / f"{stem}_graph.json"

    if graph_path.exists():
        print(f"  [skip] graph already exists: {graph_path.name}")
        return

    store = zarr.open(str(zarr_path), mode="r+")
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

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    graph.to_json_file(str(graph_path))
    print(f"  saved graph -> {graph_path.name}  (from '{seg_key}')")


def main(
    root: Path,
    checkpoint: Path,
    backend: str = NORMAL_FIELD_BACKEND,
) -> None:
    """Build a skeleton graph for every segmented zarr in ``root``.

    Parameters
    ----------
    root : Path
        Directory of ``<name>.zarr`` groups holding a segmentation array.
    checkpoint : Path
        Normal-field skeleton-prediction model checkpoint (``.ckpt``).
    backend : str
        ``"cpu"`` or ``"gpu"`` for the inward unit normal field computation.
    """
    zarr_dirs = sorted(p for p in root.glob("*.zarr") if p.is_dir())
    if not zarr_dirs:
        print(f"No *.zarr containers found in {root}. Run segment_batch.py first.")
        sys.exit(1)

    print(f"Found {len(zarr_dirs)} zarr segmentations.\n")
    print("Loading normal field model ...")
    model = load_normal_field_model(str(checkpoint))
    print("Model loaded.\n")

    errors = []
    for zarr_path in zarr_dirs:
        print(zarr_path.name)
        try:
            process_zarr(zarr_path, model, backend)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((zarr_path, e))

    n = len(zarr_dirs)
    print(f"\nDone. {n - len(errors)}/{n} processed.")
    if errors:
        print("Errors:")
        for p, e in errors:
            print(f"  {p.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ZARR_DIR)
    parser.add_argument("--checkpoint", type=Path, default=SKELETON_CHECKPOINT)
    parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default=NORMAL_FIELD_BACKEND,
        help="Backend for the inward unit normal field computation.",
    )
    args = parser.parse_args()

    main(root=args.root, checkpoint=args.checkpoint, backend=args.backend)
