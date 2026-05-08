#!/usr/bin/env python3
"""Convert a skeleplex graph to a skeleton zarr and run fusion break repair.

Rasterizes all graph edges (world-space µm coordinates) back into a binary
skeleton image in voxel space, saves it to zarr, then runs
repair_fusion_breaks_lazy on the result.

Example
-------
python scripts/graph_to_skeleton_repair_fusion.py \
    --graph path/to/graph.json \
    --segmentation path/to/seg.zarr \
    --scale-map path/to/scale_map.zarr \
    --skeleton-out path/to/skeleton.zarr \
    --repaired-out path/to/repaired.zarr \
    --repair-radius 10 \
    --chunk-shape 256 256 256 \
    --endpoint-mask-dilation 2
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from skeleplex.graph.constants import EDGE_COORDINATES_KEY
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.utils import draw_line_segment
from skeleplex.skeleton._break_detection_lazy import repair_fusion_breaks_lazy


def rasterize_graph(
    skeleton_graph: SkeletonGraph,
    shape: tuple[int, int, int],
    voxel_size: np.ndarray,
) -> np.ndarray:
    """Rasterize all graph edges into a binary skeleton image.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        Graph with edge coordinates in world space (µm).
    shape : tuple[int, int, int]
        Output image shape (z, y, x) in voxels.
    voxel_size : np.ndarray
        Voxel size (z, y, x) in µm/voxel.

    Returns
    -------
    np.ndarray
        uint8 binary skeleton image.
    """
    image = np.zeros(shape, dtype=np.uint8)
    graph = skeleton_graph.graph
    bounds = np.array(shape) - 1

    if graph.is_multigraph():
        edge_attrs = [a for _, _, a in graph.edges(data=True, keys=False)]
    else:
        edge_attrs = [a for _, _, a in graph.edges(data=True)]

    for attrs in tqdm(edge_attrs, desc="Rasterizing edges"):
        pts = attrs.get(EDGE_COORDINATES_KEY)
        if pts is None:
            continue
        vx = np.clip(
            np.round(np.asarray(pts, dtype=np.float64) / voxel_size).astype(np.int64),
            0,
            bounds,
        )
        for i in range(len(vx) - 1):
            draw_line_segment(vx[i].astype(float), vx[i + 1].astype(float), image)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--graph", required=True, type=Path,
        help="Path to the skeleplex graph JSON file."
    )
    parser.add_argument(
        "--segmentation", required=True, type=Path,
        help="Path to the segmentation zarr (sets the image shape)."
    )
    parser.add_argument(
        "--scale-map", required=True, type=Path,
        help="Path to the prediction tile ID map zarr (fusion boundary map)."
    )
    parser.add_argument(
        "--skeleton-out", required=True, type=Path,
        help="Where to write the rasterized skeleton zarr."
    )
    parser.add_argument(
        "--repaired-out", required=True, type=Path,
        help="Where to write the repaired skeleton zarr."
    )
    parser.add_argument(
        "--repair-radius", type=float, default=10.0,
        help="Break repair radius in voxels (default: 10)."
    )
    parser.add_argument(
        "--chunk-shape", type=int, nargs=3, default=[256, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Processing chunk shape in voxels (default: 256 256 256)."
    )
    parser.add_argument(
        "--zarr-chunk-shape", type=int, nargs=3, default=None,
        metavar=("Z", "Y", "X"),
        help="Zarr storage chunk shape. Defaults to --chunk-shape."
    )
    parser.add_argument(
        "--label-map", type=Path, default=None,
        help="Optional pre-computed global connected-component label map zarr."
    )
    parser.add_argument(
        "--endpoint-mask-dilation", type=int, default=0,
        help="Binary dilation iterations on the fusion boundary mask (default: 0)."
    )
    parser.add_argument(
        "--backend", choices=["cpu", "cupy"], default="cpu",
        help="Compute backend (default: cpu)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load graph and voxel size                                         #
    # ------------------------------------------------------------------ #
    print(f"Loading graph from {args.graph} …")
    sg = SkeletonGraph.from_json_file(args.graph)

    if sg.voxel_size_um is None:
        raise ValueError(
            "Graph has no voxel_size_um attribute. "
            "Cannot determine voxel-space coordinates."
        )
    voxel_size = np.array(sg.voxel_size_um, dtype=np.float64)
    print(f"  voxel size (z, y, x): {voxel_size} µm")
    print(f"  edges in graph: {sg.graph.number_of_edges()}")

    # ------------------------------------------------------------------ #
    # 2. Open segmentation to get the target image shape                  #
    # ------------------------------------------------------------------ #
    print(f"Opening segmentation at {args.segmentation} …")
    seg_zarr = zarr.open(str(args.segmentation), mode="r")
    shape = seg_zarr.shape
    print(f"  image shape (z, y, x): {shape}")

    # ------------------------------------------------------------------ #
    # 3. Rasterize graph into a binary skeleton image                     #
    # ------------------------------------------------------------------ #
    print("Rasterizing graph edges into skeleton image …")
    skeleton_image = rasterize_graph(sg, shape, voxel_size)
    print(f"  non-zero voxels: {int(skeleton_image.sum())}")

    # ------------------------------------------------------------------ #
    # 4. Write skeleton zarr                                               #
    # ------------------------------------------------------------------ #
    zarr_chunks = tuple(args.zarr_chunk_shape or args.chunk_shape)
    print(f"Writing skeleton to {args.skeleton_out} …")
    skel_zarr = zarr.open(
        str(args.skeleton_out),
        mode="w",
        shape=shape,
        chunks=zarr_chunks,
        dtype=np.uint8,
    )
    skel_zarr[:] = skeleton_image
    del skeleton_image  # free RAM before chunk-based repair
    print("  skeleton saved.")

    # ------------------------------------------------------------------ #
    # 5. Run fusion break repair                                           #
    # ------------------------------------------------------------------ #
    print("Running repair_fusion_breaks_lazy …")
    repair_fusion_breaks_lazy(
        skeleton_path=args.skeleton_out,
        segmentation_path=args.segmentation,
        scale_map_path=args.scale_map,
        output_path=args.repaired_out,
        repair_radius=args.repair_radius,
        chunk_shape=tuple(args.chunk_shape),
        label_map_path=args.label_map,
        endpoint_mask_dilation=args.endpoint_mask_dilation,
        backend=args.backend,
    )
    print(f"Done. Repaired skeleton at {args.repaired_out}")


if __name__ == "__main__":
    main()
