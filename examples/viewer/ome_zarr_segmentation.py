"""Example of viewing a segmentation as a multi-resolution OME-Zarr.

A plain zarr array has one resolution, so the viewer has to load all of it into
memory. An OME-Zarr pyramid also stores coarser copies, and the viewer streams
only the level and the region that the current view needs.

This script makes its own small dataset, so you can run it as it is:

1. Write a skeleton graph and a segmentation as a plain zarr array.
2. Convert the segmentation to an OME-Zarr pyramid.
3. Read the pyramid from python.
4. Open it in the viewer.

To use your own data, skip step 1 and pass your paths to steps 2 and 4.
Requires the viewer extras: pip install "skeleplex-v2[cellier,viz]"
"""

import tempfile
from pathlib import Path

import dask.array as da
import networkx as nx
import numpy as np
import zarr

import skeleplex
from skeleplex.app import view_skeleton
from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.spline import B3Spline
from skeleplex.utils import image_info, open_image, write_ome_zarr_labels

VOXEL_SIZE_UM = (10.0, 10.0, 10.0)  # (z, y, x)
SEGMENTATION_SHAPE = (256, 256, 256)
TUBE_RADIUS_UM = 150.0

# a Y-shaped skeleton, with (z, y, x) coordinates in micrometers
NODES = {
    0: np.array([300.0, 100.0, 300.0]),
    1: np.array([1300.0, 1300.0, 1300.0]),
    2: np.array([2300.0, 2300.0, 500.0]),
    3: np.array([500.0, 2300.0, 2300.0]),
}
EDGES = [(0, 1), (1, 2), (1, 3)]


def write_skeleton_graph(path: Path) -> None:
    """Write the Y-shaped skeleton as a graph file."""
    graph = nx.DiGraph()
    for node, coordinate in NODES.items():
        graph.add_node(node, **{NODE_COORDINATE_KEY: coordinate})
    for start, end in EDGES:
        coordinates = np.linspace(NODES[start], NODES[end], 6)
        graph.add_edge(
            start,
            end,
            **{
                EDGE_COORDINATES_KEY: coordinates,
                EDGE_SPLINE_KEY: B3Spline.from_points(coordinates),
            },
        )
    SkeletonGraph(graph=graph).to_json_file(str(path))


def write_segmentation(path: Path) -> None:
    """Write a plain zarr array with one tube-shaped label around each edge.

    The dtype is uint64 like the output of skeleplex.skeleton.label_chunks_parallel.
    """
    segmentation = zarr.open(
        str(path),
        mode="w",
        shape=SEGMENTATION_SHAPE,
        chunks=(64, 64, 64),
        dtype=np.uint64,
    )
    voxel_size = np.array(VOXEL_SIZE_UM)
    y, x = np.indices(SEGMENTATION_SHAPE[1:])
    for z in range(SEGMENTATION_SHAPE[0]):
        position = np.stack([np.full_like(y, z), y, x], axis=-1) * voxel_size
        plane = np.zeros(SEGMENTATION_SHAPE[1:], dtype=np.uint64)
        for label, (start, end) in enumerate(EDGES, start=1):
            direction = NODES[end] - NODES[start]
            along_edge = (position - NODES[start]) @ direction / (direction @ direction)
            closest = NODES[start] + np.clip(along_edge, 0, 1)[..., None] * direction
            distance = np.linalg.norm(position - closest, axis=-1)
            plane[distance <= TUBE_RADIUS_UM] = label
        segmentation[z] = plane


# the example data are deleted when the script ends
temporary_directory = tempfile.TemporaryDirectory(prefix="skeleplex_ome_zarr_")
data_directory = Path(temporary_directory.name)
graph_path = data_directory / "skeleton_graph.json"
plain_path = data_directory / "segmentation.zarr"
pyramid_path = data_directory / "segmentation.ome.zarr"

# 1. Make the example data. This is what the pipeline gives you.
print(f"Writing the example data to {data_directory}")
write_skeleton_graph(graph_path)
write_segmentation(plain_path)

# 2. Convert the segmentation to an OME-Zarr pyramid. This does not change the
# original and it refuses to overwrite an existing path. By default levels are
# added until the largest axis is at most 128 voxels. Pass n_levels to choose.
write_ome_zarr_labels(plain_path, pyramid_path, voxel_size=VOXEL_SIZE_UM)

# 3. Read it from python. These two functions accept plain zarr and OME-Zarr, so
# code using them does not need to know which one it gets.
for path in (plain_path, pyramid_path):
    info = image_info(path)
    # a plain zarr array has no metadata, so its voxel size is None
    print(f"{path.name}: {info.n_levels} level(s), voxel size: {info.voxel_size}")

# open_image returns one zarr array: the finest level unless you ask for another.
finest = open_image(pyramid_path)
coarsest = open_image(pyramid_path, level=image_info(pyramid_path).n_levels - 1)
print(f"finest level: {finest.shape}, coarsest level: {coarsest.shape}")

# Arrays are read lazily, so this is also how to process a large image with dask.
voxels_per_label = da.unique(da.from_zarr(coarsest), return_counts=True)
labels, counts = da.compute(*voxels_per_label)
print("voxels per label in the coarsest level:")
for label, count in zip(labels, counts, strict=True):
    print(f"  {label}: {count}")

# 4. Open it in the viewer. The voxel size comes from the metadata of the file. To
# view the plain zarr array instead, pass plain_path and segmentation_voxel_size_um.
viewer = view_skeleton(str(graph_path), segmentation_path=str(pyramid_path))

# Show the segmentation, which is off by default. You can also do this with the
# "Segmentation View" controls. "all" streams the pyramid, so zoom in to see the
# finer levels load. "bounding_box" crops from the finest level and loads the crop
# into memory.
viewer.data.segmentation_view.mode = "all"

# start the GUI event loop and block until the application is closed
skeleplex.app.run()
