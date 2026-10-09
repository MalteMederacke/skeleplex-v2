"""Tests for bridging at most once between skeleton fragments."""

import numpy as np
import pytest
import zarr
from scipy.ndimage import label

from skeleplex.skeleton import repair_breaks, repair_breaks_lazy
from skeleplex.skeleton._break_detection import BridgeRegistry, filter_repairs

STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def _two_fragments_two_gaps():
    """Two U-shaped fragments facing each other across two gaps.

    Bridging both gaps connects the same two fragments twice and closes a loop.
    """
    segmentation = np.zeros((14, 24, 60), dtype=bool)
    segmentation[2:12, 2:22, 2:58] = True
    skeleton = np.zeros_like(segmentation)
    for y in (6, 16):
        skeleton[7, y, 5:26] = True  # left fragment
        skeleton[7, y, 34:55] = True  # right fragment
    skeleton[7, 6:17, 5] = True  # closes the left U
    skeleton[7, 6:17, 54] = True  # closes the right U
    # cut the corners: a right angle is a triangle of 26-connected voxels
    for y in (6, 16):
        for x in (5, 54):
            skeleton[7, y, x] = False
    return skeleton, segmentation


def _betti(skeleton):
    """Number of components and loops of a one-voxel-wide skeleton."""
    from skan import Skeleton, summarize

    summary = summarize(Skeleton(skeleton), separator="-")
    n_edges = len(summary)
    n_nodes = len(set(summary["node-id-src"]) | set(summary["node-id-dst"]))
    b0 = label(skeleton, structure=STRUCTURE)[1]
    return b0, n_edges - n_nodes + b0


def test_bridge_registry_pair():
    registry = BridgeRegistry(mode="pair")
    assert registry.accept(1, 2)
    assert not registry.accept(2, 1)
    assert registry.accept(2, 3)
    # the third side of the triangle is a new pair
    assert registry.accept(1, 3)
    assert not registry.accept(4, 4)


def test_bridge_registry_tree():
    registry = BridgeRegistry(mode="tree")
    assert registry.accept(1, 2)
    assert not registry.accept(2, 1)
    assert registry.accept(2, 3)
    # 1 and 3 are already connected through 2
    assert not registry.accept(1, 3)
    # a fragment can be bridged to several others
    assert registry.accept(2, 4)
    assert registry.accept(2, 5)
    assert not registry.accept(4, 5)


def test_bridge_registry_rejects_unknown_mode():
    with pytest.raises(ValueError):
        BridgeRegistry(mode="all")


def test_filter_repairs_keeps_shortest_bridge():
    label_map = np.zeros((5, 5, 30), dtype=np.int32)
    label_map[2, 1, :10] = 1
    label_map[2, 1, 12:] = 2
    label_map[2, 3, :8] = 1
    label_map[2, 3, 14:] = 2
    repair_start = np.array([[2, 3, 7], [2, 1, 9], [-1, -1, -1]])
    repair_end = np.array([[2, 3, 14], [2, 1, 12], [-1, -1, -1]])

    start, end = filter_repairs(
        repair_start, repair_end, label_map, BridgeRegistry(mode="tree")
    )

    # the long bridge (first row) is dropped, the short one is kept
    np.testing.assert_array_equal(start[0], [-1, -1, -1])
    np.testing.assert_array_equal(start[1], [2, 1, 9])
    np.testing.assert_array_equal(end[1], [2, 1, 12])
    # inputs are not modified
    np.testing.assert_array_equal(repair_start[0], [2, 3, 7])


@pytest.mark.parametrize(
    ("bridging", "expected_loops"), [("all", 1), ("pair", 0), ("tree", 0)]
)
def test_repair_breaks_bridging(bridging, expected_loops):
    skeleton, segmentation = _two_fragments_two_gaps()
    assert _betti(skeleton) == (2, 0)

    repaired = repair_breaks(
        skeleton, segmentation, repair_radius=12, bridging=bridging
    )

    assert _betti(repaired) == (1, expected_loops)


@pytest.mark.parametrize(
    ("bridging", "expected_loops"), [("all", 1), ("pair", 0), ("tree", 0)]
)
def test_repair_breaks_lazy_bridging_across_chunks(tmp_path, bridging, expected_loops):
    """With a global label map, the two gaps are in different chunks."""
    skeleton, segmentation = _two_fragments_two_gaps()
    label_map, _ = label(skeleton, structure=STRUCTURE)

    paths = {}
    for name, array in (
        ("skeleton", skeleton.astype(np.uint8)),
        ("segmentation", segmentation.astype(np.uint8)),
        ("labels", label_map.astype(np.int32)),
    ):
        paths[name] = tmp_path / f"{name}.zarr"
        z = zarr.open(
            str(paths[name]),
            mode="w",
            shape=array.shape,
            chunks=(14, 12, 60),
            dtype=array.dtype,
        )
        z[:] = array

    repair_breaks_lazy(
        skeleton_path=paths["skeleton"],
        segmentation_path=paths["segmentation"],
        output_path=tmp_path / "repaired.zarr",
        repair_radius=12,
        chunk_shape=(14, 12, 60),  # one gap per chunk along y
        label_map_path=paths["labels"],
        bridging=bridging,
    )

    repaired = zarr.open(str(tmp_path / "repaired.zarr"), mode="r")[:] > 0
    assert _betti(repaired) == (1, expected_loops)
