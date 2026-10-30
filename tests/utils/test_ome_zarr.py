"""Tests for reading zarr and OME-Zarr images and writing label pyramids."""

import numpy as np
import pytest
import zarr

from skeleplex.utils import image_info, open_image, write_ome_zarr_labels


def _write_array(path, data):
    array = zarr.open(
        str(path), mode="w", shape=data.shape, chunks=(8, 8, 8), dtype=data.dtype
    )
    array[:] = data
    return path


@pytest.fixture
def labels():
    """Return a label image with a shape that is not a power of two."""
    data = np.zeros((33, 40, 21), dtype=np.uint16)
    data[4:20, 8:30, 2:12] = 7
    return data


def test_plain_array(tmp_path, labels):
    """A plain zarr array opens as it is and has no metadata."""
    path = _write_array(tmp_path / "plain.zarr", labels)

    info = image_info(path)
    assert (info.n_levels, info.voxel_size, info.level_paths) == (1, None, ())
    np.testing.assert_array_equal(open_image(path)[:], labels)
    np.testing.assert_array_equal(open_image(str(path))[:], labels)


def test_array_in_a_plain_group(tmp_path, labels):
    """An array in a plain group opens with the path that includes its key."""
    group = zarr.open_group(str(tmp_path / "flat.zarr"), mode="w")
    group.create_array("segmentation", data=labels, chunks=(8, 8, 8))

    np.testing.assert_array_equal(
        open_image(tmp_path / "flat.zarr" / "segmentation")[:], labels
    )


def test_plain_group_without_key_raises(tmp_path, labels):
    """A plain group is refused with a message naming what to pass instead."""
    group = zarr.open_group(str(tmp_path / "flat.zarr"), mode="w")
    group.create_array("segmentation", data=labels, chunks=(8, 8, 8))

    with pytest.raises(ValueError, match="without OME-Zarr multiscale metadata"):
        open_image(tmp_path / "flat.zarr")


def test_missing_level_raises(tmp_path, labels):
    """Asking for a level that does not exist raises."""
    path = _write_array(tmp_path / "plain.zarr", labels)
    with pytest.raises(IndexError, match="1 level"):
        open_image(path, level=1)


def test_older_ome_zarr_metadata(tmp_path, labels):
    """Metadata at the top level of the group is read as well (OME-Zarr v0.4)."""
    group = zarr.open_group(str(tmp_path / "old.zarr"), mode="w")
    group.create_array("0", data=labels, chunks=(8, 8, 8))
    group.attrs["multiscales"] = [
        {
            "axes": [{"name": name, "type": "space"} for name in "zyx"],
            "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 1, 1.5]}
                    ],
                }
            ],
        }
    ]

    assert image_info(tmp_path / "old.zarr").voxel_size == (1.0, 2.0, 3.0)
    np.testing.assert_array_equal(open_image(tmp_path / "old.zarr")[:], labels)


def test_write_pyramid(tmp_path, labels):
    """Each level takes every second voxel and the metadata says where it is."""
    input_path = _write_array(tmp_path / "plain.zarr", labels)
    output_path = tmp_path / "pyramid.ome.zarr"

    write_ome_zarr_labels(input_path, output_path, (2.0, 3.0, 4.0), n_levels=3)

    info = image_info(output_path)
    assert info.n_levels == 3
    assert info.voxel_size == (2.0, 3.0, 4.0)
    for level in range(3):
        factor = 2**level
        expected = labels[::factor, ::factor, ::factor]
        np.testing.assert_array_equal(open_image(output_path, level)[:], expected)
        assert open_image(output_path, level).dtype == np.int32

    multiscale = zarr.open_group(str(output_path), mode="r").attrs["ome"]["multiscales"]
    scales = [
        dataset["coordinateTransformations"][0]["scale"]
        for dataset in multiscale[0]["datasets"]
    ]
    assert scales == [[2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [8.0, 12.0, 16.0]]
    # the input is not changed
    np.testing.assert_array_equal(open_image(input_path)[:], labels)
    assert open_image(input_path).dtype == np.uint16


def test_default_number_of_levels(tmp_path):
    """By default levels are added until the largest axis is small."""
    input_path = _write_array(
        tmp_path / "plain.zarr", np.zeros((300, 40, 40), dtype=np.int32)
    )
    write_ome_zarr_labels(input_path, tmp_path / "pyramid.zarr", (1.0, 1.0, 1.0))

    assert open_image(tmp_path / "pyramid.zarr", 0).dtype == np.int32
    coarsest = image_info(tmp_path / "pyramid.zarr").n_levels - 1
    assert coarsest == 2
    assert open_image(tmp_path / "pyramid.zarr", coarsest).shape == (75, 10, 10)


def test_wide_labels_that_fit_are_converted(tmp_path):
    """The uint64 labels of the pipeline are stored as int32 if they fit."""
    data = np.zeros((8, 8, 8), dtype=np.uint64)
    data[0, 0, 0] = 2**31 - 1
    input_path = _write_array(tmp_path / "plain.zarr", data)

    write_ome_zarr_labels(input_path, tmp_path / "pyramid.zarr", (1.0, 1.0, 1.0))

    output = open_image(tmp_path / "pyramid.zarr")
    assert output.dtype == np.int32
    assert output[0, 0, 0] == 2**31 - 1


def test_labels_that_do_not_fit_raise(tmp_path):
    """Labels are refused rather than silently wrapped around."""
    data = np.zeros((8, 8, 8), dtype=np.uint64)
    data[0, 0, 0] = 2**31
    input_path = _write_array(tmp_path / "plain.zarr", data)

    with pytest.raises(ValueError, match="does not fit"):
        write_ome_zarr_labels(input_path, tmp_path / "pyramid.zarr", (1.0, 1.0, 1.0))
    assert not (tmp_path / "pyramid.zarr").exists()


def test_float_image_raises(tmp_path):
    """A probability map is refused, since it is not a label image."""
    input_path = _write_array(tmp_path / "plain.zarr", np.zeros((8, 8, 8)))
    with pytest.raises(ValueError, match="must be integers"):
        write_ome_zarr_labels(input_path, tmp_path / "pyramid.zarr", (1.0, 1.0, 1.0))


def test_existing_output_is_not_overwritten(tmp_path, labels):
    """Writing over an existing image raises and leaves it as it is."""
    input_path = _write_array(tmp_path / "plain.zarr", labels)
    with pytest.raises(FileExistsError):
        write_ome_zarr_labels(input_path, input_path, (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(open_image(input_path)[:], labels)


def test_cellier_reads_the_pyramid(tmp_path, labels):
    """The viewer's store accepts the pyramid, with the right level placement."""
    label_store = pytest.importorskip("cellier.data.label")
    input_path = _write_array(tmp_path / "plain.zarr", labels)
    output_path = tmp_path / "pyramid.ome.zarr"
    write_ome_zarr_labels(input_path, output_path, (2.0, 3.0, 4.0), n_levels=3)

    store = label_store.OMEZarrLabelDataStore.from_path(output_path.as_uri())

    assert store.n_levels == 3
    assert store.dtype == np.int32
    assert list(store.physical_scale) == [2.0, 3.0, 4.0]
    np.testing.assert_allclose(store.level_scales[2], (4.0, 4.0, 4.0))
