"""Tests for the utilities for working with chunked arrays."""

import dask.array as da
import numpy as np
import pytest
import zarr
from scipy.ndimage import convolve

from skeleplex.utils import (
    calculate_expanded_slice,
    get_boundary_slices,
    iteratively_process_chunks_3d,
)


def test_input_array_not_3d(tmp_path):
    """Test that ValueError is raised when input array is not 3D."""
    # create a 2D dask array
    input_array = da.zeros((10, 10), chunks=(5, 5))

    # create output zarr
    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(str(output_path), mode="w", shape=(10, 10), dtype="float64")

    def dummy_func(x):
        return x

    with pytest.raises(ValueError, match=f"All input arrays must be 3D, got {input_array.ndim}D"):
        iteratively_process_chunks_3d(
            input_arrays=input_array,
            output_zarr=output_zarr,
            function_to_apply=dummy_func,
            chunk_shape=(5, 5, 5),
            extra_border=(1, 1, 1),
        )


def test_chunk_shape_not_3_tuple(tmp_path):
    """Test that ValueError is raised when chunk_shape is not a 3-tuple."""
    # create a 3D dask array
    input_array = da.zeros((10, 10, 10), chunks=(5, 5, 5))

    # create output zarr
    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        str(output_path), mode="w", shape=(10, 10, 10), dtype="float64"
    )

    def dummy_func(x):
        return x

    with pytest.raises(ValueError, match="chunk_shape must be a 3-tuple"):
        iteratively_process_chunks_3d(
            input_arrays=input_array,
            output_zarr=output_zarr,
            function_to_apply=dummy_func,
            chunk_shape=(5, 5),  # Only 2 elements
            extra_border=(1, 1, 1),
        )


def test_extra_border_not_3_tuple(tmp_path):
    """Test that ValueError is raised when extra_border is not a 3-tuple."""
    # create a 3D dask array
    input_array = da.zeros((10, 10, 10), chunks=(5, 5, 5))

    # create output zarr
    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        str(output_path), mode="w", shape=(10, 10, 10), dtype="float64"
    )

    def dummy_func(x):
        return x

    with pytest.raises(ValueError, match="extra_border must be a 3-tuple"):
        iteratively_process_chunks_3d(
            input_arrays=input_array,
            output_zarr=output_zarr,
            function_to_apply=dummy_func,
            chunk_shape=(5, 5, 5),
            extra_border=(1, 1),  # Only 2 elements
        )


def test_output_with_extra_dimensions(tmp_path):
    """Test that function works when output has extra dimensions."""
    # create a small 3D dask array
    shape = (4, 4, 4)
    chunk_shape = (2, 2, 2)

    extra_border = (1, 1, 1)

    input_data = np.random.rand(*shape).astype("float64")
    input_array = da.from_array(input_data, chunks=chunk_shape)

    # create output zarr with an extra dimension
    output_shape = (2, 4, 4, 4)  # Extra dimension of size 2
    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=output_shape,
        chunks=(2, *chunk_shape),
        dtype="float64",
    )

    def expand_func(x):
        # Expand the input chunk to have an extra dimension
        return np.stack([x, x], axis=0)

    # process the array
    iteratively_process_chunks_3d(
        input_arrays=input_array,
        output_zarr=output_zarr,
        function_to_apply=expand_func,
        chunk_shape=chunk_shape,
        extra_border=extra_border,
    )

    # load the result
    result = np.array(output_zarr[:])

    # verify the result shape
    assert result.shape == output_shape


def test_processing_with_convolution(tmp_path):
    """Test roundtrip processing with convolution to verify border handling."""
    # create a small input array with chunks that
    # require proper border handling
    shape = (5, 5, 5)
    chunk_shape = (2, 2, 2)
    extra_border = (1, 1, 1)

    # create input with unique values
    input_data = (np.arange(5 * 5 * 5) + 1).reshape(shape).astype("float64")
    input_array = da.from_array(input_data, chunks=chunk_shape)

    # create output zarr
    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        str(output_path), mode="w", shape=shape, chunks=chunk_shape, dtype="float64"
    )

    # define convolution function with a 3x3x3 kernel of ones
    kernel = np.ones((3, 3, 3), dtype="float64")

    def convolve_func(x):
        return convolve(x, kernel, mode="constant", cval=0.0)

    # process the array
    iteratively_process_chunks_3d(
        input_arrays=input_array,
        output_zarr=output_zarr,
        function_to_apply=convolve_func,
        chunk_shape=chunk_shape,
        extra_border=extra_border,
    )

    # load the result
    result = np.array(output_zarr[:])

    # compute expected result: convolve the entire input array
    expected = convolve(input_data, kernel, mode="constant", cval=0.0)

    # verify the result matches expected
    np.testing.assert_array_almost_equal(result, expected)


def test_boundary_slices():
    """Test boundary slices calculation for chunked arrays."""
    array_shape = (10, 10, 10)
    chunk_shape = (5, 7, 10)

    result = get_boundary_slices(array_shape, chunk_shape)

    # Expected boundaries:
    # - Dimension 0 (z): chunks at [0:5, 5:10] -> boundary at z=5
    # - Dimension 1 (y): chunks at [0:7, 7:10] -> boundary at y=7
    # - Dimension 2 (x): chunks at [0:10] -> no internal boundaries (only 1 chunk)

    expected = [
        (slice(4, 6), slice(None), slice(None)),  # z-boundary at 5
        (slice(None), slice(6, 8), slice(None)),  # y-boundary at 7
    ]

    assert len(result) == len(
        expected
    ), f"Expected {len(expected)} boundaries, got {len(result)}"
    assert result == expected, f"Expected {expected}, got {result}"


def test_calculate_expanded_slice():
    """Test expanded slice calculation for three different chunk positions."""
    array_shape = (100, 100, 100)
    border_size = (5, 5, 5)

    # Test 1: Chunk at corner (0, 0, 0)
    chunk_slice_corner = (slice(0, 10), slice(0, 10), slice(0, 10))
    expanded_slice, actual_border = calculate_expanded_slice(
        chunk_slice_corner, border_size, array_shape
    )

    # At corner, border cannot extend before 0
    assert expanded_slice == (slice(0, 15), slice(0, 15), slice(0, 15))
    assert actual_border == (0, 0, 0)

    # Test 2: Chunk at max corner
    chunk_slice_max = (slice(90, 100), slice(90, 100), slice(90, 100))
    expanded_slice, actual_border = calculate_expanded_slice(
        chunk_slice_max, border_size, array_shape
    )

    # At max corner, border cannot extend beyond 100
    assert expanded_slice == (slice(85, 100), slice(85, 100), slice(85, 100))
    assert actual_border == (5, 5, 5)

    # Test 3: Chunk entirely in middle
    chunk_slice_middle = (slice(40, 50), slice(40, 50), slice(40, 50))
    expanded_slice, actual_border = calculate_expanded_slice(
        chunk_slice_middle, border_size, array_shape
    )

    # In middle, full border can be applied on both sides
    assert expanded_slice == (slice(35, 55), slice(35, 55), slice(35, 55))
    assert actual_border == (5, 5, 5)
