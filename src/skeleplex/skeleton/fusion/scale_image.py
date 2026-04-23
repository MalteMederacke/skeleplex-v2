from typing import Any

import dask.array as da
import numpy as np
import skimage
import zarr


def resize(
    image: da.Array, output_shape: tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    """
    Resize function.

    Adapted from:
    https://github.com/ome/ome-zarr-py/blob/5c5b45e46e468a3f582a583c915b9eefb636b82c/ome_zarr/dask_utils.py#L11C1-L62C63

    Wrapped copy of "skimage.transform.resize"
    Resize image to match a certain size.

    Parameters
    ----------
    image: da.Array
        Input image (dask array) to be resized.
    output_shape: tuple[int, ...]
        The shape that the resized output image should have.
    *args: Any
        Additional positional arguments to pass to skimage.transform.resize.
    **kwargs: Any
        Additional keyword arguments to pass to skimage.transform.resize.

    Returns
    -------
    da.Array
        Input arrays scaled to the specified scales.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output
    # blocks size of full numbers.
    better_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    # If E.g. we resize image from 6675 by 0.5 to 3337, factor is 0.49992509 so each
    # chunk of size e.g. 1000 will resize to 499. When assumbled into a new array, the
    # array will now be of size 3331 instead of 3337 because each of 6 chunks was
    # smaller by 1. When we compute() this, dask will read 6 chunks of 1000 and expect
    # last chunk to be 337 but instead it will only be 331.
    # So we use ceil() here (and in resize_block) to round 499.925 up to chunk of 500
    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    # Map overlap
    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        # if the input block is smaller than a 'regular' chunk (e.g. edge of image)
        # we need to calculate target size for each chunk...
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)




def scale_image(
    image: np.ndarray,
    zarr_root: str,
    scale_number: int,
) -> None:
    """
    Scale the image for the fusion algorithm.

    The image will be scaled to the powers of two
    defined in the scale ranges dictionary.
    The scale ranges dictionary should therefore contain
    the exponent as the key.
    (scale-factor = 2**key)

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    zarr_root : str
        Root zarr path under which scale arrays are stored as
        ``{zarr_root}/scale{scale_number}``.
    scale_number : int
        This value is used to map the scales to the radii in the radius_map.
    """
    scale_factor = int(1 / (2 ** (scale_number)))
    print("Scale factor: ", scale_factor)

    sz, sy, sx = image.shape

    dask_arr = resize(
        image,
        order=0,
        output_shape=(sz // scale_factor, sy // scale_factor, sx // scale_factor),
    )

    print(dask_arr.shape)

    dask_arr.to_zarr(f"{zarr_root}/scale{scale_number}", overwrite=True)
    group = zarr.open_group(zarr_root, mode="a")
    group[f"scale{scale_number}"].attrs["scale"] = [
        scale_factor,
        scale_factor,
        scale_factor,
    ]

def pad_to_match(Original_Image, rescaled_image, value=0):
    """Pad rescaled image to match the shape of the original image.

    Parameters
    ----------
    Original_Image : np.ndarray
        The original image that serves as the reference for padding.
    rescaled_image : np.ndarray
        The rescaled image that needs to be padded to match
        the shape of the original image.
    value : int, optional
        The value to use for padding, by default 0.
    """
    if Original_Image.ndim != rescaled_image.ndim:
        raise ValueError(
            "Original_Image and rescaled_image must have the same number of dimensions"
        )

    pad_width = []
    for a, b in zip(Original_Image.shape, rescaled_image.shape, strict=False):
        diff = a - b
        if diff < 0:
            raise ValueError(
                "rescaled_image must not be larger than Original_Image in any dimension"
            )

        before = diff // 2
        after = diff - before
        pad_width.append((before, after))

    return da.pad(rescaled_image, pad_width, mode="constant", constant_values=value)
