"""Utilities for the skeletonization process."""

import numpy as np
import pooch
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)

# make a registry for the skeleton files
SKELETONIZATION_MODEL_REGISTRY = pooch.create(
    # folder where the data will be stored
    path=pooch.os_cache("skeleplex"),
    # Environment variable to override the cache location
    env="SKELEPLEX_CACHE",
    base_url="doi:10.5281/zenodo.14764608",
    registry={"skel-best.ckpt": "md5:fd1676cf743c1bd2f672425fcc366b5c"},
)

# registry for the normal-field model of the batch pipeline
NORMAL_FIELD_MODEL_REGISTRY = pooch.create(
    path=pooch.os_cache("skeleplex"),
    env="SKELEPLEX_CACHE",
    base_url="doi:10.5281/zenodo.23396664",
    registry={"skeleton-reg-best.ckpt": "md5:ee08849ced275384c4258ca0c7afd513"},
)


def get_skeletonization_model() -> MultiscaleSkeletonizationNet:
    """Get a pretrained model from the SkelePlex repository.

    Returns
    -------
    MultiscaleSkeletonizationNet
        The pretrained skeletonization model.
    """
    # download the weights
    file_path = SKELETONIZATION_MODEL_REGISTRY.fetch("skel-best.ckpt")

    return MultiscaleSkeletonizationNet.load_from_checkpoint(file_path)


def get_normal_field_model_path() -> str:
    """Download the pretrained normal-field model if needed and return its path.

    The weights are cached like those of the distance-field model; set
    SKELEPLEX_CACHE to change the location.

    Returns
    -------
    str
        Path to the checkpoint of the normal-field model.
    """
    return NORMAL_FIELD_MODEL_REGISTRY.fetch("skeleton-reg-best.ckpt")


def make_image_5d(image: np.ndarray) -> np.ndarray:
    """Make a 5D image from a 3D, 4D, or 5D image."""
    if image.ndim == 5:
        return image
    elif image.ndim == 4:
        return np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        return np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    else:
        raise ValueError("Image must be 3D, 4D or 5D")
