"""Shared intensity normalisation.

Training patches and inference inputs must go through the *same* function,
otherwise the network sees a different intensity distribution at test time
than it was trained on. Import this from both sides rather than duplicating.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _constants import NORM_PERCENTILES  # noqa: E402


def normalize_image(image, percentiles=None):
    """Clip to the given percentiles and rescale to [0, 1].

    Plain min-max is set by a single hot voxel, which rescales the whole
    volume; clipping the tails is robust to that.

    Idempotent: re-applying this to an already normalised volume is a no-op,
    because the clipped tails become exact 0/1 atoms. That means it is safe to
    call on volumes written by an older min-max version of the pipeline.

    Parameters
    ----------
    image : array_like
        Image of any shape; cast to float32.
    percentiles : tuple of float, optional
        Lower and upper percentile to clip at. Defaults to NORM_PERCENTILES
        from ``_constants.py``.

    Returns
    -------
    np.ndarray
        float32 array in [0, 1], same shape as ``image``.
    """
    if percentiles is None:
        percentiles = NORM_PERCENTILES
    image = np.asarray(image, dtype=np.float32).copy()
    lo, hi = np.percentile(image, percentiles)
    if hi <= lo:
        return np.zeros_like(image)
    np.clip(image, lo, hi, out=image)
    image -= lo
    image /= hi - lo
    return image
