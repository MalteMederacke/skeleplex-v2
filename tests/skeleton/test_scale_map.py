import numpy as np
import pytest

from skeleplex.skeleton.fusion.scale_map import (
    SCALE_MAP_BACKGROUND,
    scale_map_generator_gpu,
    scale_map_processing_gpu,
)

pytest.importorskip("cupy")

SCALE_RANGES = {0: (0, 5), -1: (5, 100)}


def _radius_map():
    radius_map = np.zeros((12, 12, 12), dtype=np.float32)
    radius_map[2:10, 2:10, 2:6] = 3  # thin part -> scale 0
    radius_map[2:10, 2:10, 6:10] = 8  # thick part -> scale -1
    return radius_map


def test_scale_map_generator_background_and_scale_zero():
    radius_map = _radius_map()
    scale_map = scale_map_generator_gpu(radius_map, SCALE_RANGES)

    assert (scale_map[radius_map == 0] == SCALE_MAP_BACKGROUND).all()
    assert (scale_map[radius_map == 3] == 0).all()
    assert (scale_map[radius_map == 8] == -1).all()


def test_scale_map_generator_uncovered_radius_raises():
    with pytest.raises(ValueError, match="not covered"):
        scale_map_generator_gpu(_radius_map(), {0: (0, 5), -1: (10, 100)})


def test_scale_map_generator_rejects_background_as_scale():
    with pytest.raises(ValueError, match="smaller than"):
        scale_map_generator_gpu(_radius_map(), {SCALE_MAP_BACKGROUND: (0, 100)})


def test_scale_map_processing_keeps_background_out_of_foreground():
    radius_map = _radius_map()
    image = radius_map > 0
    scale_map = scale_map_generator_gpu(radius_map, SCALE_RANGES)
    processed = scale_map_processing_gpu(image, scale_map, radius_map)

    assert (processed[~image] == SCALE_MAP_BACKGROUND).all()
    assert set(np.unique(processed[image])) <= {0.0, -1.0}
    # the coarser scale spreads into its surroundings, never the background
    assert (processed[radius_map == 8] == -1).all()
