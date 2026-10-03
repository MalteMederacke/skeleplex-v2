import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from skeleplex.skeleton._skeletonize import (  # noqa: E402
    _resolve_device,
    skeletonize,
    skeletonize_chunkwise,
)


def _identity_model():
    model = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def test_resolve_device_explicit():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto():
    device = _resolve_device(None)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_skeletonize_cpu():
    image = np.random.default_rng(0).random((20, 24, 28), dtype=np.float32)
    result = skeletonize(
        image,
        model=_identity_model(),
        roi_size=(16, 16, 16),
        progress_bar=False,
        device="cpu",
    )
    assert result.shape == image.shape
    np.testing.assert_allclose(result, image, atol=1e-5)


def test_skeletonize_chunkwise_cpu():
    import dask.array as da

    image = np.random.default_rng(0).random((20, 24, 28), dtype=np.float32)
    result = skeletonize_chunkwise(
        da.from_array(image),
        model=_identity_model(),
        chunk_size=(10, 12, 14),
        roi_size=(8, 8, 8),
        padding=(2, 2, 2),
        device="cpu",
    ).compute()
    assert result.shape == image.shape
    np.testing.assert_allclose(result, image, atol=1e-5)
