"""Sliding-window segmentation inference for a trained DynUNet checkpoint.

Helper module imported by ``segment_batch.py`` and ``review_segmentations.py``.
"""

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from morphospaces.networks.semantic_unet import SemanticDynSkelLossUNet


def run_inference(
    image: np.ndarray,
    checkpoint_path: str,
    device: str = "cuda",
    roi_size: tuple = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    mode: str = "gaussian",
    progress: bool = True,
) -> np.ndarray:
    """Run segmentation inference on a 3-D image.

    Parameters
    ----------
    image : np.ndarray
        Raw image, shape (D, H, W). It is cast to float32 and passed straight
        to the network, so it must be normalised the same way as the training
        data — use ``normalization.normalize_image``, which is what
        ``prepare_patches.py`` applies.
    checkpoint_path : str
        Path to a ``.ckpt`` file saved during training.
    device : str
        ``"cuda"`` or ``"cpu"``.
    roi_size : tuple
        Sliding-window patch size. Must match the crop size the checkpoint was
        TRAINED on: DynUNet uses InstanceNorm, whose statistics are computed
        over the whole spatial extent of the window, so a differently sized
        window shifts every normalisation statistic in the network. Pass
        ``_constants.inference_kwargs(channel)`` to get this right by
        construction.
    sw_batch_size : int
        Number of patches processed in parallel during sliding-window inference.
    overlap : float
        Fractional overlap between adjacent sliding-window patches.
    mode : str
        Window blending. ``"gaussian"`` down-weights each window's border, where
        the network has the least context and is most likely to bridge a gap
        between two neighbouring branches. ``"constant"`` (MONAI's default)
        weights the border as much as the centre.
    progress : bool
        Show a progress bar during inference.

    Returns
    -------
    segmentation : np.ndarray
        Binary segmentation mask (foreground = 1), shape (D, H, W), dtype uint8.
    """
    net = SemanticDynSkelLossUNet.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    net.eval()
    net.to(device)

    # add batch and channel dims: (1, 1, D, H, W)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=net.forward,
            overlap=overlap,
            mode=mode,
            progress=progress,
        )
        # logits shape: (1, out_channels, D, H, W)
        probs = torch.softmax(logits, dim=1)
        segmentation = torch.argmax(probs, dim=1).squeeze(0)  # (D, H, W)

    return segmentation.cpu().numpy().astype(np.uint8)
