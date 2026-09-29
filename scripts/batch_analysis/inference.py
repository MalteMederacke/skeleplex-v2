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
    progress: bool = True,
) -> np.ndarray:
    """Run segmentation inference on a 3-D image.

    Parameters
    ----------
    image : np.ndarray
        Raw image, shape (D, H, W). It is cast to float32 and passed straight
        to the network, so it must be normalised the same way as the training
        data (no normalisation if the patches were not normalised).
    checkpoint_path : str
        Path to a ``.ckpt`` file saved during training.
    device : str
        ``"cuda"`` or ``"cpu"``.
    roi_size : tuple
        Sliding-window patch size — must match the network's expected input.
    sw_batch_size : int
        Number of patches processed in parallel during sliding-window inference.
    overlap : float
        Fractional overlap between adjacent sliding-window patches.
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
            progress=progress,
        )
        # logits shape: (1, out_channels, D, H, W)
        probs = torch.softmax(logits, dim=1)
        segmentation = torch.argmax(probs, dim=1).squeeze(0)  # (D, H, W)

    return segmentation.cpu().numpy().astype(np.uint8)
