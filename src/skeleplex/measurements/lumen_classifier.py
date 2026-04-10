import logging  # noqa
import os
import random
import shutil

import h5py
import numpy as np
import pytorch_lightning as pl
import skimage as ski
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from PIL import Image
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget


from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from skeleplex.measurements.utils import grey2rgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SaveClassifiedSlices(QWidget):
    """Widget to sort SAM-labelled slices into lumen / branch / bad folders.

    Point it at a directory of .h5 files (each with ``image`` and
    ``segmentation`` datasets, as produced by ``generate_sam_training_data``).
    The widget loads each file in turn, stepping through slices one by one.
    After every save the viewer advances to the next slice; when the last
    slice of a file is saved the next file is loaded automatically.

    Parameters
    ----------
    viewer : napari.Viewer
    input_dir : str
        Directory containing .h5 source files.
    """

    _IMAGE_LAYER = "image"
    _SEG_LAYER = "segmentation"

    def __init__(self, viewer, input_dir):
        super().__init__()
        self.viewer = viewer
        self.input_dir = input_dir
        self.files = sorted(f for f in os.listdir(input_dir) if f.endswith(".h5"))
        self.current_file_idx = 0

        self.setWindowTitle("Sort Slices")
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Progress label
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)

        # Output path selectors
        self.lumen_path = ""
        self.branch_path = ""
        self.bad_path = ""
        for label, cb in [
            ("Lumen", self.set_lumen_path),
            ("Branch", self.set_branch_path),
            ("Bad", self.set_bad_path),
        ]:
            btn = QPushButton(f"Set {label} Save Path")
            btn.clicked.connect(cb)
            layout.addWidget(btn)
            lbl = QLabel(f"{label} path: not set")
            setattr(self, f"_{label.lower()}_path_label", lbl)
            layout.addWidget(lbl)

        # Save buttons
        for label, cb in [
            ("Save as Lumen", self.save_lumen_segmentation),
            ("Save as Branch", self.save_branch_segmentation),
            ("Save as Bad", self.save_bad_segmentation),
            ("Save as Bad Lumen", self.save_bad_lumen),
            ("Skip", self._advance),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            layout.addWidget(btn)

        if self.files:
            self._load_current_file()
        else:
            logger.warning("No .h5 files found in %s", input_dir)

    # ------------------------------------------------------------------
    # Path setters
    # ------------------------------------------------------------------

    def set_lumen_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Lumen Save Directory")
        if path:
            self.lumen_path = path
            self._lumen_path_label.setText(f"Lumen path: {path}")

    def set_branch_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Branch Save Directory")
        if path:
            self.branch_path = path
            self._branch_path_label.setText(f"Branch path: {path}")

    def set_bad_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Bad Save Directory")
        if path:
            self.bad_path = path
            self._bad_path_label.setText(f"Bad path: {path}")

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_current_file(self):
        """Load the current file into the viewer, replacing existing layers."""
        file = self.files[self.current_file_idx]
        path = os.path.join(self.input_dir, file)
        logger.info("Loading %s", path)

        with h5py.File(path, "r") as f:
            image = f["image"][:]
            seg = f["segmentation"][:]

        # Replace layers (remove old ones first)
        for name in (self._IMAGE_LAYER, self._SEG_LAYER):
            if name in [l.name for l in self.viewer.layers]:
                self.viewer.layers.remove(name)

        self.viewer.add_image(image, name=self._IMAGE_LAYER)
        self.viewer.add_labels(seg, name=self._SEG_LAYER)
        self.viewer.dims.current_step = (0,) + self.viewer.dims.current_step[1:]

        n = len(self.files)
        self.progress_label.setText(
            f"File {self.current_file_idx + 1}/{n}: {file}  "
            f"({len(image)} slices)"
        )

    # ------------------------------------------------------------------
    # Current slice access
    # ------------------------------------------------------------------

    def _current_slice(self):
        """Return (image_slice, seg_slice, output_filename)."""
        step = self.viewer.dims.current_step[0]
        image = self.viewer.layers[self._IMAGE_LAYER].data
        seg = self.viewer.layers[self._SEG_LAYER].data
        stem = os.path.splitext(self.files[self.current_file_idx])[0]
        name = f"{stem}_slice{step}.h5"
        return image[step], seg[step], name

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_slice(self, save_dir, image_masked, label):
        """Crop to bounding box and write to save_dir."""
        _, _, name = self._current_slice()
        props = ski.measure.regionprops(ski.measure.label(label != 0))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            image_masked = image_masked[minr:maxr, minc:maxc]
            label = label[minr:maxr, minc:maxc]
        out = os.path.join(save_dir, name)
        with h5py.File(out, "w") as f:
            f.create_dataset("image", data=image_masked)
            f.create_dataset("label", data=label)
        logger.info("Saved → %s", out)

    def save_lumen_segmentation(self):
        if not self.lumen_path:
            logger.info("Lumen path not set!")
            return
        image, label, _ = self._current_slice()
        lumen = label.copy()
        lumen[lumen != 1] = 0
        masked = image.copy()
        masked[lumen == 0] = 0
        self._save_slice(self.lumen_path, masked, lumen)
        self._advance()

    def save_branch_segmentation(self):
        if not self.branch_path:
            logger.info("Branch path not set!")
            return
        image, label, _ = self._current_slice()
        branch = label.copy()
        branch[branch != 1] = 0
        masked = image.copy()
        masked[branch == 0] = 0
        self._save_slice(self.branch_path, masked, branch)
        self._advance()

    def save_bad_segmentation(self):
        if not self.bad_path:
            logger.info("Bad path not set!")
            return
        image, label, _ = self._current_slice()
        bad = (label != 0).astype(np.uint8)
        masked = image * bad
        self._save_slice(self.bad_path, masked, bad)
        self._advance()

    def save_bad_lumen(self):
        if not self.bad_path:
            logger.info("Bad path not set!")
            return
        image, label, _ = self._current_slice()
        bad = label.copy()
        bad[bad != 1] = 0
        masked = image.copy()
        masked[bad != 1] = 0
        self._save_slice(self.bad_path, masked, bad)
        self._advance()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _advance(self):
        """Move to next slice; load next file when current one is exhausted."""
        n_slices = len(self.viewer.layers[self._IMAGE_LAYER].data)
        next_step = self.viewer.dims.current_step[0] + 1

        if next_step < n_slices:
            self.viewer.dims.current_step = (next_step,) + self.viewer.dims.current_step[1:]
        else:
            self.current_file_idx += 1
            if self.current_file_idx < len(self.files):
                self._load_current_file()
            else:
                logger.info("All files processed!")


def add_class_based_on_folder_struct(lumen_path, branch_path, bad_path):
    """
    Add class labels to the h5 files based on their folder structure.

    Parameters
    ----------
    lumen_path : str
        Path to the folder containing lumen files.
    branch_path : str
        Path to the folder containing branch files.
    bad_path : str
        Path to the folder containing bad files.

    """
    # count number of files per class
    class_count = np.array([0, 0, 0], dtype="int32")
    lumen_files = os.listdir(lumen_path)
    for file in lumen_files:
        with h5py.File(lumen_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([0], dtype="int8")
            class_count[0] += 1

    branch_files = os.listdir(branch_path)
    for file in branch_files:
        with h5py.File(branch_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([1], dtype="int8")
            class_count[1] += 1
    bad_files = os.listdir(bad_path)
    for file in bad_files:
        with h5py.File(bad_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([2], dtype="int8")
            class_count[2] += 1
    logger.info("Number of files per class:")
    logger.info(class_count)


def generate_sam_training_data(
    input_dir,
    output_dir,
    sam_checkpoint,
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    score_threshold=0.7,
    device=None,
):
    """
    Generate SAM2-labelled slices ready for sorting with SaveClassifiedSlices.

    For each slice in every .h5 file in ``input_dir``, runs SAM2 with a
    centre-point prompt (restricted to the segmentation bounding box when a
    segmentation is present) and writes a new .h5 file containing:

    - ``image``       : original greyscale slices  (N x H x W)
    - ``segmentation``: integer label array         (N x H x W)
                         0 = background
                         1 = input binary segmentation (branch candidate)
                         2 = SAM highest-scoring mask  (lumen candidate)

    The outputs can be loaded into napari and sorted with
    ``SaveClassifiedSlices``.

    Parameters
    ----------
    input_dir : str
        Folder of .h5 files with ``image`` (N x H x W) and ``segmentation``
        (N x H x W, binary or integer) datasets.
    output_dir : str
        Destination folder (created if it does not exist).
    sam_checkpoint : str
        Path to the SAM2 checkpoint file (.pt).
    model_cfg : str
        SAM2 config path, e.g. ``"configs/sam2.1/sam2.1_hiera_l.yaml"``.
    score_threshold : float
        Minimum SAM quality score (0–1) for a mask to be considered. Among
        masks that pass, the one with the lowest mean image intensity is chosen
        — the lumen is air and therefore the darkest region in the slice.
        If no mask passes, the highest-scoring mask is used as a fallback.
        Default is 0.7.
    device : str or None
        Torch device string (``"cuda"`` / ``"cpu"``). Auto-detected if None.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "sam2 is required for this function. "
            "Install it from https://github.com/facebookresearch/segment-anything-2"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)

    sam2_model = build_sam2(model_cfg, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".h5"))
    logger.info("Found %d files in %s", len(files), input_dir)

    for file in files:
        logger.info("Processing %s", file)
        with h5py.File(os.path.join(input_dir, file), "r") as f:
            image_slices = f["image"][:]
            seg_slices = (f["segmentation"][:] != 0).astype(np.uint8)

        n_slices = len(image_slices)
        if n_slices == 0:
            logger.warning("No slices found in %s, skipping.", file)
            continue
        out_seg = np.zeros((n_slices, *image_slices.shape[1:]), dtype=np.uint8)

        for i in range(n_slices):
            image_slice = image_slices[i]
            seg_slice = seg_slices[i]

            img_rgb = grey2rgb(image_slice)
            predictor.set_image(img_rgb)

            h, w = image_slice.shape

            # Point prompt: centroid of segmentation, fallback to image centre.
            # Box prompt: bounding box of segmentation when available.
            box = None
            if seg_slice.any():
                # props = ski.measure.regionprops(ski.measure.label(seg_slice))
                # if props:
                #     cy, cx = props[0].centroid
                #     # SAM uses (x, y) = (col, row) ordering
                #     point = np.array([[cx, cy]])
                #     minr, minc, maxr, maxc = props[0].bbox
                #     box = np.array([[minc, minr, maxc, maxr]], dtype=float)
                    
                # else:
                point = np.array([[w / 2, h / 2]])
            else:
                point = np.array([[w / 2, h / 2]])

            masks, scores, _ = predictor.predict(
                point_coords=point,
                point_labels=np.array([1]),
                box=box,
                multimask_output=True,
            )

            # Lumen = air = darkest region in the slice.
            # Among masks above the quality threshold, pick the one whose
            # mean image intensity is lowest — this is more reliable than
            # area or score alone, especially with noisy input segmentations.
            # Fall back to highest-scoring mask if none pass the threshold.
            above_threshold = scores >= score_threshold
            if above_threshold.any():
                candidate_masks = masks[above_threshold]
                mean_intensities = np.array(
                    [image_slice[m.astype(bool)].mean() for m in candidate_masks]
                )
                chosen = candidate_masks[np.argmin(mean_intensities)].astype(bool)
            else:
                chosen = masks[np.argmax(scores)].astype(bool)

            label_slice = np.zeros((h, w), dtype=np.uint8)
            label_slice[chosen] = 1

            out_seg[i] = label_slice

        out_path = os.path.join(output_dir, file)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("image", data=image_slices)
            f.create_dataset("segmentation", data=out_seg)

        logger.info("Saved → %s", out_path)


def split_and_copy_files(
    file_list, source_dir, train_dir, val_dir, val_split_ratio=0.2
):
    """
    Creates training and validation files from a file list.

    The files are copied to the respective directories.

    Parameters
    ----------
    file_list : list
        List of files to be split.
    source_dir : str
        Directory containing the source files.
    train_dir : str
        Directory to copy training files to.
    val_dir : str
        Directory to copy validation files to.
    val_split_ratio : float
        Ratio of files to be used for validation.
        Default is 0.2 (20% for validation).
    """
    random.shuffle(file_list)
    split_idx = int(len(file_list) * val_split_ratio)
    val_files = file_list[:split_idx]
    train_files = file_list[split_idx:]

    # Copy validation files
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

    # Copy training files
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))


class H5FileDataset(Dataset):
    """
    Dataset reader for HDF5 files.

    Reads images and class labels from HDF5 files.
    Image needs to be stored as "image" and class label as "class_id".
    A bunch of augmentations are applied.


    """

    def __init__(self, file_paths):
        self.model_size = (256, 256)
        self.file_paths = file_paths

        self.transform = Compose(
            [
                Resize(self.model_size),
                # Spatial augmentations on PIL image
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomApply([RandomRotation(degrees=15)], p=0.3),
                RandomApply([RandomAffine(degrees=0, scale=(0.85, 1.15))], p=0.3),
                RandomResizedCrop(
                    size=self.model_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)
                ),
                # Color augmentations on PIL image
                RandomApply(
                    [
                        ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.4,
                ),
                # Normalize last, after ToTensor
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Gets item from the dataset."""
        file_path = self.file_paths[idx]

        # Open the HDF5 file
        with h5py.File(file_path, "r") as h5f:
            img = h5f["image"][:]  # Load image
            class_id = h5f["class_id"][:]  # Scalar value

        img = np.nan_to_num(img, 0)
        # Convert to PIL Image (ensure it's in RGB mode)
        img = grey2rgb(img)
        img = Image.fromarray(img)
        # Apply transformations
        img = self.transform(img)

        # Convert class_id to tensor
        class_id = torch.tensor(class_id, dtype=torch.long).squeeze()

        return img, class_id


class H5DataModule(pl.LightningDataModule):
    """Data module for loading HDF5 files for training and validation."""

    def __init__(self, data_dir="dataset", batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # def setup(self, stage=None):
        # Load all .h5 files from train and val directories
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        train_files = [
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".h5")
        ]
        val_files = [
            os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".h5")
        ]

        self.train_dataset = H5FileDataset(train_files)
        self.val_dataset = H5FileDataset(val_files)

    def train_dataloader(self):
        """Returns the training data loader."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class ConvNext3ClassClassifier(pl.LightningModule):
    """A pl model for classifying images into 3 classes using ConvNeXt-Tiny.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default is 3.
    pretrained : bool
        Whether to load ImageNet pretrained weights. Default is True.
    class_weights : list or None
        Per-class weights for the cross-entropy loss, used to handle class
        imbalance. Should be raw counts or inverse-frequency weights — they
        are normalised internally. If None, uniform weights are used.
    lr : float
        Learning rate for AdamW. Default is 1e-4.
    max_epochs : int
        Total training epochs, used to configure CosineAnnealingLR.
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        class_weights=None,
        lr=1e-4,
        max_epochs=50,
    ):
        super().__init__()
        self.save_hyperparameters()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = convnext_tiny(weights=weights)
        # Replace the classifier head
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        # Register class weights as a buffer so they move with the device
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", w / w.sum())

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """AdamW optimizer with CosineAnnealingLR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        acc = self.train_acc(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        acc = self.val_acc(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss


class ConvNext3ClassPredictor:
    """
    Predictor for the ConvNeXt-Tiny 3-class classifier.

    Loads a trained checkpoint and runs inference on single images.
    """

    def __init__(self, model_path, device=None, model_size=(256, 256)):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvNext3ClassClassifier.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.transform = Compose(
            [
                Resize(model_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, img):
        """
        Predict the class of an image.

        Parameters
        ----------
        img : numpy.ndarray
            The input image to classify.

        Returns
        -------
        int
            The predicted class index.
        float
            The confidence of the prediction.
        """
        img = np.nan_to_num(img, 0)
        img = grey2rgb(img)
        # Ensure the input is a PIL Image
        img = Image.fromarray(img)

        # Apply the transformations
        img = self.transform(img)

        # Add batch dimension
        img = img.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(img)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence
