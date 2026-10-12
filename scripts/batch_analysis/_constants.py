# ============================================================
# USER CONFIGURATION — adapt these values before running
# ============================================================
#
# Shared configuration for the batch analysis pipeline. Every script in this
# folder reads its defaults from here; command-line flags (where a script
# exposes them) override these values for a single run.
#
# Pipeline order:
#   0. czi_to_h5.py             raw .czi                 -> per-channel HDF5
#   0b preprocess_to_isotropic  HDF5                     -> isotropic+normalised
#   1. prepare_patches.py       curated volumes          -> HDF5 training patches
#   2. train_dynunet.py         training patches         -> segmentation model
#   3. segment_batch.py         images + model           -> segmentation zarrs
#   3b add_channel_to_zarrs.py  segmentation zarrs       -> extra image channels
#   4. review_segmentations.py  segmentation zarrs       -> curated segmentations
#   5. segmentation_to_graph_batch.py  segmentations     -> skeleton graphs
#   6. review_graphs.py         graphs + segmentations   -> curated/directed graphs
#   6b train_loop_breaker.py    curated graphs           -> loop-breaker model for 6
#   7. analyze_graphs_batch.py  curated graphs           -> measurements CSVs
#
# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
# Rather than editing this file for each dataset, point the environment
# variable BATCH_ANALYSIS_CONFIG at a small Python file that assigns only the
# settings that differ:
#
#     BATCH_ANALYSIS_CONFIG=/path/to/my_dataset_config.py python segment_batch.py
#
# Every UPPERCASE name that file defines replaces the default below. Paths
# derived from PROJECT_ROOT (ZARR_DIR, GRAPHS_DIR, ...) are re-derived from the
# config's PROJECT_ROOT unless the config sets them explicitly.
# See ``configs/`` for a worked example.

import os
import runpy
from pathlib import Path

# --- Directory layout -------------------------------------------------------
# "flat"   one PROJECT_ROOT holding zarr/, graphs/, graphs_fixed/, ... — one
#          sample per <name>.zarr. Best for a single experiment.
# "nested" several EXPERIMENT_ROOTS, each holding <condition>/ directories with
#          their own iso35/, segmentation_<channel>/, graphs/, graphs_fixed/,
#          graphs_final/. Best for treatment screens with many conditions.
# Discovery for both lives in _layout.py; scripts never glob directly.
LAYOUT = "flat"  # ADAPT HERE

# --- Project layout (LAYOUT = "flat") ---------------------------------------
# One project root holds all inputs and outputs; every other path is derived
# from it. Point this at your dataset.
PROJECT_ROOT = Path("/path/to/project")  # ADAPT HERE

# --- Experiment layout (LAYOUT = "nested") ----------------------------------
# Roots searched recursively for <condition>/ directories. Ignored when
# LAYOUT == "flat".
EXPERIMENT_ROOTS = []  # ADAPT HERE

# Sub-directory of each condition holding the preprocessed (isotropic) HDF5
# files that segmentation reads.
ISO_SUBDIR = "iso35"

# --- Channels ---------------------------------------------------------------
# Image channels the pipeline knows about. Each is segmented by its own model
# and, in the nested layout, written to its own segmentation_<channel>/ folder.
#
# CHANNEL_PRIORITY orders them for the steps that want ONE result per sample
# (review, graph building, analysis): the first channel with a segmentation
# wins. Put your best-quality channel first.
CHANNELS = ("image",)  # ADAPT HERE  — e.g. ("ssh", "dapi")
CHANNEL_PRIORITY = CHANNELS

# Array key holding the raw image inside a zarr container, per channel. In the
# flat layout a single container usually stores its image under "image".
IMAGE_KEY = "image"
LABEL_KEY = "label"

# --- Model checkpoints ------------------------------------------------------
# Segmentation model per channel, trained by train_dynunet.py.
#
# Each checkpoint is paired with the crop size it was TRAINED on. DynUNet uses
# InstanceNorm, whose statistics span the whole sliding window, so inferring
# with a different window size than training shifts every normalisation
# statistic in the network. Always change a checkpoint and its TRAIN_CROP
# together — a mismatch degrades the segmentation silently.
SEG_CHECKPOINTS = {  # ADAPT HERE
    "image": Path("/path/to/project/seg-best.ckpt"),
}
TRAIN_CROP = {  # ADAPT HERE
    "image": (128, 128, 128),
}

# Normal-field skeleton-prediction model (segmentation_to_graph_batch.py).
SKELETON_CHECKPOINT = Path("/path/to/project/reg-best.ckpt")  # ADAPT HERE

# --- Acquisition ------------------------------------------------------------
# Fallback voxel size (z, y, x) in micrometres when a container has no
# "voxel_size_um" attribute.
DEFAULT_VOXEL_SIZE_UM = (1.0, 1.0, 1.0)  # ADAPT HERE

# --- CZI conversion (czi_to_h5.py) ------------------------------------------
# Illumination wavelength (nm) -> channel name. Channels with no match are
# skipped rather than guessed at.
CZI_CHANNEL_MAP = {
    405: "dapi",
    488: "ssh",
    554: "tdtomato",
    561: "tdtomato",
}
CZI_WAVELENGTH_TOLERANCE_NM = 10

# --- Isotropic resampling (preprocess_to_isotropic.py) ----------------------
# Target isotropic voxel size in micrometres. Must match the data the
# segmentation models were trained on.
TARGET_VOXEL_SIZE_UM = 3.5
# Channels carried through resampling + normalisation.
PREPROCESS_CHANNELS = ("dapi", "ssh", "tdtomato")

# --- Intensity normalisation (normalization.py) -----------------------------
# Percentiles clipped before rescaling to [0, 1]. Training patches and
# inference inputs both go through this, so changing it invalidates existing
# checkpoints.
NORM_PERCENTILES = (0.5, 99.5)

# --- Patch extraction (prepare_patches.py) ----------------------------------
# Patches on disk are larger than the network input so training can take a
# random crop out of each one (see train_dynunet.py). Keep
# PATCH_SIZE - CROP_SIZE at about 64: that span is what teaches the network to
# be invariant to where the sliding window lands at inference time.
# Both sizes must stay divisible by 32 (the product of the DynUNet strides).
PATCH_SIZE = 192
PATCH_STRIDE = 64
MIN_FOREGROUND_FRACTION = 0.03  # keep a patch only if >= this fraction is foreground
SKELETON_DILATION_RADIUS = 2    # ball radius used to dilate the derived skeleton
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
# Wipe the patch output directories before writing. Appending to a set cut with
# a different PATCH_SIZE mixes differently sized files and silently breaks the
# random crop, so this defaults to on.
CLEAN_PATCH_OUTPUT = True

# Curated source volumes for prepare_patches.py. Each entry is a dict:
#   path       .zarr or .h5 holding the image and a hand-checked label mask
#   label_key  binary segmentation dataset
#   images     {channel: dataset key} — one training patch set per channel
#
# Use ONLY hand-curated segmentations. Training on raw model output teaches the
# new network to reproduce the old one's mistakes, merged branches included.
PATCH_SOURCES = []  # ADAPT HERE

# Where each channel's patch set lives. Left empty, a channel's directory is
# derived as "<PATCHES_DIR>_<channel>". Set entries explicitly when the sets on
# disk are already named something else.
PATCH_DIRS = {}  # ADAPT HERE  — e.g. {"ssh": PROJECT_ROOT / "patches_ssh_192"}

# --- Inference (segment_batch.py, review_segmentations.py) -------------------
# roi_size is filled in per channel from TRAIN_CROP, so it always matches the
# checkpoint being used.
INFERENCE_KWARGS = dict(
    overlap=0.75,
    sw_batch_size=4,
    # gaussian blending down-weights each window border, where the network has
    # least context and is most likely to bridge neighbouring branches.
    mode="gaussian",
)
# Discard connected components smaller than this many voxels after inference.
MIN_COMPONENT_SIZE = 5000

# --- Foreground cropping (segment_batch.py) ---------------------------------
# Sparse volumes (mostly background) are cropped to a padded bounding box
# around the sample before inference. Set CROP_ENABLED = False to segment the
# full volume.
CROP_ENABLED = True
CROP_FACTOR = 4          # downsample factor used when locating the sample
CROP_MARGIN = 48         # full-res voxels of background kept around the sample
CROP_THR_FRAC = 0.05     # foreground threshold as a fraction of the downsampled max
CROP_MIN_BLOB_FRAC = 0.02  # drop blobs smaller than this fraction of the mask

# --- Skeleton graph building (segmentation_to_graph_batch.py) ----------------
SKELETONIZE_KWARGS = dict(
    roi_size=(128, 128, 128),
    overlap=0.5,
    batch_size=3,
)
SKELETON_THRESHOLD = 0.2   # binarize the normal-field skeleton prediction
BREAK_DISTANCE = 60        # max gap (voxels) bridged by prune_and_fix_skeleton
BRANCH_TRIMMING_LEN = 15   # trim spurious skeleton branches shorter than this
# Backend for the inward unit normal field: "cpu" or "gpu".
NORMAL_FIELD_BACKEND = "cpu"

# --- Graph measurement (analyze_graphs_batch.py) ----------------------------
LENGTH_PRUNE_THRESHOLD = 30   # remove leaf edges shorter than this (branch-length units)
SLICE_SPACING = 0.05
SAMPLE_GRID_SPACING_UM = 1
SLICE_SIZE_UM = 300
NUM_WORKERS = 10
APPROX = True
SAMPLE_POSITIONS = (0.01, 0.1, 0.3)
ECCENTRICITY_THRESHOLD = 0.8
CIRCULARITY_THRESHOLD = 0.1

# Metadata columns attached to every row of a sample's CSV. Replace with a
# function of (stem, sample) to record genotype / treatment / timepoint — see
# analyze_graphs_batch.sample_metadata for the signature and an example.
SAMPLE_METADATA = None  # ADAPT HERE


# ============================================================
# Per-dataset overrides — nothing below needs editing
# ============================================================

CONFIG_ENV_VAR = "BATCH_ANALYSIS_CONFIG"

# Paths derived from PROJECT_ROOT. A config that moves PROJECT_ROOT gets these
# re-derived automatically; one that sets a name here explicitly keeps its own.
_DERIVED_FROM_ROOT = {
    "ZARR_DIR": "zarr",                # <name>.zarr sample containers
    "PATCHES_DIR": "patches",          # HDF5 training patches
    "GRAPHS_DIR": "graphs",            # raw graphs from skeletonization
    "GRAPHS_FIXED_DIR": "graphs_fixed",  # curated in review_graphs.py
    "GRAPHS_FINAL_DIR": "graphs_final",  # measured in analyze_graphs_batch.py
    "CSVS_DIR": "csvs",                # per-sample + combined measurements
    # learned from curated graphs by train_loop_breaker.py, used by review_graphs.py
    "LOOP_BREAKER_MODEL": "loop_breaker.joblib",
}


def _load_config():
    """Apply the config file named by $BATCH_ANALYSIS_CONFIG, if any.

    Returns the set of names the config assigned, so derived paths know which
    ones not to overwrite.
    """
    config_path = os.environ.get(CONFIG_ENV_VAR)
    if not config_path:
        return set()

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{CONFIG_ENV_VAR}={config_path!r} does not point at a file"
        )

    namespace = runpy.run_path(str(path))
    overrides = {
        name: value
        for name, value in namespace.items()
        if name.isupper() and not name.startswith("_")
    }
    globals().update(overrides)
    return set(overrides)


CONFIG_OVERRIDES = _load_config()

for _name, _subdir in _DERIVED_FROM_ROOT.items():
    if _name not in CONFIG_OVERRIDES:
        globals()[_name] = PROJECT_ROOT / _subdir

# CHANNEL_PRIORITY defaults to CHANNELS; re-derive if only CHANNELS was set.
if "CHANNELS" in CONFIG_OVERRIDES and "CHANNEL_PRIORITY" not in CONFIG_OVERRIDES:
    CHANNEL_PRIORITY = CHANNELS


def inference_kwargs(channel):
    """Inference keyword arguments for ``channel``, with the matching roi_size.

    roi_size comes from TRAIN_CROP so the sliding window always matches the
    crop the checkpoint was trained on.
    """
    if channel not in TRAIN_CROP:
        raise KeyError(
            f"no TRAIN_CROP entry for channel {channel!r}; "
            f"known channels: {sorted(TRAIN_CROP)}"
        )
    return dict(INFERENCE_KWARGS, roi_size=tuple(TRAIN_CROP[channel]))


def patches_dir_for(channel):
    """Directory holding one channel's training/ and validation/ patches."""
    if channel in PATCH_DIRS:
        return Path(PATCH_DIRS[channel])
    return Path(f"{PATCHES_DIR}_{channel}")


def checkpoint_for(channel):
    """Segmentation checkpoint for ``channel``."""
    if channel not in SEG_CHECKPOINTS:
        raise KeyError(
            f"no SEG_CHECKPOINTS entry for channel {channel!r}; "
            f"known channels: {sorted(SEG_CHECKPOINTS)}"
        )
    return Path(SEG_CHECKPOINTS[channel])
