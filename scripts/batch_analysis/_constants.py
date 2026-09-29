# ============================================================
# USER CONFIGURATION — adapt these values before running
# ============================================================
#
# Shared configuration for the batch analysis pipeline. Every script in this
# folder reads its defaults from here; command-line flags (where a script
# exposes them) override these values for a single run.
#
# Pipeline order:
#   1. prepare_patches.py      raw zarr + labels        -> HDF5 training patches
#   2. train_dynunet.py        training patches         -> segmentation model
#   3. segment_batch.py        raw zarr + model         -> segmentation arrays
#   4. review_segmentations.py segmentation arrays      -> curated segmentations
#      (build skeleton graphs from the curated segmentations, then:)
#   5. review_graphs.py        graphs + segmentations   -> curated/directed graphs
#   6. analyze_graphs_batch.py curated graphs           -> measurements CSVs

from pathlib import Path

# --- Project layout ---------------------------------------------------------
# One project root holds all inputs and outputs; every other path is derived
# from it. Point this at your dataset.
PROJECT_ROOT = Path("/path/to/project")  # ADAPT HERE

# Directory of per-sample zarr containers (<name>.zarr). Each container is a
# zarr group with at least an "image" array and a "voxel_size_um" attribute.
ZARR_DIR = PROJECT_ROOT / "zarr"

# HDF5 training patches (prepare_patches.py -> train_dynunet.py)
PATCHES_DIR = PROJECT_ROOT / "patches"

# Skeleton graphs and their curated / final variants
GRAPHS_DIR = PROJECT_ROOT / "graphs"              # raw graphs from skeletonization
GRAPHS_FIXED_DIR = PROJECT_ROOT / "graphs_fixed"  # curated in review_graphs.py
GRAPHS_FINAL_DIR = PROJECT_ROOT / "graphs_final"  # measured in analyze_graphs_batch.py
CSVS_DIR = PROJECT_ROOT / "csvs"                  # per-sample + combined measurements

# --- Model checkpoints ------------------------------------------------------
# Segmentation model trained by train_dynunet.py.
SEG_CHECKPOINT = PROJECT_ROOT / "seg-best.ckpt"  # ADAPT HERE

# Normal-field skeleton-prediction model (segmentation_to_graph_batch.py).
SKELETON_CHECKPOINT = PROJECT_ROOT / "reg-best.ckpt"  # ADAPT HERE

# --- Zarr array keys --------------------------------------------------------
# Which arrays inside each zarr group hold the image and the label mask.
IMAGE_KEY = "image"
LABEL_KEY = "label"

# --- Acquisition ------------------------------------------------------------
# Fallback voxel size (z, y, x) in micrometres when a zarr has no
# "voxel_size_um" attribute.
DEFAULT_VOXEL_SIZE_UM = (1.0, 1.0, 1.0)  # ADAPT HERE

# --- Patch extraction (prepare_patches.py) ----------------------------------
PATCH_SIZE = 128
PATCH_STRIDE = 64
MIN_FOREGROUND_FRACTION = 0.1   # keep a patch only if >= this fraction is foreground
SKELETON_DILATION_RADIUS = 2    # ball radius used to dilate the derived skeleton
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

# --- Inference (segment_batch.py, review_segmentations.py) -------------------
INFERENCE_KWARGS = dict(
    overlap=0.7,
    roi_size=(128, 128, 128),
    sw_batch_size=2,
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
