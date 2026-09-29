"""Worked example of a per-dataset config for the batch_analysis pipeline.

Copy this next to your data, edit it, and run the pipeline against it::

    export BATCH_ANALYSIS_CONFIG=/path/to/my_dataset/config.py
    python /path/to/scripts/batch_analysis/segment_batch.py

Every UPPERCASE name here replaces the matching default in ``_constants.py``;
anything left out keeps the default, so a real config is usually much shorter
than this one. Names starting with an underscore are private to the file and are
not treated as settings.
"""

import re
from pathlib import Path

# ── layout ───────────────────────────────────────────────────────────────────
# "flat"   one PROJECT_ROOT with zarr/, graphs/, graphs_fixed/, ...
# "nested" several EXPERIMENT_ROOTS of <condition>/ directories
LAYOUT = "flat"

PROJECT_ROOT = Path("/path/to/my_dataset")

# Only for LAYOUT = "nested":
# EXPERIMENT_ROOTS = [
#     PROJECT_ROOT / "2026-09-14_experiment_9",
#     PROJECT_ROOT / "2026-09-16_experiment_10",
# ]
# ISO_SUBDIR = "iso35"   # per-condition folder of preprocessed HDF5 files

# ── channels ─────────────────────────────────────────────────────────────────
# One entry per imaged channel. CHANNEL_PRIORITY decides which segmentation the
# single-result stages (review, graphs, analysis) use when a sample has several.
CHANNELS = ("image",)
CHANNEL_PRIORITY = CHANNELS

# ── checkpoints ──────────────────────────────────────────────────────────────
# Keep each checkpoint and its TRAIN_CROP in sync: DynUNet uses InstanceNorm,
# whose statistics span the whole sliding window, so inferring at a different
# window size than training shifts every normalisation statistic in the network.
SEG_CHECKPOINTS = {
    "image": PROJECT_ROOT / "log_segmentation/seg-best.ckpt",
}
TRAIN_CROP = {
    "image": (128, 128, 128),
}

# Normal-field skeleton-prediction model — a different model from the above.
SKELETON_CHECKPOINT = PROJECT_ROOT / "log_normal_field/reg-best.ckpt"

# ── acquisition ──────────────────────────────────────────────────────────────
# Fallback for containers with no voxel_size_um attribute, (z, y, x) in um.
DEFAULT_VOXEL_SIZE_UM = (1.0, 1.0, 1.0)

# Only used by preprocess_to_isotropic.py:
# TARGET_VOXEL_SIZE_UM = 3.5

# ── segmentation ─────────────────────────────────────────────────────────────
# True for sparse volumes that are mostly background (much faster inference);
# False when the volume is already framed around the sample.
CROP_ENABLED = True
MIN_COMPONENT_SIZE = 5000

# ── training patches ─────────────────────────────────────────────────────────
# Patches on disk are larger than the network input so training can take a
# random crop; keep PATCH_SIZE - TRAIN_CROP at about 64.
PATCH_SIZE = 192
PATCH_STRIDE = 64
MIN_FOREGROUND_FRACTION = 0.03

# Hand-curated volumes ONLY. Raw model output here teaches the new network to
# reproduce the old network's mistakes, merged branches included.
PATCH_SOURCES = [
    dict(
        path=PROJECT_ROOT / "zarr" / "curated_sample_1.zarr",
        label_key="label",
        images={"image": "image"},   # {channel: dataset key}
    ),
]

# Set only if the patch sets on disk are not named "<PATCHES_DIR>_<channel>":
# PATCH_DIRS = {"image": PROJECT_ROOT / "patches_192"}

# ── measurement ──────────────────────────────────────────────────────────────
SLICE_SPACING = 0.05
SLICE_SIZE_UM = 300
SAMPLE_GRID_SPACING_UM = 1
SAMPLE_POSITIONS = (0.01, 0.1, 0.3)
LENGTH_PRUNE_THRESHOLD = 30
ECCENTRICITY_THRESHOLD = 0.8
CIRCULARITY_THRESHOLD = 0.1
NUM_WORKERS = 10


# ── metadata parsing ─────────────────────────────────────────────────────────
# Columns added to every row of each sample's CSV. Delete this to get just the
# sample name. `sample` is the _layout.Sample, so sample.sample_dir.name is the
# condition directory in the nested layout and sample.channel is the channel the
# segmentation came from.

def SAMPLE_METADATA(stem, sample):
    """Parse a genotype out of the sample name."""
    s = stem.lower()
    genotype = (
        "het" if "het" in s else
        "MUT" if "mut" in s else
        "WT" if "wt" in s else "unknown"
    )
    m = re.search(r"sample(\d+)", stem)
    return {
        "sample": stem,
        "genotype": genotype,
        "sample_number": int(m.group(1)) if m else 0,
    }
