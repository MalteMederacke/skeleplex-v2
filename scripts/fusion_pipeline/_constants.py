# ============================================================
# USER CONFIGURATION — adapt these values before running
# ============================================================

IMAGE_PREFIX = "LADAF-2021-17-left-v9_processed"  # ADAPT HERE
DATA_DIR = "/data"  # ADAPT HERE — root directory for all data
TMP_DIR = f"{DATA_DIR}/tmp"  # ADAPT HERE — scratch space for intermediate files

# Input image: path to the segmentation array inside the input zarr
# ADAPT HERE
INPUT_IMAGE_PATH = (
    f"{DATA_DIR}/LADAF-2021-17_left_lung_scale1.zarr/segmentation_final_v9"
)

SCALE_RANGES_MANUAL = {
    -1: (1, 10),
    -3: (10, 100),
    -4: (100, 1000),
}  # ADAPT HERE

THRESHOLDS = {
    -1: 0.55,
    -3: 0.7,
    -4: 0.7,
}  # ADAPT HERE

DISTANCE_FIELD_TYPE = 'normal_field'  # ADAPT HERE — 'distance_field' or 'normal_field'

CHECKPOINT_PATH = "reg-best.ckpt"  # ADAPT HERE — path to model checkpoint

# ============================================================
# DERIVED PATHS — do not edit
# All intermediate outputs live inside one zarr container.
# ============================================================

_OUTPUT_ZARR = f"{DATA_DIR}/{IMAGE_PREFIX}.zarr"

# --- Part 1 outputs ---
RADIUS_MAP_PATH = f"{_OUTPUT_ZARR}/_radius_map_new/scale_original"
SCALE_MAP_PATH = f"{_OUTPUT_ZARR}/_image_scale_map/scale_original"
SCALE_MAP_PROCESSED_PATH = f"{_OUTPUT_ZARR}/_image_scale_map_processed/scale_original"

# --- Part 2 zarr roots (scripts append /scale{n} or /scale{n}_suffix) ---
SCALED_IMAGE_ZARR = f"{_OUTPUT_ZARR}/_image_scaled"
DISTANCE_FIELD_ZARR = f"{_OUTPUT_ZARR}/_distance_field_on_scales"
SKELETON_PREDICTIONS_ZARR = f"{_OUTPUT_ZARR}/_skeleton_predictions_on_scales"
SKELETONIZED_ON_SCALES_ZARR = f"{_OUTPUT_ZARR}/_skeletonized_on_scales"
SKELETONIZED_LABELS_ON_SCALES_ZARR = f"{_OUTPUT_ZARR}/_skeletonized_labels_on_scales"
SKELETONIZED_REPAIRED_ON_SCALES_ZARR = f"{_OUTPUT_ZARR}/_skeletonized_repaired_on_scales"
SKELETONIZED_RESCALED_ZARR = f"{_OUTPUT_ZARR}/_skeletonized_rescaled"

# --- Part 3 outputs ---
FUSED_TREE_PATH = f"{_OUTPUT_ZARR}/_fused_tree"
FINAL_SKELETON_PATH = f"{_OUTPUT_ZARR}/_final_skeleton"
FINAL_SKELETON_SKELETONIZED_PATH = f"{_OUTPUT_ZARR}/_final_skeleton_skeletonized"
