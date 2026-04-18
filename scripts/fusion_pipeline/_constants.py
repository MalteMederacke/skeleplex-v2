# ============================================================
# USER CONFIGURATION — adapt these values before running
# ============================================================

IMAGE_PREFIX = "LADAF-2021-17-left-v9_processed"  # ADAPT HERE
DATA_DIR = "/data"  # ADAPT HERE — root directory for all data

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

CHECKPOINT_PATH = "reg-best.ckpt"  # ADAPT HERE — path to model checkpoint

# ============================================================
# DERIVED PATHS — do not edit
# ============================================================

# --- Inputs ---
INPUT_IMAGE_PATH = f"{DATA_DIR}/{IMAGE_PREFIX}.zarr"

# --- Part 1 outputs ---
RADIUS_MAP_PATH = f"{DATA_DIR}/{IMAGE_PREFIX}_radius_map_new.zarr/scale_original"
SCALE_MAP_PATH = f"{DATA_DIR}/{IMAGE_PREFIX}_image_scale_map.zarr/scale_original"
SCALE_MAP_PROCESSED_PATH = (
    f"{DATA_DIR}/{IMAGE_PREFIX}_image_scale_map_processed.zarr/scale_original"
)

# --- Part 2 zarr roots (scripts append /scale{n} or /scale{n}_suffix) ---
SCALED_IMAGE_ZARR = f"{DATA_DIR}/{IMAGE_PREFIX}_image_scaled.zarr"
DISTANCE_FIELD_ZARR = f"{DATA_DIR}/{IMAGE_PREFIX}_distance_field_on_scales.zarr"
SKELETON_PREDICTIONS_ZARR = (
    f"{DATA_DIR}/{IMAGE_PREFIX}_skeleton_predictions_on_scales.zarr"
)
SKELETONIZED_ON_SCALES_ZARR = f"{DATA_DIR}/{IMAGE_PREFIX}_skeletonized_on_scales.zarr"
SKELETONIZED_RESCALED_ZARR = f"{DATA_DIR}/{IMAGE_PREFIX}_skeletonized_rescaled.zarr"

# --- Part 3 outputs ---
FUSED_TREE_PATH = f"{DATA_DIR}/{IMAGE_PREFIX}_fused_tree.zarr"
FINAL_SKELETON_PATH = f"{DATA_DIR}/{IMAGE_PREFIX}_final_skeleton.zarr"
FINAL_SKELETON_SKELETONIZED_PATH = (
    f"{DATA_DIR}/{IMAGE_PREFIX}_final_skeleton_skeletonized.zarr"
)
