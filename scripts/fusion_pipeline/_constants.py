#define your constants for the fusion pipeline

IMAGE_PREFIX = "LADAF-2021-17-left-v9_processed"  # ADAPT HERE
SCALE_RANGES_MANUAL = {
    -1: (1, 10),
    -3: (10, 100),
    -4: (100, 1000),
}  # ADAPT HERE

thresholds = {
    -1: 0.55,
    -3: 0.7,
    -4: 0.7,
}  # ADAPT HERE