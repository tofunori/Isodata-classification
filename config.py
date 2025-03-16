import os

# File paths
INPUT_FILE = r"D:\UQTR\Hiver 2025\Télédétection\TP3\TR_clip.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_FILE), "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Band selection and exclusion
SELECTED_BANDS = [2, 4, 7, 13]
BANDS_TO_EXCLUDE = [10, 14, 15, 16, 17, 18]

# IsoData parameters
N_CLUSTERS_MIN = 6
N_CLUSTERS_MAX = 9
MAX_ITERATIONS = 200
MIN_SAMPLES = 40
MAX_STD_DEV = 0.45
MIN_DIST = 0.4
MAX_MERGE_PAIRS = 2
CONVERGENCE_THRESHOLD = 0.03