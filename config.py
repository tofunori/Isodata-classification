import os

# File paths
INPUT_FILE = r"D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\TR_clip.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_FILE), "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Band selection and exclusion
SELECTED_BANDS = [2, 4, 7, 13]
BANDS_TO_EXCLUDE = [10, 14, 15, 16, 17, 18]

# Spectral Indices Configuration
USE_SPECTRAL_INDICES = True  # Set to False to disable spectral indices
SELECTED_INDICES = ['ndvi', 'ndwi', 'ndbi']  # Indices to include in the classification
# Available indices: 'ndvi', 'ndwi', 'ndbi', 'mndwi', 'bai', 'nbr'

# Band mapping for spectral indices calculation
# Adjust these based on your sensor and band ordering
BAND_MAPPING = {
    'blue': 2,    # Band 2 in Sentinel-2
    'green': 3,   # Band 3 in Sentinel-2
    'red': 4,     # Band 4 in Sentinel-2
    'nir': 8,     # Band 8 in Sentinel-2
    'swir1': 11,  # Band 11 in Sentinel-2
    'swir2': 12   # Band 12 in Sentinel-2
}

# IsoData parameters
N_CLUSTERS_MIN = 6
N_CLUSTERS_MAX = 9
MAX_ITERATIONS = 200
MIN_SAMPLES = 40
MAX_STD_DEV = 0.45
MIN_DIST = 0.4
MAX_MERGE_PAIRS = 2
CONVERGENCE_THRESHOLD = 0.03
