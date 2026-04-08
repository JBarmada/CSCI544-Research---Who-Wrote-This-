import os

# Dataset Settings
TEXT_COLUMN = "text"
BATCH_SIZE = 8
MIN_WORDS = 50  # Binoculars needs at least ~64 tokens to be accurate

# Binoculars Settings
# We are aligning the mode and the threshold
BINO_MODE = "accuracy" 
BINO_THRESHOLD = 0.9015310749276843 

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")