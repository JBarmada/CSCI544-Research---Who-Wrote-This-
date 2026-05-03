import os

# Dataset Settings
TEXT_COLUMN = "text"
BATCH_SIZE = 8
# Minimum word count pre-filter. Binoculars handles very short texts internally
# (returns "human" when token count is too low), so 20 words is a safe floor.
MIN_WORDS = 20

# Default HuggingFace dataset
HF_DATASET = "validname/reddit-ai-detection-english-80k"

# Binoculars Models (Falcon-7B pair used in the paper)
OBSERVER_MODEL = "tiiuae/falcon-7b"
PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"

# Binoculars Thresholds (from the paper / official repo)
# accuracy mode  — optimised for overall F1
BINO_MODE = "accuracy"
BINO_THRESHOLD = 0.9015310749276843
# low-fpr mode   — optimised for 0.01 % FPR
BINO_THRESHOLD_LOW_FPR = 0.8536432310785527

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")