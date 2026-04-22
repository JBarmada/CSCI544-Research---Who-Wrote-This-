import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

HF_DATASET = "validname/reddit-ai-detection-english-80k"
DEFAULT_SPLIT = "pre_2022"
DEFAULT_LOCAL_FILE = "sample_reddit_pre_2022.csv"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_RANDOM_SEED = 42

MIN_SEED_WORDS = 50
MIN_GENERATED_WORDS = 20

GENERATION_MODES = (
    "controlled_rewrite",
    "continuation",
    "style_conditioned",
)

REQUIRED_SEED_COLUMNS = (
    "text",
    "subreddit",
    "domain",
    "post_type",
    "year",
    "score",
    "created_utc",
    "id",
)


def ensure_directories():
    for path in (RAW_DATA_DIR, RESULTS_DIR, LOG_DIR, CHECKPOINT_DIR):
        os.makedirs(path, exist_ok=True)
