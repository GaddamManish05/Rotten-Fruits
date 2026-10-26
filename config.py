"""
FreshGuard AI - Configuration Module
Defines paths, class mappings, hyperparameters, and environment settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                  MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image Dimensions & Normalization
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Standard Categories (6-Class Fruit Quality Dataset)
DEFAULT_CLASSES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

CLASS_DISPLAY_NAMES = {
    "freshapples": "Fresh Apple 🍎",
    "freshbanana": "Fresh Banana 🍌",
    "freshoranges": "Fresh Orange 🍊",
    "rottenapples": "Rotten Apple 🍏🟤",
    "rottenbanana": "Rotten Banana 🍌🟤",
    "rottenoranges": "Rotten Orange 🍊🟤",
}

CLASS_FRESHNESS = {
    "freshapples": "Fresh",
    "freshbanana": "Fresh",
    "freshoranges": "Fresh",
    "rottenapples": "Rotten",
    "rottenbanana": "Rotten",
    "rottenoranges": "Rotten",
}

CLASS_FRUIT_TYPE = {
    "freshapples": "Apple",
    "freshbanana": "Banana",
    "freshoranges": "Orange",
    "rottenapples": "Apple",
    "rottenbanana": "Banana",
    "rottenoranges": "Orange",
}

# Model Checkpoint Paths
MODEL_SCRATCH_PATH = MODELS_DIR / "model_scratch.keras"
MODEL_FE_PATH = MODELS_DIR / "model_feature_extraction.keras"
MODEL_FT_PATH = MODELS_DIR / "model_finetuned.keras"
MODEL_EFFICIENTNET_PATH = MODELS_DIR / "model_efficientnet.keras"

# History and Metric Paths
HISTORY_SCRATCH_PATH = METRICS_DIR / "history_scratch.json"
HISTORY_FE_PATH = METRICS_DIR / "history_feature_extraction.json"
HISTORY_FT_PATH = METRICS_DIR / "history_finetuned.json"
COMPARISON_METRICS_PATH = METRICS_DIR / "model_comparison.json"

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
PATIENCE_EARLY_STOPPING = 4
PATIENCE_REDUCE_LR = 2

# Learning Rates
LR_SCRATCH = 1e-3
LR_FEATURE_EXTRACTION = 1e-3
LR_FINETUNING = 1e-5

# Confidence Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.80
CONFIDENCE_THRESHOLD_MEDIUM = 0.50

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
