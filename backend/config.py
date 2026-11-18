"""
config.py
=========
Configuration for the Flask backend API.

All settings can be overridden via environment variables.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Project paths & environment configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
IMAGES_DIR = os.environ.get(
    "IMAGES_DIR", os.path.join(DATA_DIR, "Flickr8k", "Images")
)

# Ensure offline & sandbox compatibility for Ultralytics and Keras
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(MODELS_DIR, ".ultralytics"))
os.environ.setdefault("KERAS_HOME", os.path.join(MODELS_DIR, ".keras"))

# ──────────────────────────────────────────────────────────────────────────────
# Model file paths
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(MODELS_DIR, "final_model.keras")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
MAX_LENGTH_PATH = os.path.join(MODELS_DIR, "max_length.pkl")
EVAL_RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

# ──────────────────────────────────────────────────────────────────────────────
# Server settings
# ──────────────────────────────────────────────────────────────────────────────

HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", "5000"))
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")

# ──────────────────────────────────────────────────────────────────────────────
# API settings
# ──────────────────────────────────────────────────────────────────────────────

MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "16")) * 1024 * 1024  # 16 MB default
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
NUM_SAMPLE_IMAGES = 12
URL_DOWNLOAD_TIMEOUT = 15  # seconds

# ──────────────────────────────────────────────────────────────────────────────
# Inference defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_METHOD = "beam"
DEFAULT_BEAM_WIDTH = 3
MIN_BEAM_WIDTH = 1
MAX_BEAM_WIDTH = 5

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
