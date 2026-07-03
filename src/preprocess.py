"""
preprocess.py
=============
Handles loading, cleaning, and tokenizing captions from the Flickr8k dataset.

Pipeline:
  1. Load captions.txt (CSV with image,caption columns)
  2. Clean text: lowercase, remove punctuation/special chars
  3. Add <startseq> / <endseq> tokens
  4. Split images into train / val / test (60 / 20 / 20)
  5. Fit a Keras Tokenizer on training captions only
  6. Save tokenizer to models/tokenizer.pkl

Expected performance: ~8,500 unique words in vocabulary
"""

import os
import re
import pickle
import string
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 40          # Maximum caption length (in tokens)
START_TOKEN    = "<startseq>"
END_TOKEN      = "<endseq>"
RANDOM_SEED    = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_captions(captions_path: str) -> Dict[str, List[str]]:
    """
    Load raw captions from a CSV file.

    Parameters
    ----------
    captions_path : str
        Path to captions.txt — must have header: image,caption

    Returns
    -------
    dict mapping image filename → list of raw caption strings
    """
    logger.info(f"Loading captions from: {captions_path}")
    df = pd.read_csv(captions_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    captions: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        img_name = str(row["image"]).strip()
        caption  = str(row["caption"]).strip()
        captions[img_name].append(caption)

    logger.info(f"Loaded captions for {len(captions)} images.")
    return dict(captions)


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_caption(caption: str) -> str:
    """
    Clean a single caption string.

    Steps
    -----
    - Lowercase
    - Remove punctuation
    - Remove non-alphabetic tokens (numbers, special chars)
    - Collapse whitespace
    - Prepend START_TOKEN and append END_TOKEN
    """
    caption = caption.lower()
    caption = caption.translate(_PUNCT_TABLE)
    # Keep only alphabetic tokens
    tokens  = [w for w in caption.split() if w.isalpha()]
    caption = " ".join(tokens)
    caption = f"{START_TOKEN} {caption} {END_TOKEN}"
    return caption


def clean_captions(captions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Apply clean_caption() to every caption in the dataset."""
    cleaned: Dict[str, List[str]] = {}
    for img_name, caps in captions.items():
        cleaned[img_name] = [clean_caption(c) for c in caps]
    logger.info("Caption cleaning complete.")
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Dataset split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    captions: Dict[str, List[str]],
    train_ratio: float = 0.60,
    val_ratio:   float = 0.20,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split image filenames into disjoint train / val / test sets.

    Parameters
    ----------
    captions    : cleaned captions dict
    train_ratio : fraction of images for training (default 0.60)
    val_ratio   : fraction of images for validation (default 0.20)

    Returns
    -------
    train_captions, val_captions, test_captions — each is a dict
    mapping image filename → list of cleaned captions
    """
    image_names = sorted(captions.keys())  # sort for reproducibility

    # First split off the test set
    test_ratio = 1.0 - train_ratio - val_ratio
    train_imgs, temp_imgs = train_test_split(
        image_names, test_size=(val_ratio + test_ratio), random_state=RANDOM_SEED
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=RANDOM_SEED,
    )

    train_caps = {img: captions[img] for img in train_imgs}
    val_caps   = {img: captions[img] for img in val_imgs}
    test_caps  = {img: captions[img] for img in test_imgs}

    logger.info(
        f"Split sizes — Train: {len(train_caps)}, "
        f"Val: {len(val_caps)}, Test: {len(test_caps)}"
    )
    return train_caps, val_caps, test_caps


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def build_tokenizer(train_captions: Dict[str, List[str]]) -> Tokenizer:
    """
    Fit a Keras Tokenizer on training captions only.

    The tokenizer uses word-level tokenisation and
    maps every word that appears at least once.

    Parameters
    ----------
    train_captions : dict of training image → caption list

    Returns
    -------
    Fitted Keras Tokenizer
    """
    all_train_caps: List[str] = []
    for caps in train_captions.values():
        all_train_caps.extend(caps)

    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(all_train_caps)

    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index 0
    logger.info(f"Vocabulary size: {vocab_size} unique tokens.")
    return tokenizer


def get_max_length(captions: Dict[str, List[str]]) -> int:
    """Compute max caption length across all captions (capped at MAX_SEQ_LENGTH)."""
    max_len = 0
    for caps in captions.values():
        for cap in caps:
            max_len = max(max_len, len(cap.split()))
    return min(max_len, MAX_SEQ_LENGTH)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """Pickle the tokenizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to: {path}")


def load_tokenizer(path: str) -> Tokenizer:
    """Load a pickled tokenizer from disk."""
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    logger.info(f"Tokenizer loaded from: {path}")
    return tokenizer


def save_captions(captions: Dict[str, List[str]], path: str) -> None:
    """Pickle a captions dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(captions, f)
    logger.info(f"Captions saved to: {path}")


def load_captions_pkl(path: str) -> Dict[str, List[str]]:
    """Load a pickled captions dict from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    captions_path: str,
    models_dir: str,
) -> Tuple[Dict, Dict, Dict, Tokenizer, int]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    captions_path : path to data/Flickr8k/captions.txt
    models_dir    : directory to save tokenizer and split captions

    Returns
    -------
    train_caps, val_caps, test_caps, tokenizer, max_length
    """
    # 1. Load raw captions
    raw_captions = load_captions(captions_path)

    # 2. Clean captions
    clean_caps = clean_captions(raw_captions)

    # 3. Split
    train_caps, val_caps, test_caps = split_dataset(clean_caps)

    # 4. Build tokenizer from training set only
    tokenizer = build_tokenizer(train_caps)

    # 5. Compute max length from training set
    max_length = get_max_length(train_caps)
    logger.info(f"Max sequence length (training): {max_length}")

    # 6. Persist artifacts
    save_tokenizer(tokenizer, os.path.join(models_dir, "tokenizer.pkl"))
    save_captions(train_caps, os.path.join(models_dir, "train_captions.pkl"))
    save_captions(val_caps,   os.path.join(models_dir, "val_captions.pkl"))
    save_captions(test_caps,  os.path.join(models_dir, "test_captions.pkl"))

    # Save max_length as a simple pickle
    with open(os.path.join(models_dir, "max_length.pkl"), "wb") as f:
        pickle.dump(max_length, f)

    return train_caps, val_caps, test_caps, tokenizer, max_length


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Flickr8k captions")
    parser.add_argument(
        "--captions",
        default=os.path.join("data", "Flickr8k", "captions.txt"),
        help="Path to captions.txt",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save output artifacts",
    )
    args = parser.parse_args()

    train_caps, val_caps, test_caps, tokenizer, max_len = run_preprocessing(
        captions_path=args.captions,
        models_dir=args.models_dir,
    )
    print(f"\nDone! Vocab size: {len(tokenizer.word_index) + 1}, Max length: {max_len}")
    print(f"Train: {len(train_caps)} | Val: {len(val_caps)} | Test: {len(test_caps)}")
