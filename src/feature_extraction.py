"""
feature_extraction.py
=====================
Extracts 2048-dimensional image feature vectors using InceptionV3
(pretrained on ImageNet, no top classification layer) via transfer learning.

Features are cached to disk so extraction only runs once.
Each image is resized to 299×299 (InceptionV3 input size).
"""

import os
import pickle
import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE      = (299, 299)   # InceptionV3 expected input
FEATURE_DIM   = 2048         # Output dimension after GlobalAveragePooling2D
DEFAULT_BATCH = 32           # Batch size for extraction


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_extractor() -> Model:
    """
    Build an InceptionV3-based feature extractor.

    Returns a Keras Model that takes a preprocessed image tensor of shape
    (batch, 299, 299, 3) and outputs a (batch, 2048) feature vector.
    """
    logger.info("Building InceptionV3 feature extractor…")
    
    # Check for local cached weights first to support offline / sandboxed environments
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_weights = os.path.join(
        project_root, "models", ".keras", "models",
        "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    if os.path.isfile(local_weights):
        logger.info(f"Loading InceptionV3 weights from local cache: {local_weights}")
        base = InceptionV3(weights=local_weights, include_top=False, pooling="avg")
    else:
        base = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    # pooling='avg' → GlobalAveragePooling2D → output shape (batch, 2048)
    model = Model(inputs=base.input, outputs=base.output)
    model.trainable = False  # Freeze all weights
    logger.info("InceptionV3 feature extractor ready (weights frozen).")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Image loading utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load a single image, resize to IMG_SIZE, and apply InceptionV3 preprocessing.

    Returns np.ndarray of shape (299, 299, 3) or None if loading fails.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)  # Scale to [-1, 1]
        return arr
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logger.warning(f"Could not load image {image_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    image_names: List[str],
    images_dir: str,
    model: Model,
    batch_size: int = DEFAULT_BATCH,
    existing_features: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract InceptionV3 features for a list of images.

    Parameters
    ----------
    image_names       : list of image filenames (basename only)
    images_dir        : directory containing the image files
    model             : Keras feature extractor model
    batch_size        : number of images processed per forward pass
    existing_features : previously cached features dict (skip these images)

    Returns
    -------
    dict mapping image_filename → np.ndarray of shape (2048,)
    """
    features = dict(existing_features) if existing_features else {}
    
    # Only process images not already cached
    todo = [n for n in image_names if n not in features]
    if not todo:
        logger.info("All features already cached — skipping extraction.")
        return features

    logger.info(f"Extracting features for {len(todo)} images (batch_size={batch_size})…")

    for batch_start in range(0, len(todo), batch_size):
        batch_names = todo[batch_start : batch_start + batch_size]
        batch_imgs  = []
        valid_names = []

        for name in batch_names:
            img_path = os.path.join(images_dir, name)
            arr = load_and_preprocess_image(img_path)
            if arr is not None:
                batch_imgs.append(arr)
                valid_names.append(name)

        if not batch_imgs:
            continue

        batch_tensor = np.stack(batch_imgs, axis=0)   # (B, 299, 299, 3)
        batch_feats  = model.predict(batch_tensor, verbose=0)  # (B, 2048)

        for name, feat in zip(valid_names, batch_feats):
            features[name] = feat  # shape (2048,)

        done = min(batch_start + batch_size, len(todo))
        logger.info(f"  Processed {done}/{len(todo)} images…")

    logger.info(f"Feature extraction complete. Total features: {len(features)}")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_features(features: Dict[str, np.ndarray], path: str) -> None:
    """Pickle the features dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Features saved to: {path}")


def load_features(path: str) -> Dict[str, np.ndarray]:
    """Load a pickled features dict from disk."""
    with open(path, "rb") as f:
        features = pickle.load(f)
    logger.info(f"Features loaded from: {path} ({len(features)} images)")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Full extraction pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_extraction(
    images_dir: str,
    models_dir: str,
    image_names: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH,
) -> Dict[str, np.ndarray]:
    """
    End-to-end feature extraction pipeline with caching.

    Parameters
    ----------
    images_dir   : path to Flickr8k Images/ folder
    models_dir   : directory to load/save image_features.pkl
    image_names  : list of filenames to process (None → all images in dir)
    batch_size   : batch size for GPU/CPU forward pass

    Returns
    -------
    dict mapping image filename → 2048-dim feature vector
    """
    cache_path = os.path.join(models_dir, "image_features.pkl")

    # Load existing cache if available
    existing: Dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        logger.info("Found existing feature cache — loading…")
        existing = load_features(cache_path)

    # Discover all image names if not specified
    if image_names is None:
        image_names = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        logger.info(f"Found {len(image_names)} images in {images_dir}")

    # Build model only if we have new images to process
    todo = [n for n in image_names if n not in existing]
    if todo:
        model = build_feature_extractor()
        features = extract_features(
            image_names=image_names,
            images_dir=images_dir,
            model=model,
            batch_size=batch_size,
            existing_features=existing,
        )
    else:
        features = existing
        logger.info("All images already cached — no extraction needed.")

    # Save updated cache
    save_features(features, cache_path)
    return features


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract InceptionV3 features from Flickr8k")
    parser.add_argument(
        "--images-dir",
        default=os.path.join("data", "Flickr8k", "Images"),
        help="Path to Images directory",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save image_features.pkl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size for feature extraction",
    )
    args = parser.parse_args()

    features = run_feature_extraction(
        images_dir=args.images_dir,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
    )
    print(f"\nDone! Extracted features for {len(features)} images.")
    sample_name = next(iter(features))
    print(f"Sample: {sample_name} → shape {features[sample_name].shape}")
