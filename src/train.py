"""
train.py
========
Training pipeline for the CNN-LSTM image captioning model.

Uses tf.data.Dataset.from_generator() with explicit output_signature —
the correct approach for Keras 3 / TF 2.17 (raw Python generators and
tf.keras.utils.Sequence both fail with the multi-input list format).

Key components:
  - build_pairs()   : pre-computes all (img_name, in_seq, next_id) triples
  - make_dataset()  : wraps pairs in a tf.data pipeline with shuffle/batch/prefetch
  - run_training()  : builds model, creates datasets, fits with callbacks
"""

import os
import pickle
import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_caption_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default training hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
MAX_EPOCHS  = 15
PATIENCE    = 5    # Early stopping patience (increased from 3 — val_loss improves slowly)
LR_FACTOR   = 0.5  # ReduceLROnPlateau factor
LR_PATIENCE = 3    # Epochs before LR reduction (increased from 2)


# ──────────────────────────────────────────────────────────────────────────────
# Build training pairs
# ──────────────────────────────────────────────────────────────────────────────

def build_pairs(
    captions: Dict[str, List[str]],
    image_features: Dict[str, np.ndarray],
    tokenizer,
) -> List[Tuple[str, List[int], int]]:
    """
    Pre-compute all autoregressive (img_name, in_seq, next_id) triples.

    For caption "startseq a dog runs endseq" this produces:
        (img, [startseq_id],              a_id)
        (img, [startseq_id, a_id],        dog_id)
        (img, [startseq_id, a_id, dog_id],runs_id)
        ...
    """
    pairs: List[Tuple[str, List[int], int]] = []
    for img_name, cap_list in captions.items():
        if img_name not in image_features:
            continue
        for caption in cap_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                pairs.append((img_name, seq[:i], seq[i]))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# tf.data pipeline
# ──────────────────────────────────────────────────────────────────────────────

def make_dataset(
    captions: Dict[str, List[str]],
    image_features: Dict[str, np.ndarray],
    object_embeddings: Dict[str, np.ndarray],
    tokenizer,
    vocab_size: int,
    max_length: int,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a tf.data.Dataset from caption pairs.

    Each sample yielded by the generator is a single training example:
        inputs : (img_feat[2048], obj_emb[256], seq[max_length])
        label  : one_hot[vocab_size]

    The dataset is then batched, optionally shuffled, and prefetched.

    Returns
    -------
    (dataset, n_pairs)
      dataset : ready-to-use tf.data.Dataset
      n_pairs : total number of individual samples (for steps_per_epoch)
    """
    zero_obj = np.zeros(256, dtype=np.float32)

    # Pre-compute all pairs (img_name, in_seq, next_id)
    pairs = build_pairs(captions, image_features, tokenizer)
    logger.info(f"Dataset: {len(pairs)} pairs → "
                f"{math.ceil(len(pairs)/batch_size)} batches/epoch "
                f"(shuffle={shuffle})")

    # Capture references for the closure
    _image_features    = image_features
    _object_embeddings = object_embeddings
    _pairs             = pairs

    def generator():
        """Yield individual (inputs, label) samples."""
        indices = np.arange(len(_pairs))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            img_name, in_seq, next_id = _pairs[idx]

            # Image feature (2048,)
            img_feat = _image_features[img_name].astype(np.float32)

            # Object embedding (256,)
            obj_emb = _object_embeddings.get(img_name, zero_obj).astype(np.float32)

            # Pad/truncate partial caption to max_length
            trunc  = in_seq[-max_length:]
            padded = np.zeros(max_length, dtype=np.int32)
            padded[:len(trunc)] = trunc

            # One-hot label
            label = np.zeros(vocab_size, dtype=np.float32)
            if 0 < next_id < vocab_size:
                label[next_id] = 1.0

            yield (img_feat, obj_emb, padded), label

    # Explicit output_signature — required by Keras 3 / TF 2.17
    output_signature = (
        (
            tf.TensorSpec(shape=(2048,),      dtype=tf.float32, name="image_input"),
            tf.TensorSpec(shape=(256,),       dtype=tf.float32, name="object_input"),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32,  name="caption_input"),
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(8000, len(pairs)))

    dataset = (
        dataset
        .repeat()                               # Infinite stream — epoch length controlled by steps_per_epoch
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, len(pairs)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

def build_callbacks(models_dir: str) -> list:
    """
    Build training callbacks:
      - EarlyStopping   : stop if val_loss doesn't improve for 3 epochs
      - ModelCheckpoint : save best weights (by val_loss)
      - ReduceLROnPlateau: halve LR after 2 stagnant epochs
      - CSVLogger       : write per-epoch metrics to training_log.csv
    """
    os.makedirs(models_dir, exist_ok=True)
    # Save full model (.keras) so Adam optimizer state is preserved on resume
    best_model_path = os.path.join(models_dir, "best_model.keras")

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,   # Full model save → preserves Adam optimizer state
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(
            filename=os.path.join(models_dir, "training_log.csv"),
            append=False,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_training(
    train_captions:     Dict[str, List[str]],
    val_captions:       Dict[str, List[str]],
    image_features:     Dict[str, np.ndarray],
    object_embeddings:  Dict[str, np.ndarray],
    tokenizer,
    vocab_size:   int,
    max_length:   int,
    models_dir:   str,
    batch_size:   int = BATCH_SIZE,
    max_epochs:   int = MAX_EPOCHS,
):
    """
    Full training pipeline.

    Parameters
    ----------
    train_captions    : training split captions dict
    val_captions      : validation split captions dict
    image_features    : {img_name: np.ndarray(2048,)}
    object_embeddings : {img_name: np.ndarray(256,)}
    tokenizer         : fitted Keras Tokenizer
    vocab_size        : total vocabulary size
    max_length        : maximum caption sequence length
    models_dir        : directory to save weights and logs
    batch_size        : samples per batch
    max_epochs        : maximum training epochs (fixed at 15)
    """
    logger.info("Building model…")
    model = build_caption_model(vocab_size=vocab_size, max_length=max_length)

    # Resume from full model checkpoint (includes optimizer state)
    best_model_path = os.path.join(models_dir, "best_model.keras")
    if os.path.exists(best_model_path):
        logger.info(f"Resuming from full model checkpoint: {best_model_path}")
        from tensorflow.keras.models import load_model as _load_model
        model = _load_model(best_model_path)  # Restores weights + Adam state

    # Build tf.data datasets
    train_ds, n_train = make_dataset(
        captions=train_captions,
        image_features=image_features,
        object_embeddings=object_embeddings,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds, n_val = make_dataset(
        captions=val_captions,
        image_features=image_features,
        object_embeddings=object_embeddings,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(f"Train pairs: {n_train} | Val pairs: {n_val}")

    # Pre-compute steps so Keras shows a proper progress bar (N/4459 not N/Unknown)
    train_steps = math.ceil(n_train / batch_size)
    val_steps   = math.ceil(n_val   / batch_size)

    # Train — epochs fixed at 15; EarlyStopping will halt earlier if needed
    logger.info(f"Starting training — epochs: 15 (fixed), batch_size: {batch_size}")
    history = model.fit(
        train_ds,
        epochs=15,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=build_callbacks(models_dir),
        verbose=1,
    )

    # Save final model
    model.save(os.path.join(models_dir, "final_model.keras"))
    logger.info("Training complete. Final model saved.")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CNN-LSTM captioning model")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    models_dir = args.models_dir

    def _load_pkl(name):
        with open(os.path.join(models_dir, name), "rb") as f:
            return pickle.load(f)

    logger.info("Loading preprocessed artifacts…")
    tokenizer      = _load_pkl("tokenizer.pkl")
    max_length     = _load_pkl("max_length.pkl")
    train_caps     = _load_pkl("train_captions.pkl")
    val_caps       = _load_pkl("val_captions.pkl")
    img_features   = _load_pkl("image_features.pkl")
    obj_embeddings = _load_pkl("object_embeddings.pkl")

    vocab_size = len(tokenizer.word_index) + 1

    run_training(
        train_captions=train_caps,
        val_captions=val_caps,
        image_features=img_features,
        object_embeddings=obj_embeddings,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_length=max_length,
        models_dir=models_dir,
        batch_size=args.batch_size,
    )
