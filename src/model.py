"""
model.py
========
CNN-LSTM image captioning model with concatenation-based multi-modal fusion.

Architecture (three-input):
  ┌─────────────────────────────────────────────────────────┐
  │  Input 1: image_features   (2048,)  ← InceptionV3       │
  │  Input 2: object_features  (256,)   ← YOLOv8 embedding  │
  │  Input 3: caption_sequence (max_len,) ← partial caption  │
  └─────────────────────────────────────────────────────────┘
          ↓                    ↓                ↓
   Dense(512)+BN+Drop    Dense(256)+BN+Drop   Embedding(vocab,256)
          ↓                    ↓                ↓
          └────────────────────┘           Dropout(0.4)
                  Concatenate                   ↓
                      ↓                    LSTM(512, return_seq=True)
                 Dense(512)                     ↓
                 RepeatVector(1)           TimeDistributed Dense→ Add
                      ↓                         ↓
                      └─────────── Add ─────────┘
                                   ↓
                              LSTM(512)
                                   ↓
                            Dense(256, relu)
                                   ↓
                           Dense(vocab_size, softmax)
                                   ↓
                            Next-word prediction
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Embedding, Add,
    BatchNormalization, RepeatVector, TimeDistributed,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_FEATURE_DIM  = 2048
OBJECT_FEATURE_DIM = 256
EMBEDDING_DIM      = 256
LSTM_UNITS         = 512
DENSE_UNITS        = 512
DROPOUT_RATE       = 0.4
LEARNING_RATE      = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def build_caption_model(
    vocab_size: int,
    max_length: int,
    image_feat_dim:  int = IMAGE_FEATURE_DIM,
    object_feat_dim: int = OBJECT_FEATURE_DIM,
    embedding_dim:   int = EMBEDDING_DIM,
    lstm_units:      int = LSTM_UNITS,
    dense_units:     int = DENSE_UNITS,
    dropout_rate:    float = DROPOUT_RATE,
    learning_rate:   float = LEARNING_RATE,
) -> Model:
    """
    Build and compile the three-input CNN-LSTM captioning model.

    Parameters
    ----------
    vocab_size      : total vocabulary size (including padding token 0)
    max_length      : maximum caption sequence length
    image_feat_dim  : dimension of InceptionV3 feature vector (default 2048)
    object_feat_dim : dimension of YOLO embedding vector (default 256)
    embedding_dim   : word embedding dimension (default 256)
    lstm_units      : number of LSTM units in each LSTM layer (default 512)
    dense_units     : units in intermediate Dense layers (default 512)
    dropout_rate    : dropout probability (default 0.4)
    learning_rate   : Adam learning rate (default 1e-3)

    Returns
    -------
    Compiled Keras Model
    """
    # ── Branch 1: Image features ──────────────────────────────────────────────
    image_input = Input(shape=(image_feat_dim,), name="image_input")
    img_x = Dense(dense_units, activation="relu", name="img_dense")(image_input)
    img_x = BatchNormalization(name="img_bn")(img_x)
    img_x = Dropout(dropout_rate, name="img_dropout")(img_x)

    # ── Branch 2: Object embeddings ───────────────────────────────────────────
    object_input = Input(shape=(object_feat_dim,), name="object_input")
    obj_x = Dense(256, activation="relu", name="obj_dense")(object_input)
    obj_x = BatchNormalization(name="obj_bn")(obj_x)
    obj_x = Dropout(dropout_rate, name="obj_dropout")(obj_x)

    # ── Fuse image + object branches ─────────────────────────────────────────
    fused = Concatenate(name="img_obj_concat")([img_x, obj_x])   # (512+256=768)
    fused = Dense(dense_units, activation="relu", name="fused_dense")(fused)  # → 512
    fused = Dropout(dropout_rate, name="fused_dropout")(fused)

    # Expand to sequence dimension so we can Add with LSTM output
    # RepeatVector: (batch, 512) → (batch, 1, 512)
    fused_seq = RepeatVector(1, name="fused_repeat")(fused)       # (batch, 1, 512)

    # Project to lstm_units if needed (fused is 512, lstm output is 512 — match)
    fused_seq = TimeDistributed(
        Dense(lstm_units, activation="relu"),
        name="fused_proj",
    )(fused_seq)                                                   # (batch, 1, 512)

    # ── Branch 3: Caption sequence ───────────────────────────────────────────
    caption_input = Input(shape=(max_length,), name="caption_input")
    cap_x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="caption_embedding",
    )(caption_input)                                               # (batch, max_len, 256)
    cap_x = Dropout(dropout_rate, name="emb_dropout")(cap_x)
    cap_x = LSTM(lstm_units, return_sequences=True, name="lstm_1")(cap_x)
    # (batch, max_len, 512)

    # ── Multi-modal fusion via Add ────────────────────────────────────────────
    # Add fused image features to every time-step of LSTM output
    # fused_seq: (batch, 1, 512) — broadcast across max_len steps
    merged = Add(name="multimodal_add")([cap_x, fused_seq])       # (batch, max_len, 512)

    # ── Decoder LSTM ──────────────────────────────────────────────────────────
    dec = LSTM(lstm_units, return_sequences=False, name="lstm_2")(merged)  # (batch, 512)
    dec = Dropout(dropout_rate, name="dec_dropout")(dec)

    # ── Output head ──────────────────────────────────────────────────────────
    dec = Dense(dense_units // 2, activation="relu", name="output_dense")(dec)
    output = Dense(vocab_size, activation="softmax", name="word_output")(dec)

    # ── Assemble & compile ────────────────────────────────────────────────────
    model = Model(
        inputs=[image_input, object_input, caption_input],
        outputs=output,
        name="CNN_LSTM_ImageCaptioner",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model built and compiled.")
    model.summary(print_fn=logger.info)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_model_weights(model: Model, weights_path: str) -> Model:
    """Load saved weights into an existing model."""
    model.load_weights(weights_path)
    logger.info(f"Weights loaded from: {weights_path}")
    return model


def model_param_count(model: Model) -> dict:
    """Return a dict of total, trainable, non-trainable parameter counts."""
    total        = model.count_params()
    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = total - trainable
    return {
        "total":         total,
        "trainable":     trainable,
        "non_trainable": non_trainable,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and summarise the captioning model")
    parser.add_argument("--vocab-size",  type=int, default=8500)
    parser.add_argument("--max-length",  type=int, default=40)
    args = parser.parse_args()

    model = build_caption_model(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )
    counts = model_param_count(model)
    print(f"\nTotal parameters:     {counts['total']:,}")
    print(f"Trainable:            {counts['trainable']:,}")
    print(f"Non-trainable:        {counts['non_trainable']:,}")
