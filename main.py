"""
main.py
=======
Orchestrator for the Image Caption Generator pipeline.

Runs one or more pipeline stages depending on the command-line flags:
  --preprocess        : Clean captions, build tokenizer, split dataset
  --extract-features  : Run InceptionV3 feature extraction
  --detect-objects    : Run YOLOv8 object detection + embedding
  --train             : Train the CNN-LSTM model
  --evaluate          : Evaluate on the test set (BLEU-1/2, METEOR)
  --all               : Run ALL stages in order (end-to-end)

Example usage:
  # Full pipeline
  python main.py --all

  # Only train (assumes preprocess + extract are done)
  python main.py --train

  # Evaluate with greedy decoding on first 200 test images
  python main.py --evaluate --method greedy --max-images 200
"""

import os
import sys
import pickle
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "Flickr8k")
IMAGES_DIR   = os.path.join(DATA_DIR, "Images")
CAPTIONS_TXT = os.path.join(DATA_DIR, "captions.txt")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


def _load_pkl(name: str):
    """Convenience: load a pickled artifact from models/."""
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            f"Run the prerequisite pipeline stage first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def stage_preprocess() -> None:
    """Run caption preprocessing: clean, split, build tokenizer."""
    from src.preprocess import run_preprocessing

    logger.info("═" * 60)
    logger.info("STAGE 1: Preprocessing captions")
    logger.info("═" * 60)

    train_caps, val_caps, test_caps, tokenizer, max_length = run_preprocessing(
        captions_path=CAPTIONS_TXT,
        models_dir=MODELS_DIR,
    )

    logger.info(f"Preprocessing complete.")
    logger.info(f"  Vocabulary size : {len(tokenizer.word_index) + 1:,}")
    logger.info(f"  Max seq length  : {max_length}")
    logger.info(f"  Train images    : {len(train_caps)}")
    logger.info(f"  Val images      : {len(val_caps)}")
    logger.info(f"  Test images     : {len(test_caps)}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def stage_extract_features(batch_size: int = 32) -> None:
    """Run InceptionV3 feature extraction on all images."""
    from src.feature_extraction import run_feature_extraction

    logger.info("═" * 60)
    logger.info("STAGE 2: Extracting InceptionV3 image features")
    logger.info("═" * 60)

    features = run_feature_extraction(
        images_dir=IMAGES_DIR,
        models_dir=MODELS_DIR,
        batch_size=batch_size,
    )
    logger.info(f"Feature extraction complete. {len(features)} images cached.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Object detection
# ──────────────────────────────────────────────────────────────────────────────

def stage_detect_objects() -> None:
    """Run YOLOv8 object detection and generate 256-dim embeddings."""
    from src.object_detection import run_object_detection

    logger.info("═" * 60)
    logger.info("STAGE 3: YOLOv8 object detection + embeddings")
    logger.info("═" * 60)

    embeddings = run_object_detection(
        images_dir=IMAGES_DIR,
        models_dir=MODELS_DIR,
    )
    logger.info(f"Object detection complete. {len(embeddings)} images cached.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Training
# ──────────────────────────────────────────────────────────────────────────────

def stage_train(batch_size: int = 64, max_epochs: int = 15) -> None:
    """Train the CNN-LSTM captioning model."""
    from src.train import run_training

    logger.info("═" * 60)
    logger.info("STAGE 4: Training CNN-LSTM model")
    logger.info("═" * 60)

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
        models_dir=MODELS_DIR,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )
    logger.info("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def stage_evaluate(
    method: str = "beam",
    beam_width: int = 3,
    max_images: int = None,
) -> None:
    """Evaluate the trained model on the test split."""
    from tensorflow.keras.models import load_model
    from src.evaluate import evaluate_model

    logger.info("═" * 60)
    logger.info("STAGE 5: Evaluating model on test set")
    logger.info("═" * 60)

    tokenizer      = _load_pkl("tokenizer.pkl")
    max_length     = _load_pkl("max_length.pkl")
    test_caps      = _load_pkl("test_captions.pkl")
    img_features   = _load_pkl("image_features.pkl")
    obj_embeddings = _load_pkl("object_embeddings.pkl")

    model_path = os.path.join(MODELS_DIR, "final_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run --train first."
        )
    model = load_model(model_path)

    evaluate_model(
        test_captions=test_caps,
        image_features=img_features,
        object_embeddings=obj_embeddings,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        models_dir=MODELS_DIR,
        method=method,
        beam_width=beam_width,
        max_images=max_images,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Caption Generator — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                    # Full pipeline
  python main.py --preprocess             # Only preprocess captions
  python main.py --train --max-epochs 10  # Train for 10 epochs
  python main.py --evaluate --method greedy
        """,
    )

    # Stage flags
    parser.add_argument("--all",              action="store_true", help="Run all pipeline stages")
    parser.add_argument("--preprocess",       action="store_true", help="Run preprocessing stage")
    parser.add_argument("--extract-features", action="store_true", help="Run feature extraction")
    parser.add_argument("--detect-objects",   action="store_true", help="Run object detection")
    parser.add_argument("--train",            action="store_true", help="Run training")
    parser.add_argument("--evaluate",         action="store_true", help="Run evaluation")

    # Training options
    parser.add_argument("--batch-size",  type=int, default=64,   help="Training batch size")
    parser.add_argument("--max-epochs",  type=int, default=15,   help="Max training epochs")
    parser.add_argument("--feat-batch",  type=int, default=32,   help="Feature extraction batch size")

    # Evaluation options
    parser.add_argument("--method",      choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-width",  type=int, default=3)
    parser.add_argument("--max-images",  type=int, default=None,
                        help="Evaluate on first N test images (None = all)")

    return parser.parse_args()


def main():
    args = parse_args()

    run_all         = args.all
    run_preprocess  = args.preprocess or run_all
    run_extract     = args.extract_features or run_all
    run_detect      = args.detect_objects or run_all
    run_train       = args.train or run_all
    run_eval        = args.evaluate or run_all

    if not any([run_preprocess, run_extract, run_detect, run_train, run_eval]):
        print("No stage selected. Use --help to see available options.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if run_preprocess:
        stage_preprocess()

    if run_extract:
        stage_extract_features(batch_size=args.feat_batch)

    if run_detect:
        stage_detect_objects()

    if run_train:
        stage_train(batch_size=args.batch_size, max_epochs=args.max_epochs)

    if run_eval:
        stage_evaluate(
            method=args.method,
            beam_width=args.beam_width,
            max_images=args.max_images,
        )

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
