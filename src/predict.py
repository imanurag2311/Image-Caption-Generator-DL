"""
predict.py
==========
Caption generation using a trained CNN-LSTM model.

Implements two decoding strategies:

1. Greedy Search
   At each time step, choose the word with the highest softmax probability.
   Fast (O(L) steps) but can get stuck in locally suboptimal sequences.

2. Beam Search
   Maintain the top-K candidate sequences at every step.
   Explores a wider search space at the cost of K× more computation.
   Generally produces higher-quality captions.

Usage:
    from src.predict import greedy_predict, beam_search_predict
"""

import os
import sys
import logging
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tokens
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: preprocess.py strips < > during cleaning, so the tokenizer
# stores these as plain words without angle brackets.
START_TOKEN = "startseq"
END_TOKEN   = "endseq"


# ──────────────────────────────────────────────────────────────────────────────
# Greedy search
# ──────────────────────────────────────────────────────────────────────────────

def greedy_predict(
    image_feature: np.ndarray,
    object_embedding: np.ndarray,
    tokenizer,
    model,
    max_length: int,
) -> str:
    """
    Generate a caption using greedy (argmax) decoding.

    At each step, the word with the highest predicted probability is selected
    and appended to the sequence.  Generation stops when <endseq> is produced
    or max_length is reached.

    Parameters
    ----------
    image_feature    : np.ndarray of shape (2048,) — InceptionV3 features
    object_embedding : np.ndarray of shape (256,)  — YOLOv8 embedding
    tokenizer        : fitted Keras Tokenizer
    model            : trained Keras captioning model
    max_length       : maximum number of tokens to generate

    Returns
    -------
    str — generated caption (without start/end tokens)
    """
    # Build reverse word index: id → word
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    # Resolve start token — handle both 'startseq' and '<startseq>'
    start_text = START_TOKEN
    if START_TOKEN not in tokenizer.word_index:
        start_text = "<startseq>" if "<startseq>" in tokenizer.word_index else START_TOKEN
    end_text = END_TOKEN
    if END_TOKEN not in tokenizer.word_index:
        end_text = "<endseq>" if "<endseq>" in tokenizer.word_index else END_TOKEN

    in_seq  = tokenizer.texts_to_sequences([start_text])[0]
    img_feat = image_feature.reshape(1, -1)   # (1, 2048)
    obj_feat = object_embedding.reshape(1, -1) # (1, 256)

    for _ in range(max_length):
        # Pad current sequence
        padded = _pad_sequence(in_seq, max_length)          # (1, max_len)

        # Predict next word distribution
        preds = model.predict(
            [img_feat, obj_feat, padded],
            verbose=0,
        )  # (1, vocab_size)

        # Greedy: pick argmax
        next_id = int(np.argmax(preds[0]))

        if next_id == 0:
            break  # Padding token — stop

        next_word = index_to_word.get(next_id, "")
        if next_word in (END_TOKEN, "<endseq>"):
            break

        in_seq.append(next_id)

    # Decode token ids to words, skipping start/end tokens
    _special = {"", START_TOKEN, END_TOKEN, "<startseq>", "<endseq>", "<unk>", "unk"}
    words = []
    for token_id in in_seq:
        word = index_to_word.get(token_id, "")
        if word in _special:
            continue
        words.append(word)

    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# Beam search
# ──────────────────────────────────────────────────────────────────────────────

def beam_search_predict(
    image_feature: np.ndarray,
    object_embedding: np.ndarray,
    tokenizer,
    model,
    max_length: int,
    beam_width: int = 3,
) -> str:
    """
    Generate a caption using beam search decoding.

    Maintains the top `beam_width` candidate sequences at every decoding step.
    Scores are cumulative log-probabilities (length-normalised at the end).

    Parameters
    ----------
    image_feature    : np.ndarray of shape (2048,)
    object_embedding : np.ndarray of shape (256,)
    tokenizer        : fitted Keras Tokenizer
    model            : trained Keras captioning model
    max_length       : maximum number of tokens to generate
    beam_width       : number of beams to maintain (default 3)

    Returns
    -------
    str — best generated caption (without start/end tokens)
    """
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    # Resolve tokens robustly — handle both 'startseq' and '<startseq>' variants
    start_text = START_TOKEN if START_TOKEN in tokenizer.word_index else "<startseq>"
    end_text   = END_TOKEN   if END_TOKEN   in tokenizer.word_index else "<endseq>"

    start_id = tokenizer.word_index.get(start_text, 1)
    end_id   = tokenizer.word_index.get(end_text, -1)
    _special = {"", START_TOKEN, END_TOKEN, "<startseq>", "<endseq>", "<unk>", "unk"}

    img_feat = image_feature.reshape(1, -1)    # (1, 2048)
    obj_feat = object_embedding.reshape(1, -1)  # (1, 256)

    # Each beam: (cumulative_log_prob, token_id_sequence, is_complete)
    beams: List[Tuple[float, List[int], bool]] = [
        (0.0, [start_id], False)
    ]
    completed: List[Tuple[float, List[int]]] = []

    for step in range(max_length):
        all_candidates: List[Tuple[float, List[int], bool]] = []

        for score, seq, done in beams:
            if done:
                completed.append((score / max(len(seq), 1), seq))
                continue

            padded = _pad_sequence(seq, max_length)
            preds  = model.predict(
                [img_feat, obj_feat, padded],
                verbose=0,
            )[0]  # (vocab_size,)

            # Take top beam_width words for expansion
            top_ids = np.argsort(preds)[::-1][:beam_width]

            for word_id in top_ids:
                word_id = int(word_id)
                if word_id == 0:
                    continue  # Skip padding

                log_prob  = np.log(preds[word_id] + 1e-10)
                new_score = score + log_prob
                new_seq   = seq + [word_id]
                is_done   = (word_id == end_id)

                all_candidates.append((new_score, new_seq, is_done))

        if not all_candidates:
            break

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

        # Move fully completed beams to completed list
        remaining = []
        for score, seq, done in beams:
            if done:
                completed.append((score / max(len(seq), 1), seq))
            else:
                remaining.append((score, seq, done))
        beams = remaining

        if not beams:
            break

    # Collect any remaining beams as completed
    for score, seq, done in beams:
        completed.append((score / max(len(seq), 1), seq))

    # Pick the best completed sequence (highest length-normalised log-prob)
    if not completed:
        return ""

    completed.sort(key=lambda x: x[0], reverse=True)
    best_seq = completed[0][1]

    # Decode to words, skipping special tokens
    words = []
    for token_id in best_seq:
        word = index_to_word.get(token_id, "")
        if word in _special:
            continue
        words.append(word)

    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _pad_sequence(seq: List[int], max_length: int) -> np.ndarray:
    """
    Post-pad a token sequence to max_length and return shape (1, max_length).
    Truncates from the front if longer than max_length.
    """
    seq = seq[-max_length:]  # Truncate to last max_length tokens
    padded = np.zeros((1, max_length), dtype=np.int32)
    padded[0, : len(seq)] = seq
    return padded


# ──────────────────────────────────────────────────────────────────────────────
# Predict from image file (convenience function)
# ──────────────────────────────────────────────────────────────────────────────

def predict_from_image(
    image_path: str,
    model,
    tokenizer,
    max_length: int,
    feature_extractor=None,
    yolo_model=None,
    method: str = "beam",
    beam_width: int = 3,
) -> Tuple[str, np.ndarray]:
    """
    End-to-end prediction from an image file path.

    Parameters
    ----------
    image_path        : path to the input image
    model             : trained Keras captioning model
    tokenizer         : fitted Keras Tokenizer
    max_length        : max caption length
    feature_extractor : InceptionV3 feature extractor (built if None)
    yolo_model        : YOLO detector (built if None)
    method            : 'greedy' or 'beam'
    beam_width        : beam width if method == 'beam'

    Returns
    -------
    (caption_str, object_embedding)
    """
    from src.feature_extraction import (
        build_feature_extractor,
        load_and_preprocess_image,
    )
    from src.object_detection import (
        build_detector,
        detect_objects_single,
        _detections_to_raw_vector,
        raw_vector_to_embedding,
        get_zero_embedding,
    )

    # Build models if not provided
    if feature_extractor is None:
        feature_extractor = build_feature_extractor()
    if yolo_model is None:
        yolo_model = build_detector()

    # Extract image features
    arr = load_and_preprocess_image(image_path)
    if arr is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_feat = feature_extractor.predict(arr[np.newaxis], verbose=0)[0]

    # Extract object embedding
    cls_ids, confs, bboxes = detect_objects_single(image_path, yolo_model)
    raw = _detections_to_raw_vector(cls_ids, confs, bboxes)
    obj_emb = raw_vector_to_embedding(raw)

    # Generate caption
    if method == "greedy":
        caption = greedy_predict(img_feat, obj_emb, tokenizer, model, max_length)
    else:
        caption = beam_search_predict(
            img_feat, obj_emb, tokenizer, model, max_length, beam_width
        )

    return caption, obj_emb


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pickle
    from tensorflow.keras.models import load_model

    parser = argparse.ArgumentParser(description="Generate a caption for an image")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--method",     choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-width", type=int, default=3)
    args = parser.parse_args()

    models_dir = args.models_dir

    # Load artifacts
    with open(os.path.join(models_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(models_dir, "max_length.pkl"), "rb") as f:
        max_length = pickle.load(f)

    model = load_model(os.path.join(models_dir, "final_model.keras"))

    caption, _ = predict_from_image(
        image_path=args.image,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        method=args.method,
        beam_width=args.beam_width,
    )

    print(f"\nCaption ({args.method}): {caption}")
