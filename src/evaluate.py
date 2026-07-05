"""
evaluate.py
===========
Evaluation module for the CNN-LSTM image captioning model.

Computes:
  - BLEU-1 (unigram precision)
  - BLEU-2 (bigram precision)
  - METEOR (accounting for synonymy and word order)

Evaluation procedure:
  For each test image:
    1. Generate a caption using beam search (or greedy)
    2. Tokenise both the generated caption and all 5 reference captions
    3. Compute BLEU-1, BLEU-2, METEOR against all references
  Average scores across all test images → final metrics.

Results are printed to stdout and saved to models/evaluation_results.json.
"""

import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# NLTK data downloads (required on first run)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_nltk_data() -> None:
    """Download required NLTK resources if not already present."""
    for resource in ("punkt", "wordnet", "averaged_perceptron_tagger", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass  # Offline environment — meteor may not work


# ──────────────────────────────────────────────────────────────────────────────
# Caption cleanup (remove special tokens for scoring)
# ──────────────────────────────────────────────────────────────────────────────

_SPECIAL = {"<startseq>", "<endseq>", "<unk>"}


def _clean_for_scoring(caption: str) -> List[str]:
    """
    Tokenise a caption string into a list of word tokens,
    stripping special tokens and empty strings.
    """
    return [w for w in caption.lower().split() if w not in _SPECIAL and w]


# ──────────────────────────────────────────────────────────────────────────────
# Per-image scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_scores_for_caption(
    hypothesis: str,
    references: List[str],
) -> Tuple[float, float, float]:
    """
    Compute BLEU-1, BLEU-2, and METEOR for a single hypothesis
    against multiple reference captions.

    Parameters
    ----------
    hypothesis : generated caption string
    references : list of reference caption strings (up to 5)

    Returns
    -------
    (bleu1, bleu2, meteor) — all in [0, 1]
    """
    hyp_tokens   = _clean_for_scoring(hypothesis)
    ref_token_list = [_clean_for_scoring(r) for r in references]

    if not hyp_tokens:
        return 0.0, 0.0, 0.0

    smoother = SmoothingFunction().method1

    bleu1 = sentence_bleu(
        ref_token_list,
        hyp_tokens,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smoother,
    )
    bleu2 = sentence_bleu(
        ref_token_list,
        hyp_tokens,
        weights=(0.5, 0.5, 0.0, 0.0),
        smoothing_function=smoother,
    )

    # NLTK 3.8+ meteor_score expects pre-tokenized lists (NOT strings)
    try:
        meteor = meteor_score(
            ref_token_list,   # List[List[str]]
            hyp_tokens,       # List[str]
        )
    except Exception:
        meteor = 0.0

    return float(bleu1), float(bleu2), float(meteor)


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    test_captions:     Dict[str, List[str]],
    image_features:    Dict[str, np.ndarray],
    object_embeddings: Dict[str, np.ndarray],
    tokenizer,
    model,
    max_length: int,
    models_dir: str,
    method:     str = "beam",
    beam_width: int = 3,
    max_images: int = None,  # Set to a small number for quick evaluation
) -> Dict[str, float]:
    """
    Evaluate the model on the test split.

    Parameters
    ----------
    test_captions     : test split captions dict
    image_features    : {img_name: np.ndarray(2048,)} for all images
    object_embeddings : {img_name: np.ndarray(256,)} for all images
    tokenizer         : fitted Keras Tokenizer
    model             : trained Keras captioning model
    max_length        : maximum caption sequence length
    models_dir        : directory to save evaluation_results.json
    method            : decoding method ('beam' or 'greedy')
    beam_width        : beam width (used only when method='beam')
    max_images        : limit evaluation to this many images (None = all)

    Returns
    -------
    dict with keys: bleu1, bleu2, meteor, n_evaluated
    """
    _ensure_nltk_data()

    from src.predict import greedy_predict, beam_search_predict

    bleu1_scores  = []
    bleu2_scores  = []
    meteor_scores = []

    zero_obj = np.zeros(256, dtype=np.float32)

    image_list = list(test_captions.keys())
    if max_images is not None:
        image_list = image_list[:max_images]

    logger.info(
        f"Evaluating on {len(image_list)} test images "
        f"(method={method}, beam_width={beam_width})…"
    )

    for i, img_name in enumerate(image_list, 1):
        if img_name not in image_features:
            logger.warning(f"No features for {img_name} — skipping.")
            continue

        img_feat = image_features[img_name]
        obj_emb  = object_embeddings.get(img_name, zero_obj)
        references = test_captions[img_name]

        # Generate caption
        if method == "greedy":
            generated = greedy_predict(img_feat, obj_emb, tokenizer, model, max_length)
        else:
            generated = beam_search_predict(
                img_feat, obj_emb, tokenizer, model, max_length, beam_width
            )

        b1, b2, m = compute_scores_for_caption(generated, references)
        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        meteor_scores.append(m)

        if i % 100 == 0:
            logger.info(
                f"  [{i}/{len(image_list)}] "
                f"BLEU-1={np.mean(bleu1_scores):.4f} "
                f"BLEU-2={np.mean(bleu2_scores):.4f} "
                f"METEOR={np.mean(meteor_scores):.4f}"
            )

    results = {
        "bleu1":       float(np.mean(bleu1_scores))  if bleu1_scores  else 0.0,
        "bleu2":       float(np.mean(bleu2_scores))  if bleu2_scores  else 0.0,
        "meteor":      float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "n_evaluated": len(bleu1_scores),
        "method":      method,
        "beam_width":  beam_width,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Images evaluated : {results['n_evaluated']}")
    print(f"  Decoding method  : {method} (beam_width={beam_width})")
    print(f"  BLEU-1           : {results['bleu1']:.4f}")
    print(f"  BLEU-2           : {results['bleu2']:.4f}")
    print(f"  METEOR           : {results['meteor']:.4f}")
    print("=" * 50)

    # Save results
    results_path = os.path.join(models_dir, "evaluation_results.json")
    os.makedirs(models_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to: {results_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.models import load_model

    parser = argparse.ArgumentParser(description="Evaluate the captioning model on the test set")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--method",     choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to N images (for quick testing)")
    args = parser.parse_args()

    models_dir = args.models_dir

    def _load_pkl(name):
        with open(os.path.join(models_dir, name), "rb") as f:
            return pickle.load(f)

    logger.info("Loading artifacts for evaluation…")
    tokenizer      = _load_pkl("tokenizer.pkl")
    max_length     = _load_pkl("max_length.pkl")
    test_caps      = _load_pkl("test_captions.pkl")
    img_features   = _load_pkl("image_features.pkl")
    obj_embeddings = _load_pkl("object_embeddings.pkl")

    model = load_model(os.path.join(models_dir, "final_model.keras"))

    evaluate_model(
        test_captions=test_caps,
        image_features=img_features,
        object_embeddings=obj_embeddings,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        models_dir=models_dir,
        method=args.method,
        beam_width=args.beam_width,
        max_images=args.max_images,
    )
