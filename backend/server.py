"""
server.py
=========
Flask REST API for the Image Caption Generator.

Endpoints
---------
  GET  /api/health              — Health check with model status
  POST /api/caption             — Upload image, get caption + detected objects
  POST /api/caption/url         — Provide image URL, get caption + detected objects
  GET  /api/model/info          — Model metadata and evaluation results
  GET  /api/sample-images       — List of sample image filenames from dataset
  GET  /api/sample-images/<fn>  — Serve a sample image file

All ML models are loaded lazily at first request and cached for the
lifetime of the process.
"""

import io
import json
import logging
import os
import pickle
import random
import sys
import tempfile
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Ensure project root is on sys.path so `src.*` imports work ───────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    DEBUG,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_METHOD,
    EVAL_RESULTS_PATH,
    HOST,
    IMAGES_DIR,
    LOG_LEVEL,
    MAX_BEAM_WIDTH,
    MAX_CONTENT_LENGTH,
    MAX_LENGTH_PATH,
    MIN_BEAM_WIDTH,
    MODEL_PATH,
    MODELS_DIR,
    NUM_SAMPLE_IMAGES,
    PORT,
    TOKENIZER_PATH,
    URL_DOWNLOAD_TIMEOUT,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("caption_api")

# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# ──────────────────────────────────────────────────────────────────────────────
# Model cache (lazy singleton pattern)
# ──────────────────────────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


def _get_tokenizer():
    """Load and cache the Keras tokenizer."""
    if "tokenizer" not in _model_cache:
        logger.info("Loading tokenizer from %s …", TOKENIZER_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            _model_cache["tokenizer"] = pickle.load(f)
        logger.info("Tokenizer loaded.")
    return _model_cache["tokenizer"]


def _get_max_length() -> int:
    """Load and cache max_length."""
    if "max_length" not in _model_cache:
        logger.info("Loading max_length from %s …", MAX_LENGTH_PATH)
        with open(MAX_LENGTH_PATH, "rb") as f:
            _model_cache["max_length"] = pickle.load(f)
        logger.info("max_length = %d", _model_cache["max_length"])
    return _model_cache["max_length"]


def _get_caption_model():
    """Load and cache the trained Keras captioning model."""
    if "caption_model" not in _model_cache:
        logger.info("Loading caption model from %s …", MODEL_PATH)
        from tensorflow.keras.models import load_model

        _model_cache["caption_model"] = load_model(MODEL_PATH)
        logger.info("Caption model loaded.")
    return _model_cache["caption_model"]


def _get_feature_extractor():
    """Build and cache the InceptionV3 feature extractor."""
    if "feature_extractor" not in _model_cache:
        logger.info("Building InceptionV3 feature extractor …")
        from src.feature_extraction import build_feature_extractor

        _model_cache["feature_extractor"] = build_feature_extractor()
        logger.info("Feature extractor ready.")
    return _model_cache["feature_extractor"]


def _get_yolo_model():
    """Build and cache the YOLOv8 detector."""
    if "yolo_model" not in _model_cache:
        logger.info("Loading YOLOv8 detector …")
        from src.object_detection import build_detector

        _model_cache["yolo_model"] = build_detector()
        logger.info("YOLOv8 detector ready.")
    return _model_cache["yolo_model"]


def _load_all_models() -> Dict[str, bool]:
    """
    Attempt to load every model component, returning a status dict.
    Used by the health endpoint.
    """
    status: Dict[str, bool] = {}
    try:
        _get_tokenizer()
        status["tokenizer"] = True
    except Exception as exc:
        logger.error("Failed to load tokenizer: %s", exc)
        status["tokenizer"] = False

    try:
        _get_max_length()
        status["max_length"] = True
    except Exception as exc:
        logger.error("Failed to load max_length: %s", exc)
        status["max_length"] = False

    try:
        _get_caption_model()
        status["caption_model"] = True
    except Exception as exc:
        logger.error("Failed to load caption model: %s", exc)
        status["caption_model"] = False

    try:
        _get_feature_extractor()
        status["feature_extractor"] = True
    except Exception as exc:
        logger.error("Failed to load feature extractor: %s", exc)
        status["feature_extractor"] = False

    try:
        _get_yolo_model()
        status["yolo_model"] = True
    except Exception as exc:
        logger.error("Failed to load YOLO model: %s", exc)
        status["yolo_model"] = False

    return status


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _allowed_file(filename: str) -> bool:
    """Check whether a filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _run_inference(tmp_path: str, method: str, beam_width: int) -> Dict[str, Any]:
    """
    Run the full caption-generation pipeline on an image saved at *tmp_path*.

    Returns a dict with keys:
        caption, greedy_caption, beam_caption, objects, method, processing_time
    """
    from src.feature_extraction import load_and_preprocess_image
    from src.object_detection import (
        COCO_CLASSES,
        _detections_to_raw_vector,
        detect_objects_single,
        raw_vector_to_embedding,
    )
    from src.predict import beam_search_predict, greedy_predict

    t0 = time.perf_counter()

    feature_extractor = _get_feature_extractor()
    yolo_model = _get_yolo_model()
    model = _get_caption_model()
    tokenizer = _get_tokenizer()
    max_length = _get_max_length()

    # 1. Image features (InceptionV3)
    arr = load_and_preprocess_image(tmp_path)
    if arr is None:
        raise ValueError("Could not load/preprocess the image.")
    img_feat = feature_extractor.predict(arr[np.newaxis], verbose=0)[0]  # (2048,)

    # 2. Object detection (YOLOv8)
    cls_ids, confs, bboxes = detect_objects_single(tmp_path, yolo_model)
    raw = _detections_to_raw_vector(cls_ids, confs, bboxes)
    obj_emb = raw_vector_to_embedding(raw)  # (256,)

    # 3. Build detected-objects list
    objects: List[Dict[str, Any]] = []
    for cid, conf in zip(cls_ids, confs):
        if 0 <= cid < len(COCO_CLASSES):
            objects.append({"class": COCO_CLASSES[cid], "confidence": round(float(conf), 4)})
    objects = sorted(objects, key=lambda x: -x["confidence"])[:15]

    # 4. Generate captions
    greedy_caption = greedy_predict(img_feat, obj_emb, tokenizer, model, max_length)

    if method == "beam":
        beam_caption = beam_search_predict(
            img_feat, obj_emb, tokenizer, model, max_length, beam_width
        )
    else:
        beam_caption = greedy_caption  # Only greedy requested

    # Choose primary caption based on method
    primary_caption = beam_caption if method == "beam" else greedy_caption

    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "caption": primary_caption,
        "greedy_caption": greedy_caption,
        "beam_caption": beam_caption,
        "objects": objects,
        "method": method,
        "beam_width": beam_width,
        "processing_time": elapsed,
    }


def _save_upload_to_temp(image_data: bytes, suffix: str = ".jpg") -> str:
    """Write raw image bytes to a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(image_data)
    finally:
        tmp.close()
    return tmp.name


def _validate_method_and_beam(
    method: Optional[str], beam_width: Optional[int]
) -> Tuple[str, int]:
    """Validate and sanitise inference parameters."""
    if method is None:
        method = DEFAULT_METHOD
    method = method.lower().strip()
    if method not in ("greedy", "beam"):
        raise ValueError(f"Invalid method '{method}'. Choose 'greedy' or 'beam'.")

    if beam_width is None:
        beam_width = DEFAULT_BEAM_WIDTH
    beam_width = int(beam_width)
    if beam_width < MIN_BEAM_WIDTH or beam_width > MAX_BEAM_WIDTH:
        raise ValueError(
            f"beam_width must be between {MIN_BEAM_WIDTH} and {MAX_BEAM_WIDTH}, got {beam_width}."
        )

    return method, beam_width


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    GET /api/health

    Returns service health and per-model load status.
    """
    model_status = _load_all_models()
    all_ok = all(model_status.values())

    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "models_loaded": all_ok,
        "models": model_status,
        "models_dir": MODELS_DIR,
    }), 200 if all_ok else 503


@app.route("/api/caption", methods=["POST"])
def caption_image():
    """
    POST /api/caption

    Accept an image via multipart/form-data and return a generated caption.

    Form fields
    -----------
    image      : file (required) — the image to caption
    method     : str  (optional) — 'greedy' or 'beam' (default: 'beam')
    beam_width : int  (optional) — 1-5 (default: 3)
    """
    # ── Validate file upload ─────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided in the request."}), 400

    file = request.files["image"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    # ── Validate parameters ──────────────────────────────────────────────────
    try:
        method, beam_width = _validate_method_and_beam(
            request.form.get("method"),
            request.form.get("beam_width"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # ── Save to temp and run inference ───────────────────────────────────────
    tmp_path = None
    try:
        image_bytes = file.read()
        # Quick sanity check: try opening with PIL
        Image.open(io.BytesIO(image_bytes)).verify()

        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        tmp_path = _save_upload_to_temp(image_bytes, suffix=suffix)

        result = _run_inference(tmp_path, method, beam_width)
        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Inference failed")
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/caption/url", methods=["POST"])
def caption_image_url():
    """
    POST /api/caption/url

    Accept a JSON body with an image URL and return a generated caption.

    JSON body
    ---------
    {
        "url":        "https://example.com/photo.jpg",   (required)
        "method":     "beam",                            (optional)
        "beam_width": 3                                  (optional)
    }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    url = data.get("url")
    if not url or not isinstance(url, str):
        return jsonify({"error": "Missing or invalid 'url' field."}), 400

    # ── Validate parameters ──────────────────────────────────────────────────
    try:
        method, beam_width = _validate_method_and_beam(
            data.get("method"),
            data.get("beam_width"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # ── Download image ───────────────────────────────────────────────────────
    tmp_path = None
    try:
        resp = http_requests.get(url, timeout=URL_DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return jsonify({"error": f"URL does not point to an image (Content-Type: {content_type})."}), 400

        image_bytes = resp.content
        if len(image_bytes) > MAX_CONTENT_LENGTH:
            return jsonify({"error": f"Image exceeds maximum size ({MAX_CONTENT_LENGTH // (1024*1024)} MB)."}), 400

        # Verify it's a valid image
        Image.open(io.BytesIO(image_bytes)).verify()

        # Determine extension from URL or content-type
        ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
        suffix = ext_map.get(content_type.split(";")[0].strip(), ".jpg")
        tmp_path = _save_upload_to_temp(image_bytes, suffix=suffix)

        result = _run_inference(tmp_path, method, beam_width)
        result["source_url"] = url
        return jsonify(result), 200

    except http_requests.exceptions.Timeout:
        return jsonify({"error": f"Timeout downloading image (>{URL_DOWNLOAD_TIMEOUT}s)."}), 408
    except http_requests.exceptions.RequestException as exc:
        logger.warning("Failed to download image from URL: %s", exc)
        return jsonify({"error": f"Failed to download image: {str(exc)}"}), 400
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Inference failed for URL %s", url)
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """
    GET /api/model/info

    Return model metadata: vocab size, max_length, evaluation results,
    architecture details.
    """
    info: Dict[str, Any] = {}

    # Tokenizer info
    try:
        tokenizer = _get_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        info["vocab_size"] = vocab_size
    except Exception:
        info["vocab_size"] = None

    # Max length
    try:
        info["max_length"] = _get_max_length()
    except Exception:
        info["max_length"] = None

    # Evaluation results
    eval_res = None
    if os.path.exists(EVAL_RESULTS_PATH):
        try:
            with open(EVAL_RESULTS_PATH) as f:
                eval_res = json.load(f)
        except Exception:
            eval_res = None
    info["evaluation_results"] = eval_res
    info["eval_results"] = eval_res

    # Model file status
    info["model_files"] = {
        "final_model": os.path.exists(MODEL_PATH),
        "tokenizer": os.path.exists(TOKENIZER_PATH),
        "max_length": os.path.exists(MAX_LENGTH_PATH),
    }

    # Parameter count
    param_count = None
    try:
        model = _get_caption_model()
        param_count = model.count_params()
    except Exception:
        param_count = 12822456  # fallback approximate parameter count

    # Architecture summary
    info["architecture"] = {
        "name": "CNN-LSTM Image Captioner",
        "vision_backbone": "InceptionV3 (ImageNet pretrained, frozen)",
        "vision_feature_dim": 2048,
        "object_detector": "YOLOv8-nano (80 COCO classes)",
        "object_embedding_dim": 256,
        "word_embedding_dim": 256,
        "lstm_units": 512,
        "decoder": "Dual-LSTM with multi-modal Add fusion",
        "output": "Softmax over vocabulary",
        "total_params": param_count,
    }

    return jsonify(info), 200


@app.route("/api/sample-images", methods=["GET"])
def list_sample_images():
    """
    GET /api/sample-images

    Return a list of randomly-chosen sample image filenames from the dataset.
    """
    if not os.path.isdir(IMAGES_DIR):
        return jsonify({
            "error": f"Images directory not found: {IMAGES_DIR}",
            "images": [],
        }), 404

    all_images = [
        f
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_images:
        return jsonify({"images": [], "total_available": 0}), 200

    sample_count = min(NUM_SAMPLE_IMAGES, len(all_images))
    sampled = random.sample(all_images, sample_count)

    return jsonify({
        "images": sampled,
        "total_available": len(all_images),
        "sample_count": sample_count,
    }), 200


@app.route("/api/sample-images/<path:filename>", methods=["GET"])
def serve_sample_image(filename: str):
    """
    GET /api/sample-images/<filename>

    Serve a specific sample image from the dataset directory.
    """
    if not os.path.isdir(IMAGES_DIR):
        return jsonify({"error": "Images directory not found."}), 404

    # Sanitise: prevent directory traversal
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        return jsonify({"error": "Invalid filename."}), 400

    file_path = os.path.join(IMAGES_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": f"Image '{safe_name}' not found."}), 404

    return send_from_directory(IMAGES_DIR, safe_name)


# ──────────────────────────────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "HTTP method not allowed."}), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({"error": f"File too large. Maximum upload size is {max_mb} MB."}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error."}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (when run directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Image Caption Generator API …")
    logger.info("  Host:       %s", HOST)
    logger.info("  Port:       %d", PORT)
    logger.info("  Debug:      %s", DEBUG)
    logger.info("  Models dir: %s", MODELS_DIR)
    logger.info("  Images dir: %s", IMAGES_DIR)

    app.run(host=HOST, port=PORT, debug=DEBUG)
