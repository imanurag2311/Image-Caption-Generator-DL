"""
object_detection.py
===================
Uses YOLOv8n (nano) to perform object detection on all Flickr8k images
and encode the detected objects into 256-dimensional semantic embeddings.

Detection pipeline:
  1. Run YOLOv8n on each image (confidence threshold = 0.30)
  2. Extract detected class IDs (from COCO 80-class set), confidences,
     and normalised bounding boxes [cx, cy, w, h]
  3. Aggregate per-class: max confidence, mean bbox coordinates
  4. Produce a 256-dim vector via a fixed linear projection

The 256-dim embedding captures:
  - Which COCO objects are present (80 presence flags)
  - Per-class max confidence (80 values)
  - Spatial summary (avg of detected bbox centroids, up to top-5 objects)

The linear projection (80+80+10 = 170 → 256) uses a seeded random
weight matrix so it is deterministic across runs without training.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
NUM_COCO_CLASSES   = 80      # YOLOv8 / COCO classes
EMBEDDING_DIM      = 256     # Output embedding dimension
CONFIDENCE_THRESH  = 0.30    # Minimum detection confidence to keep
MAX_DETECTIONS     = 20      # Cap on detections per image
INPUT_DIM          = 170     # 80 presence + 80 confidence + 10 spatial features
PROJ_SEED          = 2024    # For reproducible random projection matrix


# COCO class names (80 classes, 0-indexed to match YOLOv8 output)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ──────────────────────────────────────────────────────────────────────────────
# Build the random projection matrix (deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def _build_projection_matrix() -> np.ndarray:
    """
    Create a fixed random projection matrix of shape (INPUT_DIM, EMBEDDING_DIM).
    Uses a seeded RNG so it is identical across Python sessions.
    """
    rng = np.random.default_rng(PROJ_SEED)
    W = rng.standard_normal((INPUT_DIM, EMBEDDING_DIM)).astype(np.float32)
    # Normalise columns so projections have unit variance
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return W


_PROJECTION_MATRIX = _build_projection_matrix()


# ──────────────────────────────────────────────────────────────────────────────
# Raw detection → feature vector
# ──────────────────────────────────────────────────────────────────────────────

def _detections_to_raw_vector(
    class_ids: np.ndarray,
    confidences: np.ndarray,
    bboxes: np.ndarray,
) -> np.ndarray:
    """
    Convert YOLOv8 detection outputs into a 170-dim raw feature vector.

    Layout:
      [0:80]   — binary presence indicator per COCO class
      [80:160] — max confidence per COCO class (0 if not detected)
      [160:170]— spatial summary: [mean_cx, mean_cy, mean_w, mean_h,
                                   std_cx,  std_cy,  std_w,  std_h,
                                   count_norm, entropy]

    Parameters
    ----------
    class_ids   : int array of detected class indices
    confidences : float array of confidence scores (same length)
    bboxes      : float array of shape (N, 4) — [cx, cy, w, h] normalised

    Returns
    -------
    np.ndarray of shape (170,)
    """
    presence    = np.zeros(NUM_COCO_CLASSES, dtype=np.float32)
    max_conf    = np.zeros(NUM_COCO_CLASSES, dtype=np.float32)

    for cls_id, conf in zip(class_ids, confidences):
        if 0 <= cls_id < NUM_COCO_CLASSES:
            presence[cls_id] = 1.0
            max_conf[cls_id] = max(max_conf[cls_id], float(conf))

    # Spatial summary (top-MAX_DETECTIONS by confidence)
    if len(bboxes) > 0:
        bboxes = np.array(bboxes, dtype=np.float32)
        mean_vals = bboxes.mean(axis=0)           # (4,)
        std_vals  = bboxes.std(axis=0) + 1e-8     # (4,)
        count_norm = min(len(bboxes), MAX_DETECTIONS) / MAX_DETECTIONS

        # Simple entropy over class distribution
        class_counts = np.bincount(class_ids.astype(int), minlength=NUM_COCO_CLASSES)
        probs = class_counts / (class_counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_norm = entropy / np.log(NUM_COCO_CLASSES + 1e-10)

        spatial = np.concatenate([mean_vals, std_vals, [count_norm, entropy_norm]])
    else:
        spatial = np.zeros(10, dtype=np.float32)

    return np.concatenate([presence, max_conf, spatial])  # (170,)


def raw_vector_to_embedding(raw: np.ndarray) -> np.ndarray:
    """
    Project a 170-dim raw feature vector to 256-dim embedding via ReLU projection.

    Parameters
    ----------
    raw : np.ndarray of shape (170,)

    Returns
    -------
    np.ndarray of shape (256,) — L2-normalised
    """
    emb = np.dot(raw, _PROJECTION_MATRIX)  # (256,)
    emb = np.maximum(emb, 0)               # ReLU
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8 detection
# ──────────────────────────────────────────────────────────────────────────────

def build_detector():
    """
    Load YOLOv8n model.
    Uses local yolov8n.pt in project root if available.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(project_root, "models", ".ultralytics"))
    from ultralytics import YOLO
    logger.info("Loading YOLOv8n model…")
    model_path = os.path.join(project_root, "yolov8n.pt")
    if os.path.isfile(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO("yolov8n.pt")
    logger.info("YOLOv8n loaded.")
    return model


def detect_objects_single(
    image_path: str,
    model,
    conf_thresh: float = CONFIDENCE_THRESH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run YOLOv8 on a single image and return raw detection arrays.

    Parameters
    ----------
    image_path  : absolute path to the image file
    model       : YOLO model instance
    conf_thresh : minimum confidence to keep

    Returns
    -------
    class_ids   : int array (N,)
    confidences : float array (N,)
    bboxes      : float array (N, 4) — normalised [cx, cy, w, h]
    """
    try:
        results = model(image_path, conf=conf_thresh, verbose=False)
        result  = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )

        boxes = result.boxes
        class_ids   = boxes.cls.cpu().numpy().astype(np.int32)
        confidences = boxes.conf.cpu().numpy().astype(np.float32)

        # xywhn = normalised cx, cy, w, h
        bboxes = boxes.xywhn.cpu().numpy().astype(np.float32)

        # Limit to MAX_DETECTIONS (sorted by confidence desc)
        if len(class_ids) > MAX_DETECTIONS:
            order       = np.argsort(-confidences)[:MAX_DETECTIONS]
            class_ids   = class_ids[order]
            confidences = confidences[order]
            bboxes      = bboxes[order]

        return class_ids, confidences, bboxes

    except Exception as e:
        logger.warning(f"Detection failed for {image_path}: {e}")
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Batch detection pipeline
# ──────────────────────────────────────────────────────────────────────────────

def extract_object_embeddings(
    image_names: List[str],
    images_dir: str,
    model,
    existing_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract 256-dim object embeddings for a list of images.

    Parameters
    ----------
    image_names        : list of image filenames (basename only)
    images_dir         : directory containing image files
    model              : YOLO model instance
    existing_embeddings: previously cached dict (skipped)

    Returns
    -------
    dict mapping image_filename → np.ndarray of shape (256,)
    """
    embeddings = dict(existing_embeddings) if existing_embeddings else {}
    todo = [n for n in image_names if n not in embeddings]

    if not todo:
        logger.info("All object embeddings already cached — skipping detection.")
        return embeddings

    logger.info(f"Running YOLOv8 detection on {len(todo)} images…")

    for i, name in enumerate(todo, 1):
        img_path = os.path.join(images_dir, name)
        class_ids, confidences, bboxes = detect_objects_single(img_path, model)
        raw = _detections_to_raw_vector(class_ids, confidences, bboxes)
        embeddings[name] = raw_vector_to_embedding(raw)

        if i % 500 == 0 or i == len(todo):
            logger.info(f"  Processed {i}/{len(todo)} images…")

    logger.info(f"Object detection complete. Total embeddings: {len(embeddings)}")
    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_object_embeddings(embeddings: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Object embeddings saved to: {path}")


def load_object_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        emb = pickle.load(f)
    logger.info(f"Object embeddings loaded from: {path} ({len(emb)} images)")
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_object_detection(
    images_dir: str,
    models_dir: str,
    image_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    End-to-end object detection + embedding pipeline with caching.

    Parameters
    ----------
    images_dir   : path to Flickr8k Images/ folder
    models_dir   : directory to load/save object_embeddings.pkl
    image_names  : list of filenames to process (None → all images in dir)

    Returns
    -------
    dict mapping image filename → 256-dim embedding vector
    """
    cache_path = os.path.join(models_dir, "object_embeddings.pkl")

    existing: Dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        logger.info("Found existing object embedding cache — loading…")
        existing = load_object_embeddings(cache_path)

    if image_names is None:
        image_names = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        logger.info(f"Found {len(image_names)} images in {images_dir}")

    todo = [n for n in image_names if n not in existing]
    if todo:
        detector = build_detector()
        embeddings = extract_object_embeddings(
            image_names=image_names,
            images_dir=images_dir,
            model=detector,
            existing_embeddings=existing,
        )
    else:
        embeddings = existing
        logger.info("All embeddings already cached — no detection needed.")

    save_object_embeddings(embeddings, cache_path)
    return embeddings


def get_zero_embedding() -> np.ndarray:
    """Return a zero 256-dim embedding (fallback for unknown images)."""
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on Flickr8k")
    parser.add_argument(
        "--images-dir",
        default=os.path.join("data", "Flickr8k", "Images"),
    )
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    embeddings = run_object_detection(
        images_dir=args.images_dir,
        models_dir=args.models_dir,
    )
    print(f"\nDone! Object embeddings for {len(embeddings)} images.")
    sample = next(iter(embeddings))
    print(f"Sample: {sample} → shape {embeddings[sample].shape}")
