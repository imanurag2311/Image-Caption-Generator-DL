"""
app.py
======
Streamlit web interface for the Image Caption Generator.

Features:
  - Upload any JPG/PNG image
  - Generate captions using greedy or beam search
  - Display detected COCO objects with confidence bars
  - Show model architecture summary
  - Display evaluation metrics (if available)
  - Dark-themed, responsive UI with animated elements
"""

import os
import sys
import json
import pickle
import logging

import numpy as np
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Font ─────────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background gradient ─────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Glassmorphism cards ─────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    /* ── Caption display box ─────────────────────────────────────────────── */
    .caption-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.40);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        font-size: 1.1rem;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.6;
        letter-spacing: 0.015em;
    }

    /* ── Section headers ─────────────────────────────────────────────────── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(165,180,252,0.2);
    }

    /* ── Metric tile ─────────────────────────────────────────────────────── */
    .metric-tile {
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Object badge ────────────────────────────────────────────────────── */
    .obj-badge {
        display: inline-block;
        background: rgba(168,85,247,0.15);
        border: 1px solid rgba(168,85,247,0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #d8b4fe;
        margin: 3px;
        font-weight: 500;
    }

    /* ── Hero title ──────────────────────────────────────────────────────── */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }

    /* ── Progress bar overrides ──────────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border-radius: 999px;
    }

    /* ── Button overrides ────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s ease, transform 0.1s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* ── Tag chip ────────────────────────────────────────────────────────── */
    .tag-chip {
        display: inline-block;
        background: rgba(251,113,133,0.12);
        border: 1px solid rgba(251,113,133,0.30);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        color: #fda4af;
        margin: 2px;
    }

    /* ── Status ──────────────────────────────────────────────────────────── */
    .status-ok {
        color: #34d399;
        font-weight: 600;
    }
    .status-warn {
        color: #fbbf24;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load tokenizer, max_length, and evaluation results from disk."""
    artifacts = {"ready": False, "error": None}

    try:
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        ml_path  = os.path.join(MODELS_DIR, "max_length.pkl")
        if not os.path.exists(tok_path):
            artifacts["error"] = "tokenizer.pkl not found — run: python main.py --preprocess"
            return artifacts

        with open(tok_path, "rb") as f:
            artifacts["tokenizer"] = pickle.load(f)
        with open(ml_path, "rb") as f:
            artifacts["max_length"] = pickle.load(f)

        artifacts["vocab_size"] = len(artifacts["tokenizer"].word_index) + 1

        # Evaluation results (optional)
        eval_path = os.path.join(MODELS_DIR, "evaluation_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                artifacts["eval_results"] = json.load(f)

        # Check for trained model
        model_path = os.path.join(MODELS_DIR, "final_model.keras")
        artifacts["model_available"] = os.path.exists(model_path)
        artifacts["ready"] = True

    except Exception as e:
        artifacts["error"] = str(e)

    return artifacts


@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Load the trained Keras captioning model."""
    from tensorflow.keras.models import load_model
    model_path = os.path.join(MODELS_DIR, "final_model.keras")
    return load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    """Load InceptionV3 feature extractor."""
    from src.feature_extraction import build_feature_extractor
    return build_feature_extractor()


@st.cache_resource(show_spinner=False)
def load_yolo():
    """Load YOLOv8 detector."""
    from src.object_detection import build_detector
    return build_detector()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run inference on an uploaded image
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(pil_image: Image.Image, method: str, beam_width: int, artifacts: dict):
    """
    Extract features + detect objects + generate caption for a PIL Image.

    Returns dict with keys: greedy_caption / beam_caption, objects, img_feat, obj_emb
    """
    import tempfile, io
    from src.feature_extraction import load_and_preprocess_image
    from src.object_detection import (
        detect_objects_single,
        _detections_to_raw_vector,
        raw_vector_to_embedding,
        COCO_CLASSES,
    )
    from src.predict import greedy_predict, beam_search_predict

    # Save PIL image to temp file (YOLO needs file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path, format="JPEG")

    try:
        feature_extractor = load_feature_extractor()
        yolo_model        = load_yolo()
        model             = load_caption_model()
        tokenizer         = artifacts["tokenizer"]
        max_length        = artifacts["max_length"]

        # Image features
        arr      = load_and_preprocess_image(tmp_path)
        img_feat = feature_extractor.predict(arr[np.newaxis], verbose=0)[0]

        # Object detection
        cls_ids, confs, bboxes = detect_objects_single(tmp_path, yolo_model)
        raw     = _detections_to_raw_vector(cls_ids, confs, bboxes)
        obj_emb = raw_vector_to_embedding(raw)

        # Detected objects list
        objects = []
        for cid, conf in zip(cls_ids, confs):
            if 0 <= cid < len(COCO_CLASSES):
                objects.append({"class": COCO_CLASSES[cid], "confidence": float(conf)})
        objects = sorted(objects, key=lambda x: -x["confidence"])[:15]

        # Generate captions
        greedy_cap = greedy_predict(img_feat, obj_emb, tokenizer, model, max_length)
        if method == "beam":
            beam_cap = beam_search_predict(
                img_feat, obj_emb, tokenizer, model, max_length, beam_width
            )
        else:
            beam_cap = greedy_cap

        return {
            "greedy_caption": greedy_cap,
            "beam_caption":   beam_cap,
            "objects":        objects,
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(artifacts: dict) -> tuple:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding: 10px 0 20px;'>"
            "<span style='font-size:2.5rem;'>🔍</span><br>"
            "<span style='color:#a5b4fc;font-weight:700;font-size:1.1rem;'>"
            "Caption Generator</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<div class='section-header'>⚙️ Settings</div>", unsafe_allow_html=True)

        method = st.selectbox(
            "Decoding method",
            options=["beam", "greedy"],
            index=0,
            help="Beam search explores multiple candidates; greedy is faster.",
        )
        beam_width = st.slider(
            "Beam width",
            min_value=1, max_value=5, value=3,
            disabled=(method == "greedy"),
            help="Number of parallel beams (only for beam search)",
        )

        st.markdown("---")
        st.markdown("<div class='section-header'>📊 Model Status</div>", unsafe_allow_html=True)

        if artifacts.get("ready"):
            vs = artifacts.get("vocab_size", "—")
            ml = artifacts.get("max_length", "—")
            st.markdown(f"<span class='status-ok'>● Tokenizer loaded</span>", unsafe_allow_html=True)
            st.markdown(f"<small style='color:#94a3b8'>Vocab: {vs:,} words | Max len: {ml}</small>", unsafe_allow_html=True)

            if artifacts.get("model_available"):
                st.markdown("<span class='status-ok'>● Trained model found</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-warn'>⚠ Model not trained yet</span>", unsafe_allow_html=True)
                st.caption("Run: `python main.py --train`")
        else:
            st.markdown(
                f"<span class='status-warn'>⚠ Not ready</span><br>"
                f"<small style='color:#94a3b8'>{artifacts.get('error','')}</small>",
                unsafe_allow_html=True,
            )

        # Evaluation metrics in sidebar
        if "eval_results" in artifacts:
            st.markdown("---")
            st.markdown("<div class='section-header'>📈 Eval Metrics</div>", unsafe_allow_html=True)
            er = artifacts["eval_results"]
            b1 = er.get('bleu1', 0)
            b2 = er.get('bleu2', 0)
            mt = er.get('meteor', 0)
            n  = er.get('n_evaluated', '?')
            st.markdown(
                f"""
                <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                            border-radius:10px;padding:12px 16px;'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='color:#94a3b8;font-size:0.78rem;'>BLEU-1</span>
                    <span style='color:#818cf8;font-weight:700;font-size:0.95rem;'>{b1:.4f}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='color:#94a3b8;font-size:0.78rem;'>BLEU-2</span>
                    <span style='color:#818cf8;font-weight:700;font-size:0.95rem;'>{b2:.4f}</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;'>
                    <span style='color:#94a3b8;font-size:0.78rem;'>METEOR</span>
                    <span style='color:#34d399;font-weight:700;font-size:0.95rem;'>{mt:.4f}</span>
                  </div>
                </div>
                <div style='color:#475569;font-size:0.72rem;margin-top:6px;text-align:right;'>
                  {n:,} test images · beam search
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;'>"
            "<small style='color:#475569;'>"
            "InceptionV3 + YOLOv8 + CNN-LSTM<br>"
            "Flickr8k Dataset · TensorFlow 2.17</small></div>",
            unsafe_allow_html=True,
        )

    return method, beam_width


# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

def render_main(artifacts: dict, method: str, beam_width: int) -> None:
    # Hero section
    st.markdown(
        "<div class='hero-title'>🖼 Image Caption Generator</div>"
        "<div class='hero-subtitle'>"
        "Multi-modal deep learning with InceptionV3 + YOLOv8 + CNN-LSTM"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture overview ─────────────────────────────────────────────────
    with st.expander("🏗 Architecture Overview", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='section-header'>Vision Branch</div>"
                "<b>InceptionV3</b><br>"
                "<small style='color:#94a3b8'>"
                "ImageNet pretrained · Frozen weights · "
                "GlobalAvgPool → 2048-dim feature vector"
                "</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='section-header'>Object Branch</div>"
                "<b>YOLOv8-nano</b><br>"
                "<small style='color:#94a3b8'>"
                "80 COCO classes · conf ≥ 0.30 · "
                "Random projection → 256-dim embedding"
                "</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='section-header'>Language Branch</div>"
                "<b>CNN-LSTM Decoder</b><br>"
                "<small style='color:#94a3b8'>"
                "Embedding(256) → LSTM(512) × 2 · "
                "Fusion via Add + Dense → Softmax"
                "</small>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📤 Upload Image</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        # Placeholder hint
        st.markdown(
            "<div class='glass-card' style='text-align:center;padding:48px;'>"
            "<span style='font-size:3rem;'>🌄</span><br><br>"
            "<span style='color:#64748b;font-size:1rem;'>"
            "Upload an image above to generate a caption</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Display + inference ───────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    pil_img = Image.open(uploaded).convert("RGB")

    with col_img:
        st.markdown("<div class='section-header'>🖼 Input Image</div>", unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.caption(
            f"Size: {pil_img.width} × {pil_img.height} px | "
            f"Format: {uploaded.type.split('/')[-1].upper()}"
        )

    with col_result:
        st.markdown("<div class='section-header'>✨ Generated Captions</div>", unsafe_allow_html=True)

        if not artifacts.get("model_available"):
            st.warning(
                "⚠️ Trained model not found.\n\n"
                "Run the full pipeline first:\n"
                "```\npython main.py --all\n```"
            )
            return

        generate_btn = st.button("🚀 Generate Caption", key="gen_btn")

        if generate_btn:
            with st.spinner("Extracting features and generating caption…"):
                try:
                    results = run_inference(pil_img, method, beam_width, artifacts)

                    # Primary caption (beam or greedy)
                    primary = results["beam_caption"] if method == "beam" else results["greedy_caption"]

                    st.markdown(
                        f"<div class='caption-box'>"
                        f"<span style='font-size:0.75rem;color:#6366f1;font-weight:600;"
                        f"text-transform:uppercase;letter-spacing:0.1em;'>"
                        f"{'Beam Search' if method == 'beam' else 'Greedy'} Caption</span><br><br>"
                        f"🗨 {primary.capitalize()}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Show greedy alongside beam for comparison
                    if method == "beam":
                        with st.expander("📝 Also show Greedy caption"):
                            st.markdown(
                                f"<div class='caption-box' style='font-size:1rem;'>"
                                f"🗨 {results['greedy_caption'].capitalize()}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    # Detected objects
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='section-header'>🎯 Detected Objects (YOLOv8)</div>",
                        unsafe_allow_html=True,
                    )

                    objs = results["objects"]
                    if objs:
                        badge_html = "".join(
                            f"<span class='obj-badge'>{o['class']} "
                            f"<span style='opacity:0.7'>{o['confidence']:.0%}</span></span>"
                            for o in objs
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        for obj in objs[:8]:
                            st.progress(
                                obj["confidence"],
                                text=f"{obj['class']:20s}  {obj['confidence']:.1%}",
                            )
                    else:
                        st.info("No objects detected above confidence threshold.")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.exception(e)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & training info tab
# ──────────────────────────────────────────────────────────────────────────────

def render_info_tab() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("📚 Dataset & Training Details"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Flickr8k Dataset**")
            items = [
                ("Total images",   "8,091"),
                ("Captions",       "40,455 (5 per image)"),
                ("Training split", "6,000 images (60%)"),
                ("Validation",     "1,000 images (20%)"),
                ("Test split",     "1,619 images evaluated"),
                ("Vocabulary",     "7,000 unique words"),
                ("Max seq length", "35 tokens"),
            ]
            for k, v in items:
                st.markdown(f"- **{k}**: {v}")

        with c2:
            st.markdown("**Training Configuration**")
            items = [
                ("Optimizer",       "Adam (lr=0.001)"),
                ("Loss",            "Categorical Cross-Entropy"),
                ("Batch size",      "64"),
                ("Max epochs",      "15"),
                ("Early stopping",  "patience=5"),
                ("LR scheduling",   "ReduceLROnPlateau ×0.5"),
                ("Hardware",        "CPU"),
            ]
            for k, v in items:
                st.markdown(f"- **{k}**: {v}")

        st.markdown("**Model Performance**")
        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU-1", "0.1653")
        col2.metric("BLEU-2", "0.0991")
        col3.metric("METEOR", "0.2678")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    artifacts = load_artifacts()
    method, beam_width = render_sidebar(artifacts)
    render_main(artifacts, method, beam_width)
    render_info_tab()


if __name__ == "__main__":
    main()
