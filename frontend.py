"""
frontend.py
===========
Optimized Streamlit frontend for the Image Caption Generator.
Calls the Flask backend API instead of loading ML models directly.

Features:
  - Tab-based layout: Caption · Gallery · About
  - Backend health indicator in sidebar
  - Caption history with session state
  - Copy-to-clipboard and download buttons
  - Sample images gallery from the dataset
  - Processing time display
  - Responsive dark glassmorphism theme
"""

import os
import io
import time
import base64
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
API_TIMEOUT = 120  # seconds for caption requests
HEALTH_TIMEOUT = 3  # seconds for health check

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
# Custom CSS — dark glassmorphism theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Font ─────────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ──────────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Glass card ──────────────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.15);
    }

    /* ── Caption display ─────────────────────────────────────────────────── */
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

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s ease, transform 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* ── Progress bar ────────────────────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border-radius: 999px;
    }

    /* ── Status indicators ───────────────────────────────────────────────── */
    .status-ok   { color: #34d399; font-weight: 600; }
    .status-warn { color: #fbbf24; font-weight: 600; }
    .status-err  { color: #f87171; font-weight: 600; }

    /* ── History item ────────────────────────────────────────────────────── */
    .history-item {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: background 0.2s ease;
    }
    .history-item:hover { background: rgba(255,255,255,0.07); }
    .history-time {
        font-size: 0.7rem;
        color: #64748b;
    }
    .history-caption {
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-top: 2px;
    }

    /* ── Gallery card ────────────────────────────────────────────────────── */
    .gallery-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 16px;
    }
    .gallery-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.2);
    }

    /* ── Tabs ─────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.2) !important;
        color: #a5b4fc !important;
    }

    /* ── Copy button animation ───────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.4s ease-out; }

    /* ── Responsive adjustments ──────────────────────────────────────────── */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.6rem; }
        .glass-card { padding: 16px; }
        .caption-box { padding: 14px 16px; font-size: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

if "caption_history" not in st.session_state:
    st.session_state.caption_history = []


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_backend_health() -> dict | None:
    """Ping /api/health. Returns dict on success, None on failure."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_info() -> dict | None:
    """Fetch model info (cached 5 min)."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/model/info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_sample_images_list() -> list[str]:
    """Fetch sample image filenames (cached 2 min)."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/sample-images", timeout=10)
        r.raise_for_status()
        return r.json().get("images", [])
    except Exception:
        return []


def caption_image(image_bytes: bytes, filename: str, method: str = "beam",
                  beam_width: int = 3) -> dict | None:
    """POST image to /api/caption and return response dict."""
    try:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data = {"method": method, "beam_width": str(beam_width)}
        r = requests.post(
            f"{BACKEND_URL}/api/caption",
            files=files,
            data=data,
            timeout=API_TIMEOUT,
        )
        if r.status_code >= 400:
            try:
                err_data = r.json()
                return {"error": err_data.get("error", f"Server error ({r.status_code})")}
            except Exception:
                return {"error": f"Server error ({r.status_code}): {r.text[:200]}"}
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the server running? Start with: python run_backend.py"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The image may be too large or the server is overloaded."}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def caption_image_url(url: str, method: str = "beam",
                      beam_width: int = 3) -> dict | None:
    """POST image URL to /api/caption/url and return response dict."""
    try:
        payload = {"url": url, "method": method, "beam_width": beam_width}
        r = requests.post(
            f"{BACKEND_URL}/api/caption/url",
            json=payload,
            timeout=API_TIMEOUT,
        )
        if r.status_code >= 400:
            try:
                err_data = r.json()
                return {"error": err_data.get("error", f"Server error ({r.status_code})")}
            except Exception:
                return {"error": f"Server error ({r.status_code}): {r.text[:200]}"}
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the server running? Start with: python run_backend.py"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def get_sample_image_url(filename: str) -> str:
    """Build URL for a sample image."""
    return f"{BACKEND_URL}/api/sample-images/{filename}"


def add_to_history(caption: str, method: str, proc_time: float,
                   thumbnail_b64: str | None = None):
    """Append a caption result to session-state history."""
    entry = {
        "caption": caption,
        "method": method,
        "time": datetime.now().strftime("%H:%M:%S"),
        "proc_time": proc_time,
        "thumbnail": thumbnail_b64,
    }
    st.session_state.caption_history.insert(0, entry)
    # Keep at most 20 entries
    st.session_state.caption_history = st.session_state.caption_history[:20]


def pil_to_thumbnail_b64(pil_img: Image.Image, max_size: int = 80) -> str:
    """Create a tiny base64 thumbnail for history display."""
    img = pil_img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        # Logo / branding
        st.markdown(
            "<div style='text-align:center; padding: 10px 0 20px;'>"
            "<span style='font-size:2.5rem;'>🔍</span><br>"
            "<span style='color:#a5b4fc;font-weight:700;font-size:1.1rem;'>"
            "Caption Generator</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Backend status ────────────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>🔗 Backend Status</div>",
            unsafe_allow_html=True,
        )
        health = check_backend_health()
        if health:
            models_ok = (
                health.get("models_loaded", False)
                or health.get("status") == "healthy"
                or (isinstance(health.get("models"), dict) and all(health.get("models", {}).values()))
            )
            if models_ok:
                st.markdown(
                    "<span class='status-ok'>● Connected — Models loaded</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='status-warn'>● Connected — Models loading…</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='status-err'>● Backend unavailable</span><br>"
                "<small style='color:#94a3b8'>Start server: "
                "<code>python run_backend.py</code></small>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>⚙️ Settings</div>",
            unsafe_allow_html=True,
        )
        method = st.selectbox(
            "Decoding method",
            options=["beam", "greedy"],
            index=0,
            help="Beam search explores multiple candidates; greedy is faster.",
        )
        beam_width = st.slider(
            "Beam width",
            min_value=1,
            max_value=5,
            value=3,
            disabled=(method == "greedy"),
            help="Number of parallel beams (only for beam search)",
        )

        # ── Caption history ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>📜 Caption History</div>",
            unsafe_allow_html=True,
        )

        history = st.session_state.caption_history
        if history:
            for i, entry in enumerate(history[:10]):
                thumb_html = ""
                if entry.get("thumbnail"):
                    thumb_html = (
                        f"<img src='data:image/jpeg;base64,{entry['thumbnail']}' "
                        f"style='width:36px;height:36px;border-radius:6px;"
                        f"object-fit:cover;float:left;margin-right:10px;'/>"
                    )
                st.markdown(
                    f"<div class='history-item'>"
                    f"{thumb_html}"
                    f"<div class='history-time'>{entry['time']} · "
                    f"{entry['method']} · {entry['proc_time']:.1f}s</div>"
                    f"<div class='history-caption'>{entry['caption']}</div>"
                    f"<div style='clear:both'></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear History", key="clear_hist"):
                st.session_state.caption_history = []
                st.rerun()
        else:
            st.caption("No captions generated yet.")

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
# Tab 1 — Caption
# ──────────────────────────────────────────────────────────────────────────────

def render_caption_tab(method: str, beam_width: int):
    st.markdown(
        "<div class='section-header'>📤 Upload Image</div>",
        unsafe_allow_html=True,
    )

    upload_col, url_col = st.columns([1, 1], gap="large")

    with upload_col:
        uploaded = st.file_uploader(
            "Drop a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with url_col:
        image_url = st.text_input(
            "Or paste an image URL",
            placeholder="https://example.com/photo.jpg",
            label_visibility="collapsed",
        )

    # Determine source
    pil_img = None
    image_bytes = None
    source_name = ""

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        source_name = uploaded.name
    elif image_url.strip():
        try:
            resp = requests.get(image_url.strip(), timeout=15, stream=True)
            resp.raise_for_status()
            image_bytes = resp.content
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            source_name = image_url.strip().split("/")[-1][:40]
        except Exception as e:
            st.error(f"Failed to fetch image from URL: {e}")

    if pil_img is None:
        st.markdown(
            "<div class='glass-card' style='text-align:center;padding:48px;'>"
            "<span style='font-size:3rem;'>🌄</span><br><br>"
            "<span style='color:#64748b;font-size:1rem;'>"
            "Upload an image or paste a URL to generate a caption</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Display + Generate ────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown(
            "<div class='section-header'>🖼 Input Image</div>",
            unsafe_allow_html=True,
        )
        st.image(pil_img, use_container_width=True)
        st.caption(
            f"Size: {pil_img.width} × {pil_img.height} px  ·  {source_name}"
        )

    with col_result:
        st.markdown(
            "<div class='section-header'>✨ Generated Caption</div>",
            unsafe_allow_html=True,
        )

        generate_btn = st.button("🚀 Generate Caption", key="gen_btn")

        if generate_btn:
            # Check backend availability first
            health = check_backend_health()
            if not health:
                st.error(
                    "⚠️ Backend is not available. Please start the server:\n\n"
                    "```\npython run_backend.py\n```"
                )
                return

            # Progress animation
            progress_bar = st.progress(0, text="🔄 Sending image to backend…")
            progress_bar.progress(15, text="🔄 Extracting features…")

            # Call the API
            if uploaded is not None:
                result = caption_image(image_bytes, source_name, method, beam_width)
            else:
                result = caption_image_url(image_url.strip(), method, beam_width)

            progress_bar.progress(90, text="🔄 Generating caption…")

            if result and "error" in result:
                progress_bar.empty()
                st.error(f"❌ {result['error']}")
                return

            if result is None:
                progress_bar.empty()
                st.error("❌ No response from backend.")
                return

            progress_bar.progress(100, text="✅ Done!")
            time.sleep(0.3)
            progress_bar.empty()

            # Extract results
            caption = result.get("caption", "")
            greedy_cap = result.get("greedy_caption", "")
            beam_cap = result.get("beam_caption", "")
            objects = result.get("objects", [])
            proc_time = result.get("processing_time", 0.0)
            used_method = result.get("method", method)

            primary = beam_cap if used_method == "beam" else greedy_cap
            if not primary:
                primary = caption  # fallback

            # Display primary caption
            st.markdown(
                f"<div class='caption-box fade-in'>"
                f"<span style='font-size:0.75rem;color:#6366f1;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.1em;'>"
                f"{'Beam Search' if used_method == 'beam' else 'Greedy'} Caption</span>"
                f"<br><br>🗨 {primary.capitalize()}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Processing time
            st.markdown(
                f"<small style='color:#64748b;'>⏱ Processing time: "
                f"<strong>{proc_time:.2f}s</strong></small>",
                unsafe_allow_html=True,
            )

            # Action buttons: copy + download
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                # Copy button via st.code fallback (Streamlit has built-in copy on code blocks)
                st.code(primary.capitalize(), language=None)
            with btn_col2:
                # Download annotated result as text
                result_text = (
                    f"Image Caption Generator — Result\n"
                    f"{'=' * 40}\n"
                    f"Method: {used_method}\n"
                    f"Caption: {primary.capitalize()}\n"
                )
                if greedy_cap and used_method == "beam":
                    result_text += f"Greedy:  {greedy_cap.capitalize()}\n"
                if objects:
                    result_text += f"\nDetected Objects:\n"
                    for o in objects:
                        result_text += f"  - {o['class']} ({o['confidence']:.0%})\n"
                result_text += f"\nProcessing Time: {proc_time:.2f}s\n"

                st.download_button(
                    "📥 Download Result",
                    data=result_text,
                    file_name="caption_result.txt",
                    mime="text/plain",
                    key="dl_result",
                )

            # Show greedy alongside beam
            if used_method == "beam" and greedy_cap:
                with st.expander("📝 Also show Greedy caption"):
                    st.markdown(
                        f"<div class='caption-box' style='font-size:1rem;'>"
                        f"🗨 {greedy_cap.capitalize()}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Detected objects
            if objects:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-header'>🎯 Detected Objects (YOLOv8)</div>",
                    unsafe_allow_html=True,
                )
                badge_html = "".join(
                    f"<span class='obj-badge'>{o['class']} "
                    f"<span style='opacity:0.7'>{o['confidence']:.0%}</span></span>"
                    for o in objects
                )
                st.markdown(badge_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for obj in objects[:8]:
                    st.progress(
                        min(obj["confidence"], 1.0),
                        text=f"{obj['class']:20s}  {obj['confidence']:.1%}",
                    )

            # Save to history
            thumb = pil_to_thumbnail_b64(pil_img)
            add_to_history(primary.capitalize(), used_method, proc_time, thumb)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Gallery (sample images)
# ──────────────────────────────────────────────────────────────────────────────

def render_gallery_tab(method: str, beam_width: int):
    st.markdown(
        "<div class='section-header'>🖼 Sample Images from Dataset</div>",
        unsafe_allow_html=True,
    )

    sample_list = fetch_sample_images_list()

    if not sample_list:
        st.info(
            "No sample images available. Make sure the backend is running and "
            "has sample images in `data/Flickr8k/Images/`."
        )
        return

    st.markdown(
        f"<small style='color:#64748b;'>Showing {min(len(sample_list), 12)} "
        f"of {len(sample_list)} available sample images</small>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Try Sample button — pick a random sample
    if st.button("🎲 Try a Random Sample", key="random_sample"):
        import random
        pick = random.choice(sample_list)
        st.session_state["selected_sample"] = pick

    # Display grid of samples
    display_list = sample_list[:12]
    cols = st.columns(4)

    for idx, fname in enumerate(display_list):
        col = cols[idx % 4]
        img_url = get_sample_image_url(fname)
        with col:
            st.image(img_url, use_container_width=True)
            if st.button(f"Caption", key=f"sample_{idx}"):
                st.session_state["selected_sample"] = fname

    # If a sample is selected, caption it
    selected = st.session_state.get("selected_sample")
    if selected:
        st.markdown("---")
        st.markdown(
            f"<div class='section-header'>✨ Captioning: {selected}</div>",
            unsafe_allow_html=True,
        )

        img_url = get_sample_image_url(selected)

        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            st.image(img_url, use_container_width=True)

        with col_res:
            with st.spinner("Generating caption…"):
                result = caption_image_url(img_url, method, beam_width)

            if result and "error" not in result:
                caption = result.get("caption", "")
                primary = result.get("beam_caption") or result.get("greedy_caption") or caption
                proc_time = result.get("processing_time", 0.0)
                objects = result.get("objects", [])

                st.markdown(
                    f"<div class='caption-box fade-in'>"
                    f"🗨 {primary.capitalize()}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<small style='color:#64748b;'>⏱ {proc_time:.2f}s</small>",
                    unsafe_allow_html=True,
                )

                if objects:
                    badge_html = "".join(
                        f"<span class='obj-badge'>{o['class']} "
                        f"<span style='opacity:0.7'>{o['confidence']:.0%}</span></span>"
                        for o in objects
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)

                # Add to history
                add_to_history(primary.capitalize(), method, proc_time)
            elif result and "error" in result:
                st.error(result["error"])
            else:
                st.error("No response from backend.")

        # Clear selection button
        if st.button("✕ Clear Selection", key="clear_sample"):
            del st.session_state["selected_sample"]
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — About
# ──────────────────────────────────────────────────────────────────────────────

def render_about_tab():
    model_info = fetch_model_info()

    # ── Architecture overview ─────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>🏗 Architecture Overview</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='section-header'>Vision Branch</div>"
            "<b>InceptionV3</b><br>"
            "<small style='color:#94a3b8'>"
            "ImageNet pretrained · Frozen weights · "
            "GlobalAvgPool → 2048-dim feature vector"
            "</small></div>",
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
            "</small></div>",
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
            "</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Model info from API ───────────────────────────────────────────────
    if model_info:
        st.markdown(
            "<div class='section-header'>📊 Model Information</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            vs = model_info.get("vocab_size", "—")
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='metric-value'>{vs:,}</div>"
                f"<div class='metric-label'>Vocabulary Size</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m2:
            ml = model_info.get("max_length", "—")
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='metric-value'>{ml}</div>"
                f"<div class='metric-label'>Max Sequence Length</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m3:
            arch = model_info.get("architecture", {})
            params = arch.get("total_params", "—")
            if isinstance(params, (int, float)):
                params_display = f"{params / 1e6:.1f}M"
            else:
                params_display = str(params)
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='metric-value'>{params_display}</div>"
                f"<div class='metric-label'>Model Parameters</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Evaluation results
        eval_results = model_info.get("eval_results") or model_info.get("evaluation_results", {})
        if eval_results:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'>📈 Evaluation Metrics</div>",
                unsafe_allow_html=True,
            )

            e1, e2, e3, e4 = st.columns(4)
            metrics = [
                ("BLEU-1", eval_results.get("bleu1", 0), "#818cf8"),
                ("BLEU-2", eval_results.get("bleu2", 0), "#818cf8"),
                ("METEOR", eval_results.get("meteor", 0), "#34d399"),
                ("Evaluated", eval_results.get("n_evaluated", "—"), "#fb7185"),
            ]
            for col, (label, value, color) in zip([e1, e2, e3, e4], metrics):
                with col:
                    display = f"{value:.4f}" if isinstance(value, float) else f"{value:,}"
                    st.markdown(
                        f"<div class='metric-tile'>"
                        f"<div class='metric-value' style='color:{color};'>"
                        f"{display}</div>"
                        f"<div class='metric-label'>{label}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info(
            "Model information unavailable. "
            "Make sure the backend is running."
        )

    st.markdown("---")

    # ── Dataset & Training details ────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📚 Dataset & Training Details</div>",
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='section-header'>Flickr8k Dataset</div>",
            unsafe_allow_html=True,
        )
        items = [
            ("Total images", "8,091"),
            ("Captions", "40,455 (5 per image)"),
            ("Training split", "6,000 images (60%)"),
            ("Validation", "1,000 images (20%)"),
            ("Test split", "1,619 images evaluated"),
            ("Vocabulary", "~7,000 unique words"),
            ("Max seq length", "35 tokens"),
        ]
        for k, v in items:
            st.markdown(f"- **{k}**: {v}")
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='section-header'>Training Configuration</div>",
            unsafe_allow_html=True,
        )
        items = [
            ("Optimizer", "Adam (lr=0.001)"),
            ("Loss", "Categorical Cross-Entropy"),
            ("Batch size", "64"),
            ("Max epochs", "15"),
            ("Early stopping", "patience=5"),
            ("LR scheduling", "ReduceLROnPlateau ×0.5"),
            ("Hardware", "CPU"),
        ]
        for k, v in items:
            st.markdown(f"- **{k}**: {v}")
        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    method, beam_width = render_sidebar()

    # Hero
    st.markdown(
        "<div class='hero-title'>🖼 Image Caption Generator</div>"
        "<div class='hero-subtitle'>"
        "Multi-modal deep learning with InceptionV3 + YOLOv8 + CNN-LSTM"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_caption, tab_gallery, tab_about = st.tabs([
        "📸 Caption", "🖼 Gallery", "ℹ️ About"
    ])

    with tab_caption:
        render_caption_tab(method, beam_width)

    with tab_gallery:
        render_gallery_tab(method, beam_width)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
