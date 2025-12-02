import streamlit as st
import numpy as np
import cv2
import os
import random
from tensorflow.keras.models import load_model

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Oil Spill Forensic Tool",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render fake top bar element (single placement)
# This element visually merges the top of the page with the browser chrome in dark theme.
st.markdown(
    """
    <style>
    .fake-browser-top {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100vw;
        height: 48px;
        background-color: #0b0f14;
        z-index: 99999;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    /* push app content below the fake bar */
    .block-container { margin-top: 48px !important; }
    </style>
    <div class="fake-browser-top"></div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Theme toggle + session
# -------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"  # default; change to "Light" if you prefer

st.sidebar.markdown("## Theme")
theme_choice = st.sidebar.radio(
    "Choose theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1
)
st.session_state.theme = theme_choice

# Safe theme injector (no destructive global resets)
# --- AFTER user picks theme:
def apply_theme_css(theme: str, follow_system: bool = False):
    """
    Safe theme injector. Injects theme CSS AND the fake top bar element
    together to avoid flashes / mismatched bars.
    """
    # theme colors
    if theme.lower() == "dark":
        bg = "#0b0f14"
        panel = "#0f1720"
        card = "#1f2937"
        text = "#ffffff"
        muted = "#a9b1b9"
        severity = "rgba(150,30,30,0.22)"
        accent = "#7c5cff"
        fake_top = "#0b0f14"
        color_rule = "" if follow_system else "color-scheme: dark !important;"
    else:
        bg = "#ffffff"
        panel = "#f7f8fb"
        card = "#f3f4f6"
        text = "#0a0a0a"
        muted = "#555b61"
        severity = "#fff3cd"
        accent = "#2563eb"
        fake_top = "#ffffff"
        color_rule = "" if follow_system else "color-scheme: light !important;"

    css_and_div = f"""
    <style>
    /* Force page-level color-scheme only if not following system */
    html, body {{
        {color_rule}
        forced-color-adjust: none !important;
        background-color: {bg} !important;
    }}

    /* Fake top bar (rendered here to avoid mismatch) */
    .fake-browser-top {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100vw;
        height: 48px;
        background-color: {fake_top} !important;
        z-index: 99999;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }}
    /* push app content below the fake bar */
    .block-container {{ margin-top: 48px !important; }}

    /* Core containers */
    .stApp, .main, .block-container {{
        background-color: {bg} !important;
        color: {text} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"], .stSidebar {{
        background-color: {panel} !important;
        color: {text} !important;
        min-height: 100vh !important;
        box-sizing: border-box !important;
        padding-top: 20px !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {card} !important;
        color: {text} !important;
        border-radius: 12px !important;
        padding: 16px 22px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18) !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text} !important;
        font-weight: 900 !important;
        font-size: 1.25rem !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text} !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
    }}

    /* Severity box */
    .severity-box {{
        background-color: {severity} !important;
        color: {text} !important;
        padding: 14px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
    }}

    /* Image captions and headings */
    .stImage > figcaption, figure figcaption {{
        color: {muted} !important;
    }}
    h1,h2,h3,h4,h5,h6, .stCaption, .stSubheader {{
        color: {text} !important;
    }}

    /* Keep native appearance for inputs/controls but set accent color for radios/checkboxes */
    input, textarea, select, button, .stButton button {{
        appearance: auto !important;
        -webkit-appearance: auto !important;
        -moz-appearance: auto !important;
        color: {text} !important;
        background-color: transparent !important;
        border-color: rgba(0,0,0,0.06) !important;
    }}

    input[type="radio"], input[type="checkbox"] {{
        appearance: auto !important;
        -webkit-appearance: auto !important;
        accent-color: {accent} !important;
    }}

    /* Sidebar: make the specific "Get New Random Image" button look prominent */
    section[data-testid="stSidebar"] .stButton > button {{
        background-color: {accent} !important;
        color: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.22) !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.32) !important;
    }}
    section[data-testid="stSidebar"] .stButton > button[disabled] {{
        background-color: #e6e6e6 !important;
        color: #777 !important;
        box-shadow: none !important;
    }}

    /* Scrollbar styling (conservative) */
    ::-webkit-scrollbar {{ width: 12px; height: 12px; }}
    ::-webkit-scrollbar-track {{ background: {panel} !important; }}
    ::-webkit-scrollbar-thumb {{ background-color: rgba(0,0,0,0.25) !important; border-radius: 10px; border: 3px solid {panel} !important; }}

    /* DO NOT include global * resets which remove native visuals */
    </style>

    <!-- Fake top bar element (inserted here to ensure color matches theme instantly) -->
    <div class="fake-browser-top"></div>
    """

    st.markdown(css_and_div, unsafe_allow_html=True)


# call after theme choice is set
apply_theme_css(st.session_state.theme)


# -------------------------
# Sidebar controls (data source etc.)
# -------------------------
st.sidebar.title("🔧 Forensic Control")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Select Input Source", ["📂 Upload File", "🎲 Random Demo Data"])
input_image = None
demo_file_name = ""

if data_source == "📂 Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif data_source == "🎲 Random Demo Data":
    test_dir = "data/test/images"
    st.sidebar.info(f"Using random images from '{test_dir}'")
    if not os.path.exists(test_dir):
        st.sidebar.error(f"❌ Directory '{test_dir}' not found!")
    else:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        if not images:
            st.sidebar.error("No images found in the directory!")
        else:
            if 'current_demo_image' not in st.session_state:
                st.session_state.current_demo_image = random.choice(images)
            if st.sidebar.button("🎲 Get New Random Image"):
                st.session_state.current_demo_image = random.choice(images)
            demo_file_name = st.session_state.current_demo_image
            image_path = os.path.join(test_dir, demo_file_name)
            input_image = cv2.imread(image_path)
            if input_image is None:
                st.sidebar.error("Failed to read selected demo image.")
            else:
                st.sidebar.success(f"Loaded: {demo_file_name}")

st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider("Detection Sensitivity", 0.0, 1.0, 0.05, 0.01)

# -------------------------
# Main header
# -------------------------
st.title("🛢️ Oil Spill Forensic System")
st.caption("Satellite SAR Analysis & Environmental Damage Assessment")

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_ai_model():
    return load_model('saved_models/unet_oil_spill.h5')

model = None
try:
    model = load_ai_model()
    st.sidebar.success("AI Model Loaded Successfully")
except Exception as e:
    st.sidebar.error("AI Model failed to load.")
    st.sidebar.error(str(e))
    st.error("Model load error — visual analysis will be unavailable until model loads.")

# -------------------------
# Processing & display
# -------------------------
if input_image is not None:
    # ensure RGB
    try:
        original_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    except Exception:
        original_img = input_image.copy()

    img_resized = cv2.resize(original_img, (256, 256))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    if model is not None:
        try:
            raw_pred = model.predict(input_data)[0]
            if raw_pred.ndim == 3 and raw_pred.shape[2] == 1:
                raw_pred_2d = np.squeeze(raw_pred, axis=-1)
            elif raw_pred.ndim == 2:
                raw_pred_2d = raw_pred
            else:
                raw_pred_2d = np.mean(raw_pred, axis=-1)

            mask_chan = (raw_pred_2d > confidence_threshold).astype(np.uint8)
            mask_vis = np.stack([mask_chan * 255] * 3, axis=-1).astype(np.uint8)
            mask_red = np.zeros_like(img_resized, dtype=np.uint8)
            mask_red[:, :, 0] = mask_chan * 255
            overlay = cv2.addWeighted(img_resized.astype(np.uint8), 0.7, mask_red, 0.3, 0)

            # metrics
            oil_pixels = int(np.count_nonzero(mask_chan))
            area_sq_km = (oil_pixels * 100) / 1_000_000
            model_confidence = float(np.max(raw_pred_2d)) * 100

            # display
            st.markdown("---")

            # severity
            if area_sq_km > 1.0:
                st.markdown(f'<div class="severity-box">🚨 <strong>CRITICAL SEVERITY</strong> — Cleanup required ({area_sq_km:.2f} km²)</div>', unsafe_allow_html=True)
            elif area_sq_km > 0.1:
                st.markdown(f'<div class="severity-box">⚠️ <strong>HIGH SEVERITY</strong> — Booms advised ({area_sq_km:.2f} km²)</div>', unsafe_allow_html=True)
            elif area_sq_km > 0.0:
                st.markdown(f'<div class="severity-box">ℹ️ <strong>MODERATE SEVERITY</strong> — Minor leakage ({area_sq_km:.4f} km²)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="severity-box">✅ <strong>NO SPILL DETECTED</strong> — Area is clear</div>', unsafe_allow_html=True)

            # metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Detected Oil Pixels", f"{oil_pixels}")
            m2.metric("Est. Spill Area", f"{area_sq_km:.4f} km²")
            m3.metric("Model Confidence", f"{model_confidence:.1f}%")

            # visual analysis
            st.markdown("### 🛰️ Visual Analysis")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔍 Forensic Overlay")
                st.image(overlay, use_container_width=True, caption=f"Analysis of {demo_file_name if demo_file_name else 'Uploaded Image'}")

            with col2:
                st.subheader("🧠 AI Mask Analysis")
                st.image(mask_vis, use_container_width=True, caption="Binary Segmentation Mask")

            with col3:
                st.subheader("📷 Original Input")
                st.image(img_resized, use_container_width=True, caption="Raw Input (256×256)")

        except Exception as pred_err:
            st.error("Prediction failed — see sidebar for details.")
            st.sidebar.error(f"Prediction error: {pred_err}")
            st.warning("Unable to render visual analysis due to prediction error.")
    else:
        st.warning("Cannot run prediction: model is not loaded. Check the sidebar for details.")
else:
    st.info("👈 Select 'Random Demo Data' or Upload an image to start.")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 50px;">
            <h4>Waiting for input...</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
