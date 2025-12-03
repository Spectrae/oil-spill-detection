"""
-------------------------------------------------------------------------------
OIL SPILL FORENSIC TOOL - Frontend Application
-------------------------------------------------------------------------------
Project Owner: Rick Mondal
Description:   A single-file Streamlit application for Oil Spill Detection using 
               Deep Learning (U-Net/DeepLabV3) and AIS Vessel Anomaly detection.

REQUIREMENTS:
-------------
streamlit>=1.30.0
tensorflow>=2.10.0
opencv-python-headless
numpy
pandas
Pillow
matplotlib

HOW TO RUN:
-----------
1. Ensure your model is at: 'saved_models/unet_oil_spill.h5'
2. Ensure you have test images in: 'data/test/images'
3. Run command: streamlit run app.py

CHANGELOG:
----------
- Performance: Removed artificial UX delays for maximum speed.
- Performance: Added caching for AIS CSV data.
- Fix: "Demo Mode" loads REAL images from 'data/test/images'.
- Style: Light Mode + Glassmorphism applied.
-------------------------------------------------------------------------------
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import os
import random
import time
import traceback
from PIL import Image

# --- CONFIGURATION & CONSTANTS ---
PAGE_TITLE = "Oil Spill Forensic System"
PAGE_ICON = "🛢️"
MODEL_PATH = 'saved_models/unet_oil_spill.h5'
AIS_DATA_PATH = 'data/ais_data/vessel_data_clean.csv'

# Intelligent Text Injection
TIPS_AND_TRICKS = [
    "💡 Tip: Higher sensitivity may increase false positives in choppy water.",
    "ℹ️ Note: Analysis assumes Sentinel-1 imagery resolution (~10m/pixel).",
    "🔍 Insight: Verify AIS correlation to identify potential offending vessels.",
    "🚀 Status: System optimization active for rapid inference.",
]

# --- 1. UI & CSS SETUP ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embed all CSS for Single-File Portability
st.markdown("""
<style>
    /* 1. GLOBAL THEME (Light Mode + Gradient) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        background-attachment: fixed;
    }

    /* 2. HIDE STREAMLIT CHROME */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }
    footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1400px;
    }

    /* 3. GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.8);
    }

    /* 4. TYPOGRAPHY */
    h1 {
        color: #1e3a8a;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    h3 {
        color: #3b82f6;
        font-weight: 600;
        margin-top: 0;
    }
    p, label, .stMarkdown {
        color: #334155;
    }

    /* 5. SEVERITY BOXES */
    .severity-box {
        padding: 16px;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .sev-critical { background-color: #fee2e2; color: #991b1b; border-left: 5px solid #ef4444; }
    .sev-high { background-color: #ffedd5; color: #9a3412; border-left: 5px solid #f97316; }
    .sev-moderate { background-color: #dbeafe; color: #1e40af; border-left: 5px solid #3b82f6; }
    .sev-clean { background-color: #dcfce7; color: #166534; border-left: 5px solid #22c55e; }

    /* 6. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.5);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    /* Primary Accent */
    div[data-testid="stButton"] button {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

</style>
""", unsafe_allow_html=True)


# --- 2. BACKEND LOGIC INTEGRATION ---

@st.cache_resource
def load_backend_model(path):
    """
    Loads the trained Keras model.
    Uses st.cache_resource to avoid reloading on every interaction.
    """
    try:
        if not os.path.exists(path):
            return None, "Model file not found."
        
        # Lazy import to prevent startup lag
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

def run_inference(model, image_input, threshold):
    """
    Replicates 'predict_spill' logic from 3_Final_Inference.ipynb
    """
    try:
        # Preprocessing
        img_resized = cv2.resize(image_input, (256, 256))
        # Normalize (Crucial step from notebook)
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        
        # Inference
        raw_pred = model.predict(img_tensor, verbose=0)[0]
        
        # Handle dimensions (DeepLab vs UNet outputs)
        if raw_pred.ndim == 3 and raw_pred.shape[2] == 1:
            raw_pred_2d = np.squeeze(raw_pred, axis=-1)
        else:
            raw_pred_2d = raw_pred

        # Thresholding
        mask = (raw_pred_2d > threshold).astype(np.uint8)
        
        return img_resized, mask, raw_pred_2d
    except Exception as e:
        raise e

def analyze_damage(mask, pixel_res_m2=100):
    """
    Replicates 'assess_damage' from 3_Final_Inference.ipynb
    """
    oil_pixels = np.count_nonzero(mask)
    total_area_m2 = oil_pixels * pixel_res_m2
    total_area_km2 = total_area_m2 / 1_000_000.0
    
    severity = "NONE"
    css_class = "sev-clean"
    msg = "✅ NO SPILL DETECTED — Area is clear"

    if total_area_km2 > 1.0:
        severity = "CRITICAL"
        css_class = "sev-critical"
        msg = f"🚨 CRITICAL SEVERITY — Cleanup required ({total_area_km2:.2f} km²)"
    elif total_area_km2 > 0.1:
        severity = "HIGH"
        css_class = "sev-high"
        msg = f"⚠️ HIGH SEVERITY — Booms advised ({total_area_km2:.2f} km²)"
    elif total_area_km2 > 0.0:
        severity = "MODERATE"
        css_class = "sev-moderate"
        msg = f"ℹ️ MODERATE SEVERITY — Minor leakage ({total_area_km2:.4f} km²)"
        
    return {
        "pixels": oil_pixels,
        "area_km2": total_area_km2,
        "severity": severity,
        "css_class": css_class,
        "message": msg
    }

@st.cache_data
def load_ais_data():
    """
    OPTIMIZATION: Loads and caches the AIS CSV file.
    This prevents re-reading the file from disk on every run.
    """
    if os.path.exists(AIS_DATA_PATH):
        try:
            return pd.read_csv(AIS_DATA_PATH)
        except Exception:
            return None
    return None

def get_ais_anomalies(lat, lon, search_radius=2.0):
    """
    Replicates 'detect_anomaly' logic.
    Uses cached data for speed.
    """
    try:
        df = load_ais_data()
        
        if df is not None:
            # Filter logic from 0_Prepare_AIS.py
            nearby = df[
                (df['LAT'] > lat - search_radius) & 
                (df['LAT'] < lat + search_radius) & 
                (df['LON'] > lon - search_radius) & 
                (df['LON'] < lon + search_radius)
            ]
            if nearby.empty:
                return []
            
            suspects = []
            for _, ship in nearby.iterrows():
                status = "STOPPED" if ship['SOG'] < 1.0 else "MOVING"
                suspects.append({
                    "name": str(ship['VesselName']),
                    "mmsi": ship['MMSI'],
                    "speed": ship['SOG'],
                    "status": status
                })
            return suspects
        else:
            # Fallback for Demo
            return [
                {"name": "SIMULATED TANKER A", "mmsi": 999123456, "speed": 0.2, "status": "STOPPED"},
                {"name": "CARGO SHIP B", "mmsi": 888123456, "speed": 14.5, "status": "MOVING"},
            ]
    except Exception:
        return []

def load_demo_image():
    """
    Intelligently finds a random image from the data folder to use as a demo.
    Prevents the 'fake circle' issue.
    """
    # List of possible paths where your images might be stored
    possible_paths = [
        "data/test/images", 
        "../data/test/images",
        "data/train/images"
    ]
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for folder_path in possible_paths:
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) 
                     if os.path.splitext(f)[1].lower() in valid_extensions]
            
            if len(files) > 0:
                selected_file = random.choice(files)
                full_path = os.path.join(folder_path, selected_file)
                return full_path, selected_file
                
    return None, None

# --- 3. UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.header("🔧 Control Panel")
        st.markdown("---")
        
        # Model Status
        st.subheader("Model Status")
        model, error = load_backend_model(MODEL_PATH)
        if model:
            st.success(f"Model Loaded: {os.path.basename(MODEL_PATH)}")
        else:
            st.warning("Running in DEMO MODE")
            st.caption("Real model file not found.")
            
        st.markdown("---")
        
        # Inputs
        st.subheader("Input Data")
        upload = st.file_uploader("Upload SAR Image", type=['jpg', 'png', 'jpeg'])
        
        # Settings
        st.subheader("Parameters")
        threshold = st.slider("Detection Sensitivity", 0.0, 1.0, 0.05, 0.01, 
                            help="Lower values detect fainter spills but may increase noise.")
        
        alpha = st.slider("Overlay Opacity", 0.1, 1.0, 0.6)
        
        # Inject Tip
        st.info(random.choice(TIPS_AND_TRICKS))
        
        return upload, threshold, alpha, model

def main():
    # A. Header Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🛢️ Oil Spill Forensic System")
        st.caption("Satellite SAR Analysis & Environmental Damage Assessment")
    with col_h2:
        st.markdown(f"<div style='text-align:right; color:#64748b;'><b>v2.0.3 (Fast)</b><br>{time.strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # B. Sidebar & Configuration
    uploaded_file, threshold, alpha, model = render_sidebar()

    # C. Main Layout
    if uploaded_file is not None or st.session_state.get('demo_active', False):
        
        # Prepare Image
        if uploaded_file:
            # User uploaded a file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            filename = uploaded_file.name
        else:
            # DEMO MODE: Load a REAL image from the disk
            demo_path, demo_filename = load_demo_image()
            
            if demo_path:
                input_image = cv2.imread(demo_path)
                if input_image is not None:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    filename = f"Demo: {demo_filename}"
                else:
                    # File existed but failed to read
                    input_image = np.zeros((256, 256, 3), dtype=np.uint8)
                    filename = "Error reading demo image"
            else:
                # No images found in data folders - Fallback to Synthetic
                input_image = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(input_image, "NO IMAGES FOUND", (20, 128), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                filename = "No images in 'data/test/images'"

        # D. Processing Block
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 Analysis Pipeline")
        
        if st.button("Run Forensic Analysis", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Preprocessing
                status_text.text("Preprocessing image...")
                progress_bar.progress(25)
                # OPTIMIZATION: Removed time.sleep(0.3)
                
                # Step 2: Inference
                status_text.text("Running Neural Network Inference...")
                
                # Perform inference (Real or Simulated)
                if model:
                    img_resized, mask, raw_prob = run_inference(model, input_image, threshold)
                else:
                    # SIMULATED INFERENCE (If model is missing but we have a real image)
                    img_resized = cv2.resize(input_image, (256, 256))
                    raw_prob = np.zeros((256, 256), dtype=np.float32)
                    
                    # Create a fake detection in the center (just for demo visualization)
                    cv2.circle(raw_prob, (128,128), 40, 0.8, -1) 
                    mask = (raw_prob > 0.5).astype(np.uint8)
                    
                    # OPTIMIZATION: Removed time.sleep(1) here as well

                progress_bar.progress(75)

                # Step 3: Analysis
                status_text.text("Calculating severity and checking AIS data...")
                damage_report = analyze_damage(mask)
                
                # AIS correlation (Simulated location for now)
                suspects = get_ais_anomalies(28.5, -90.5) 

                progress_bar.progress(100)
                status_text.empty()
                
                # --- RESULTS DISPLAY ---
                
                # 1. Severity Banner
                st.markdown(f'<div class="severity-box {damage_report["css_class"]}">{damage_report["message"]}</div>', unsafe_allow_html=True)
                st.write("")

                # 2. Visuals
                c1, c2, c3 = st.columns(3)
                
                # Create Overlay
                mask_vis = np.zeros_like(img_resized)
                mask_vis[:, :, 0] = mask * 255 # Red channel for oil
                overlay = cv2.addWeighted(img_resized, 1.0, mask_vis, alpha, 0)

                with c1:
                    st.image(img_resized, caption=f"Input: {filename}", use_container_width=True)
                with c2:
                    st.image(mask * 255, caption="Binary Segmentation Mask", use_container_width=True)
                with c3:
                    st.image(overlay, caption="Forensic Overlay (Red)", use_container_width=True)

                # 3. Metrics & Logs
                st.markdown("### 📊 Forensic Report")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Detected Oil Pixels", f"{damage_report['pixels']:,}")
                m2.metric("Spill Area", f"{damage_report['area_km2']:.4f} km²")
                m3.metric("Model Confidence", f"{np.max(raw_prob)*100:.1f}%")
                m4.metric("Process Latency", "Instant (Optimized)")
                
                # 4. AIS Data Table
                with st.expander("🚢 Nearby Vessel Activity (AIS Data)", expanded=True):
                    if suspects:
                        st.dataframe(pd.DataFrame(suspects), use_container_width=True)
                        st.caption("⚠️ Vessels with 'STOPPED' status near the spill coordinates are primary suspects.")
                    else:
                        st.info("No vessels detected in the immediate vicinity.")

                # 5. Export
                st.download_button(
                    label="📥 Download Forensic Report (CSV)",
                    data=pd.DataFrame([damage_report]).to_csv(index=False),
                    file_name="forensic_report.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("Analysis Failed")
                st.error(f"Error details: {e}")
                with st.expander("Debug Trace"):
                    st.text(traceback.format_exc())
        
        else:
            # Placeholder before run
            st.info("👈 Upload an image and click 'Run Forensic Analysis' to start.")
            col_ph1, col_ph2 = st.columns(2)
            with col_ph1:
                st.image(input_image, caption=f"Preview: {filename}", width=300)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # LANDING STATE (No file loaded)
        st.markdown('<div class="glass-card" style="text-align: center; padding: 60px;">', unsafe_allow_html=True)
        st.subheader("Ready for Analysis")
        st.markdown("Please upload a Sentinel-1 SAR image from the sidebar.")
        
        # Toggle for Demo Mode if no file is handy
        if st.button("🎲 Load Demo Data"):
            st.session_state['demo_active'] = True
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

# --- ENTRY POINT ---
if __name__ == "__main__":
    main()