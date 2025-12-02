import streamlit as st
import numpy as np
import cv2
import os
import random
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Oil Spill Forensic Tool",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #3b3d45;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Metric Label - FIXED VISIBILITY */
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    /* Metric Value - FIXED VISIBILITY */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Severity Box Styling */
    .severity-box {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🔧 Forensic Control")
st.sidebar.markdown("---")

# 1. DATA SOURCE SELECTION
data_source = st.sidebar.radio("Select Input Source", ["📂 Upload File", "🎲 Random Demo Data"])

input_image = None
demo_file_name = ""

# Logic for Handling Data Sources
if data_source == "📂 Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

elif data_source == "🎲 Random Demo Data":
    # --- UPDATED DIRECTORY PATH ---
    test_dir = "data/test/images"
    
    st.sidebar.info(f"Using random images from '{test_dir}'")
    
    # Check if folder exists
    if not os.path.exists(test_dir):
        st.sidebar.error(f"❌ Directory '{test_dir}' not found!")
        st.error(f"⚠️ Directory Not Found: Please ensure the folder path '{test_dir}' is correct.")
    else:
        # Get list of valid images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not images:
            st.sidebar.error("No images found in the directory!")
        else:
            # Use Session State to keep the image stable during slider interaction
            if 'current_demo_image' not in st.session_state:
                st.session_state.current_demo_image = random.choice(images)

            # Button to pick a NEW random image
            if st.sidebar.button("🎲 Get New Random Image"):
                st.session_state.current_demo_image = random.choice(images)
            
            # Load the selected image
            demo_file_name = st.session_state.current_demo_image
            image_path = os.path.join(test_dir, demo_file_name)
            
            # Use OpenCV to read image from the path
            input_image = cv2.imread(image_path)
            
            st.sidebar.success(f"Loaded: {demo_file_name}")

st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider("Detection Sensitivity", 0.0, 1.0, 0.05, 0.01)

# --- MAIN APP ---
st.title("🛢️ Oil Spill Forensic System")
st.caption("Satellite SAR Analysis & Environmental Damage Assessment")

# Load Model (Cached)
@st.cache_resource
def load_ai_model():
    # Ensure this path is correct for your environment
    return load_model('saved_models/unet_oil_spill.h5')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

if input_image is not None:
    # --- PROCESSING LOGIC ---
    # Convert BGR to RGB (OpenCV loads BGR by default)
    original_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocess
    img_resized = cv2.resize(original_img, (256, 256))
    input_data = np.expand_dims(img_resized, axis=0) / 255.0
    
    # 3. Predict
    # Check if model is loaded before predicting
    if 'model' in locals():
        raw_prediction = model.predict(input_data)[0]
        mask = (raw_prediction > confidence_threshold).astype(np.uint8)

        # 4. Create Red Forensic Overlay
        mask_red = np.zeros_like(img_resized)
        mask_red[:,:,0] = mask[:,:,0] * 255  # Set Red channel to 255
        
        # Blend
        overlay = cv2.addWeighted(img_resized, 0.7, mask_red, 0.3, 0)

        # 5. Calculate Damage / Area
        oil_pixels = np.count_nonzero(mask)
        area_sq_km = (oil_pixels * 100) / 1_000_000 # 1 pixel = 100m^2
        model_confidence = np.max(raw_prediction) * 100

        # --- DISPLAY ---
        st.markdown("---")

        # Severity Status
        if area_sq_km > 1.0:
            st.error(f"🚨 **CRITICAL SEVERITY** | Cleanup required ({area_sq_km:.2f} km²)")
        elif area_sq_km > 0.1:
            st.warning(f"⚠️ **HIGH SEVERITY** | Booms advised ({area_sq_km:.2f} km²)")
        elif area_sq_km > 0.0:
            st.info(f"ℹ️ **MODERATE SEVERITY** | Minor leak ({area_sq_km:.4f} km²)")
        else:
            st.success("✅ **NO SPILL DETECTED** | Area is clear")

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Detected Oil Pixels", f"{oil_pixels}")
        m2.metric("Est. Spill Area", f"{area_sq_km:.4f} km²")
        m3.metric("Model Confidence", f"{model_confidence:.1f}%")

        st.markdown("### 🛰️ Visual Analysis")

        # Tabbed View
        tab1, tab2, tab3 = st.tabs(["🔍 Forensic Overlay", "🧠 AI Mask Analysis", "📷 Original Input"])

        with tab1:
            st.image(overlay, use_container_width=True, caption=f"Analysis of {demo_file_name if demo_file_name else 'Uploaded Image'}")
            
        with tab2:
            col_mask_1, col_mask_2 = st.columns([1, 3])
            with col_mask_1:
                st.markdown("**Mask Details:**\n- White: Oil\n- Black: Water\n- U-Net Generated")
            with col_mask_2:
                st.image(mask * 255, use_container_width=True, caption="Binary Segmentation Mask")
            
        with tab3:
            st.image(img_resized, use_container_width=True, caption="Raw Input (256x256)")
    else:
        st.warning("Cannot run prediction: Model failed to load.")

else:
    # Empty State
    st.info("👈 Select 'Random Demo Data' or Upload an image to start.")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 50px;">
        <h4>Waiting for input...</h4>
    </div>
    """, unsafe_allow_html=True)