import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os
import time

# --- 1. Page Configuration & Custom Styling ---
st.set_page_config(
    page_title="DeepBone | X-Ray Analysis", 
    page_icon="🦴", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inject custom CSS to make it look like a web app
st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Style the result card */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Sidebar Layout ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2867/2867345.png", width=100) # Placeholder medical icon
    st.title("DeepBone Diagnostics")
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    This tool uses a fine-tuned MobileNetV2 Deep Learning model to analyze X-ray images and detect bone fractures.
    
    ### 🛠️ How to use
    1. Upload a clear X-ray image (JPG/PNG).
    2. Wait for the AI to process the scan.
    3. Review the confidence scores.
    """)
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** This tool is for educational purposes and is not a substitute for professional medical advice, diagnosis, or treatment.")

# --- 3. Main Body Header ---
st.markdown('<p class="main-title">🦴 DeepBone X-Ray Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an X-ray scan for AI-powered fracture detection</p>', unsafe_allow_html=True)
st.markdown("---")

# --- 4. Load the Model ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = r"C:\Users\Saif AI\OneDrive - Irbid National University\Desktop\X-ray project\best_fracture_model.keras"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at: `{model_path}`")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- 5. Main App Logic ---
if model:
    # Center the uploader
    col_spacer1, col_uploader, col_spacer2 = st.columns([1, 2, 1])
    with col_uploader:
        uploaded_file = st.file_uploader("Drop your X-ray image here...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.markdown("---")
        
        # Load the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create a beautiful side-by-side layout for analysis
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📸 Scanned Image")
            # Adding a nice border radius via standard Streamlit image captioning
            st.image(image, use_column_width=True, channels="RGB")

        with col2:
            st.markdown("### 🤖 AI Analysis Results")
            
            # Add a slight delay for dramatic "processing" effect
            with st.spinner("Running neural network inference..."):
                time.sleep(1) # Optional: purely for UI feel
                
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_batch)

                # Prediction
                prediction = model.predict(img_preprocessed)
                probability = prediction[0][0]

            # Display Fancy Results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if probability > 0.5:
                confidence = probability * 100
                st.error("🚨 **Diagnosis: Fractured Bone Detected**")
                
                # Use Streamlit Metrics and Progress bars for a sleek look
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                st.progress(int(confidence))
                
                st.info("💡 **Recommendation:** High probability of structural damage. Immediate radiological review is advised.")
                
            else:
                confidence = (1 - probability) * 100
                st.success("✅ **Diagnosis: Healthy (No Fracture Detected)**")
                
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                st.progress(int(confidence))
                
                st.info("💡 **Recommendation:** Bone structure appears intact. No immediate abnormalities detected by the AI.")
                
            # Technical Details Expander
            with st.expander("📊 View Technical Details"):
                st.write(f"**Raw Probability Score:** `{probability:.4f}`")
                st.write("**Model Architecture:** `MobileNetV2 (Fine-Tuned)`")
                st.write("**Input Shape:** `(224, 224, 3)`")