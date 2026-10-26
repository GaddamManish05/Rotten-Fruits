"""
FreshGuard AI - Page 1: Fruit Freshness Prediction
Performs single-image inference with confidence scoring, probability distribution,
Grad-CAM explainability, and safety advisories.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.predict import get_available_models, predict_fruit

st.set_page_config(page_title="Predict Fruit Quality | FreshGuard AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .pred-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.3rem;
    }
    .fresh-badge {
        background-color: #2e7d32;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: bold;
        display: inline-block;
    }
    .rotten-badge {
        background-color: #c62828;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: bold;
        display: inline-block;
    }
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .warning-banner {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 15px 0;
        color: #e65100;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="pred-header">🔍 Fruit Quality & Freshness Classifier</div>', unsafe_allow_html=True)
st.caption("Upload a fruit photo or select a preloaded sample to test trained models with confidence scoring and Grad-CAM explainability.")

available_models = get_available_models()

# Sidebar options
st.sidebar.markdown("### ⚙️ Inference Settings")
if not available_models:
    st.sidebar.error("⚠️ No trained models found in `models/`. Please train models first using the training scripts or run `python scripts/run_all_experiments.py`.")
    selected_model_name = None
else:
    selected_model_name = st.sidebar.selectbox(
        "Select Model Architecture:",
        options=list(available_models.keys()),
        index=0,
    )
    st.sidebar.caption(f"Model File: `{available_models[selected_model_name].name}`")

enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Attention Heatmap", value=True)

# Image Source Selection
st.markdown("### 1. Select or Upload Image")
img_source = st.radio(
    "Choose Image Source:",
    options=["Upload My Own Image", "Select from Test Dataset Samples"],
    horizontal=True,
)

pil_image = None

if img_source == "Upload My Own Image":
    uploaded_file = st.file_uploader(
        "Upload fruit image (JPG, JPEG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error opening image file: {e}")

else:
    # Test dataset sample picker
    test_dir = config.TEST_DIR
    if test_dir.exists():
        classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
        if classes:
            col_cls, col_sample = st.columns([1, 2])
            with col_cls:
                chosen_class = st.selectbox(
                    "Fruit Category:",
                    options=classes,
                    format_func=lambda c: config.CLASS_DISPLAY_NAMES.get(c, c),
                )
            sample_files = list((test_dir / chosen_class).glob("*.*"))
            if sample_files:
                with col_sample:
                    chosen_idx = st.slider("Sample Index:", 1, len(sample_files), 1)
                    chosen_file = sample_files[chosen_idx - 1]
                    pil_image = Image.open(chosen_file)
            else:
                st.warning("No samples in this class directory.")
        else:
            st.warning("Test folder is empty. Run dataset preparation first.")
    else:
        st.warning("Dataset not found. Run dataset preparation script.")

# Inference and Results
if pil_image is not None:
    st.markdown("---")
    st.markdown("### 2. Image & Classification Results")

    col_img, col_results = st.columns([1.2, 1.8])

    with col_img:
        st.image(pil_image, caption="Input Fruit Image", use_container_width=True)

    with col_results:
        if selected_model_name is None:
            st.warning("Please wait for model training to complete or train a model to run inference.")
        else:
            with st.spinner("Analyzing image features..."):
                result = predict_fruit(
                    image_input=pil_image,
                    model_name_or_key=selected_model_name,
                    compute_gradcam=enable_gradcam,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                # Fresh vs Rotten Badge
                freshness = result["freshness"]
                badge_class = "fresh-badge" if freshness == "Fresh" else "rotten-badge"
                badge_icon = "✅ FRESH" if freshness == "Fresh" else "⚠️ ROTTEN / SPOILED"

                st.markdown(f'<div class="{badge_class}">{badge_icon}</div>', unsafe_allow_html=True)
                st.markdown(f"### **{result['display_name']}**")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Category", result["fruit_type"])
                m2.metric("Freshness State", freshness)
                m3.metric("Confidence", f"{result['confidence_percentage']}%")

                # Confidence Evaluation
                if result["is_low_confidence"]:
                    st.markdown("""
                    <div class="warning-banner">
                        ⚠️ <strong>Low Confidence Alert (&lt; 50%):</strong> The model is uncertain about this classification. 
                        This often occurs when the image features atypical lighting, partial occlusion, 
                        or visual characteristics intermediate between fresh and overripe stages.
                    </div>
                    """, unsafe_allow_html=True)
                elif result["confidence_level"] == "Moderate":
                    st.info("ℹ️ **Moderate Confidence (50% - 80%):** The prediction is likely correct, but minor ambiguity exists.")

                # Probability Distribution Bar Chart
                st.markdown("#### Class Probability Distribution")
                probs = result["all_probabilities"]
                st.bar_chart(probs)

    # Grad-CAM Section
    if enable_gradcam and "gradcam_image" in result and result["gradcam_image"] is not None:
        st.markdown("---")
        st.markdown("### 3. Visual Explainability (Grad-CAM Attention Map)")
        st.caption(
            "Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the convolutional feature map regions "
            "that contributed most strongly to the predicted classification. Warmer colors (red/yellow) indicate primary decision focus."
        )

        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.image(pil_image, caption="Original Input Image", use_container_width=True)
        with col_cam2:
            st.image(result["gradcam_image"], caption=f"Grad-CAM Heatmap ({result['model_used']})", use_container_width=True)

    # Food Safety Disclaimer
    st.markdown("---")
    st.warning(
        "🛡️ **Food Safety Disclaimer:** Computer vision detects visible exterior spoilage, surface blemishes, "
        "and discoloration. It cannot detect odorless bacterial toxins, internal rot, or pathogenic contamination. "
        "Always follow proper hygiene and food inspection standards before consuming agricultural produce."
    )
