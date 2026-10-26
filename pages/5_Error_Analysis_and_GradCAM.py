"""
FreshGuard AI - Page 5: Error Analysis & Explainability (Grad-CAM)
Deep dive into misclassifications, visual confusion causes, Grad-CAM attention maps,
confidence calibration limits, and food safety caveats.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset_validation import discover_classes
from src.predict import get_available_models, predict_fruit

st.set_page_config(page_title="Error Analysis & Grad-CAM | FreshGuard AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .error-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #C62828;
        margin-bottom: 0.3rem;
    }
    .misclass-card {
        background-color: #fff8f8;
        border: 1px solid #ffcdd2;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    .correct-card {
        background-color: #f1f8e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="error-header">🔬 Error Analysis & Explainability (Grad-CAM)</div>', unsafe_allow_html=True)
st.caption("Investigate why deep learning models make mistakes on agricultural images, explore Grad-CAM visual attention, and understand calibration limits.")

available_models = get_available_models()

if not available_models:
    st.warning("⚠️ No trained models found. Train models first to perform error analysis.")
else:
    selected_model = st.selectbox(
        "Select Model for Analysis:",
        options=list(available_models.keys()),
        index=len(available_models) - 1,  # Select latest / fine-tuned by default
    )

    st.markdown("### 1. Test Set Error Explorer")
    st.caption("Inspect how the model performs across challenging images from the held-out test split.")

    test_classes = discover_classes(config.TEST_DIR)

    col_select1, col_select2 = st.columns(2)
    with col_select1:
        target_true_class = st.selectbox(
            "True (Ground-Truth) Class:",
            options=test_classes,
            format_func=lambda c: config.CLASS_DISPLAY_NAMES.get(c, c),
        )

    class_test_dir = config.TEST_DIR / target_true_class
    test_files = list(class_test_dir.glob("*.*")) if class_test_dir.exists() else []

    if test_files:
        with col_select2:
            img_idx = st.slider("Select Image Index:", 1, len(test_files), 1)
            selected_file = test_files[img_idx - 1]

        pil_img = Image.open(selected_file)

        # Predict with Grad-CAM
        res = predict_fruit(
            pil_img,
            model_name_or_key=selected_model,
            compute_gradcam=True,
        )

        is_correct = (res["predicted_class"] == target_true_class)
        card_class = "correct-card" if is_correct else "misclass-card"
        status_text = "✅ CORRECT CLASSIFICATION" if is_correct else "❌ MISCLASSIFICATION ERROR"

        st.markdown(f"""
        <div class="{card_class}">
            <strong>Evaluation Status:</strong> {status_text}<br>
            <strong>Ground Truth:</strong> {config.CLASS_DISPLAY_NAMES.get(target_true_class, target_true_class)}<br>
            <strong>Predicted:</strong> {res['display_name']} ({res['freshness']})<br>
            <strong>Confidence:</strong> {res['confidence_percentage']}% ({res['confidence_level']} confidence)
        </div>
        """, unsafe_allow_html=True)

        col_img1, col_img2, col_probs = st.columns([1, 1, 1.2])
        with col_img1:
            st.image(pil_img, caption=f"Test Image: {selected_file.name}", use_container_width=True)
        with col_img2:
            if res.get("gradcam_image") is not None:
                st.image(res["gradcam_image"], caption="Grad-CAM Attention Heatmap", use_container_width=True)
            else:
                st.info("Grad-CAM visualization not available for this layer.")
        with col_probs:
            st.markdown("#### Softmax Distribution")
            st.bar_chart(res["all_probabilities"])

    # Academic Analysis of Common Failure Modes
    st.markdown("---")
    st.markdown("### 2. Why Agricultural Vision Systems Fail: Common Mistake Patterns")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("""
        #### 🍌 Sugar Spots vs Fungal Rot in Bananas
        - **The Challenge:** Fully ripe bananas develop benign brown senescent spots (*sugar spotting*) caused by natural ethylene synthesis and chlorophyll breakdown.
        - **Model Confusion:** Convolutional filters respond heavily to high-contrast dark spots against a yellow background. The network may misclassify a perfectly edible, ripe banana as "Rotten".
        - **Grad-CAM Insight:** Grad-CAM reveals whether the model is focusing on diffuse small speckles (sugar spots) or large localized sunken lesions with water-soaking (true anthracnose decay).
        """)

    with col_e2:
        st.markdown("""
        #### 🍊 Mold Discoloration vs Surface Blemishes in Oranges
        - **The Challenge:** Citrus fruits naturally exhibit russeting, wind scarring, or green stem ends that do not signify rot.
        - **Model Confusion:** Superficial scars can trigger high activation in edge-detection filters, yielding false positives for decay.
        - **Cross-Fruit Confusion:** Severe decay in apples and oranges both presents as dark brown, collapsed skin, leading to cross-category confusion when the fruit silhouette is partially cropped.
        """)

    # Confidence Calibration vs Accuracy
    st.markdown("---")
    st.markdown("### 3. Confidence Calibration vs Ground-Truth Correctness")
    st.markdown("""
    A fundamental limitation of modern Deep Neural Networks with Softmax output layers is **miscalibration** (Guo et al., ICML 2017):
    - **Overconfidence:** Softmax pushes probabilities toward 1.0 or 0.0 due to cross-entropy loss minimization, even when the model is evaluating an atypical or ambiguous image.
    - **High Confidence \(\\neq\) Certainty:** A model predicting 95% confidence on a bruised fruit does not guarantee a 95% likelihood of bacterial presence.
    - **Mitigation:** In FreshGuard AI, predictions with confidence below 80% trigger advisory banners, and predictions below 50% are flagged as uncertain.
    """)

    # Critical Food Safety Advisory
    st.markdown("---")
    st.error("""
    🛡️ **CRITICAL FOOD SAFETY & ETHICAL NOTE FOR ACADEMIC DEFENSE:**
    Computer vision models inspect strictly **external surface light reflectance**. 
    They cannot evaluate:
    1. Internal anaerobic bacterial rot (e.g. *Erwinia carotovora*).
    2. Odorless fungal mycotoxins (e.g. patulin in apples).
    3. Foodborne pathogens (*Salmonella*, *Listeria monocytogenes*).
    
    Therefore, FreshGuard AI is an **educational post-harvest grading assistive tool**, NOT an autonomous food safety certification system.
    """)
