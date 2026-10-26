"""
FreshGuard AI - Home & Main Dashboard
Academic Project: Fresh and Rotten Fruit Classification Using Transfer Learning
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page
st.set_page_config(
    page_title="FreshGuard AI | Fruit Quality Transfer Learning",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2E7D32, #EF6C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .status-badge-ready {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-missing {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .callout-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">FreshGuard AI 🍎🍌🍊</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Fruit Freshness & Quality Classification Using Deep Transfer Learning</div>', unsafe_allow_html=True)

# Overview Columns
col_main, col_sidebar = st.columns([2.3, 1.0])

with col_main:
    st.markdown("""
    ### 🔬 Project Overview
    **FreshGuard AI** is an academic deep learning framework designed to classify agricultural produce (Apples, Bananas, and Oranges) 
    into **Fresh** and **Rotten** states using **Deep Convolutional Neural Networks (CNNs)** and **Transfer Learning**.

    Agricultural post-harvest losses and food wastage account for billions of dollars annually. Automated visual sorting 
    at farm gates, packaging facilities, and retail stores can drastically reduce contamination and spoilage.

    This project demonstrates and rigorously evaluates three progressive deep learning paradigms:
    1. **Experiment A (Baseline)**: A custom 4-block Convolutional Neural Network initialized with random weights and trained from scratch.
    2. **Experiment B (Feature Extraction)**: An **ImageNet-pretrained MobileNetV2** backbone whose convolutional feature extractor is frozen, training only a custom classification head.
    3. **Experiment C (Fine-Tuning)**: Selective unfreezing of the top convolutional blocks of MobileNetV2 with a reduced learning rate to adapt domain-specific visual features while preserving lower-level generic visual detectors.
    """)

    st.markdown("""
    <div class="callout-box">
        <strong>🎓 Academic Focus:</strong> This application demonstrates the empirical advantages of Inductive Transfer Learning:
        drastically reduced training duration, faster loss convergence, and higher generalization performance under sample-limited regimes.
    </div>
    """, unsafe_allow_html=True)

    # Workflow diagram
    st.markdown("### 🔄 End-to-End Pipeline Workflow")
    st.markdown("""
    ```mermaid
    flowchart LR
        A[Fruit Image Input] --> B[Preprocessing 224x224]
        B --> C{Model Selection}
        C -->|Baseline| D[Scratch CNN]
        C -->|Transfer Learning| E[MobileNetV2 Frozen Base]
        C -->|Fine-Tuned| F[MobileNetV2 Deeper Layers]
        D --> G[Softmax Probabilities]
        E --> G
        F --> G
        G --> H[Fresh / Rotten Classification]
        H --> I[Grad-CAM Heatmap Visualization]
    ```
    """)

with col_sidebar:
    st.markdown("### 🚦 Model Status")

    import config

    models_info = [
        ("Scratch CNN (Baseline)", config.MODEL_SCRATCH_PATH),
        ("MobileNetV2 (Feature Ext.)", config.MODEL_FE_PATH),
        ("MobileNetV2 (Fine-Tuned)", config.MODEL_FT_PATH),
    ]

    for label, path in models_info:
        is_ready = path.exists()
        badge = '<span class="status-badge-ready">✓ Trained</span>' if is_ready else '<span class="status-badge-missing">✗ Untrained</span>'
        st.markdown(f"""
        <div class="metric-card">
            <strong>{label}</strong><br>
            Status: {badge}<br>
            <small>File: <code>{path.name}</code></small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📚 Navigation")
    st.info("""
    Select a module from the sidebar:
    - **1_Predict**: Upload an image for real-time freshness classification.
    - **2_Model_Comparison**: Inspect benchmark tables, ROC/Accuracy curves, and confusion matrices.
    - **3_Dataset_Explorer**: View class distributions, sample images, and augmentation.
    - **4_Transfer_Learning_Theory**: Academic guide to types, methodologies, and viva concepts.
    - **5_Error_Analysis_and_GradCAM**: Visual explainability, misclassification analysis, and food safety caveats.
    """)

st.markdown("---")
st.caption("FreshGuard AI | Built with TensorFlow 2.18, Keras 3, Streamlit, and Scikit-Learn | Academic Transfer Learning Project")
