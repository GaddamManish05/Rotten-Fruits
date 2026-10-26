"""
FreshGuard AI - Page 2: Model Comparison & Benchmark Results
Displays held-out test set metrics, learning curves, confusion matrices, and academic analysis.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

st.set_page_config(page_title="Model Comparison | FreshGuard AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .comp-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.3rem;
    }
    .best-banner {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .academic-note {
        background-color: #f3e5f5;
        border-left: 5px solid #8e24aa;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="comp-header">📊 Transfer Learning Benchmark & Model Comparison</div>', unsafe_allow_html=True)
st.caption("Empirical performance evaluation across held-out test data comparing Baseline Scratch CNN, Feature Extraction, and Fine-Tuning.")

# Load saved metrics
metrics_path = config.COMPARISON_METRICS_PATH

if not metrics_path.exists():
    st.warning("⏳ **Benchmark in progress or not yet run:** `model_comparison.json` has not been generated yet. "
               "Run `python scripts/run_all_experiments.py` or check terminal background task status.")
else:
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        results = {}

    if not results:
        st.info("No experiment evaluation metrics recorded yet.")
    else:
        # Build comparison table
        table_rows = []
        for exp_name, m in results.items():
            table_rows.append({
                "Architecture": exp_name,
                "Test Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Macro F1": f"{m['macro_f1']:.4f}",
                "Macro Precision": f"{m['macro_precision']:.4f}",
                "Macro Recall": f"{m['macro_recall']:.4f}",
                "Inference Latency": f"{m['test_latency_ms_per_sample']:.1f} ms",
                "Model Size": f"{m['model_size_mb']:.1f} MB",
                "Total Params": f"{m['total_parameters']:,}",
                "Training Time": f"{m['training_duration_seconds']:.1f}s",
            })

        df = pd.DataFrame(table_rows)

        # Highlight best model
        best_exp = max(results.items(), key=lambda item: item[1]["accuracy"])
        best_name = best_exp[0]
        best_acc = best_exp[1]["accuracy"] * 100
        best_f1 = best_exp[1]["macro_f1"]

        st.markdown(f"""
        <div class="best-banner">
            <h3 style="margin: 0; color: white;">🏆 Top Performing Paradigm: {best_name}</h3>
            <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                Achieved <strong>{best_acc:.2f}% Test Accuracy</strong> and <strong>{best_f1:.4f} Macro-F1 score</strong> on the held-out test dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 1. Comparative Test Metrics Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visual Metrics Comparison
        st.markdown("### 2. Performance Comparison Charts")
        chart_col1, chart_col2 = st.columns(2)

        chart_df = pd.DataFrame({
            "Architecture": [r["Architecture"] for r in table_rows],
            "Accuracy (%)": [float(r["Test Accuracy"].replace("%", "")) for r in table_rows],
            "Macro F1": [float(r["Macro F1"]) for r in table_rows],
            "Latency (ms)": [float(r["Inference Latency"].replace(" ms", "")) for r in table_rows],
            "Size (MB)": [float(r["Model Size"].replace(" MB", "")) for r in table_rows],
        }).set_index("Architecture")

        with chart_col1:
            st.markdown("#### Test Accuracy & Macro-F1")
            st.bar_chart(chart_df[["Accuracy (%)"]])

        with chart_col2:
            st.markdown("#### Model Size (MB) vs Latency (ms)")
            st.bar_chart(chart_df[["Size (MB)", "Latency (ms)"]])

        # Confusion Matrices
        st.markdown("---")
        st.markdown("### 3. Confusion Matrix Breakdown")
        st.caption("Diagonal elements represent correct predictions; off-diagonal elements indicate class confusion (e.g. fresh vs rotten within same fruit or across fruits).")

        cm_cols = st.columns(len(results))
        for idx, (exp_name, m) in enumerate(results.items()):
            with cm_cols[idx]:
                cm_plot_file = config.BASE_DIR / m.get("cm_plot_path", "")
                if cm_plot_file.exists():
                    st.image(Image.open(cm_plot_file), caption=exp_name, use_container_width=True)
                else:
                    st.caption(f"Confusion matrix plot not found for {exp_name}")

        # Training Curves
        st.markdown("---")
        st.markdown("### 4. Training & Validation Learning Curves")
        st.caption("Observing loss convergence and the gap between training and validation curves reveals overfitting and generalization stability.")

        curve_files = [
            ("Experiment A: Scratch CNN", config.FIGURES_DIR / "curves_scratch.png"),
            ("Experiment B: MobileNetV2 Feature Extraction", config.FIGURES_DIR / "curves_feature_extraction.png"),
            ("Experiment C: MobileNetV2 Fine-Tuning", config.FIGURES_DIR / "curves_finetuned.png"),
        ]

        for title, c_path in curve_files:
            if c_path.exists():
                st.markdown(f"#### {title}")
                st.image(Image.open(c_path), use_container_width=True)

        # Academic Discussion
        st.markdown("---")
        st.markdown("""
        <div class="academic-note">
            <h4 style="margin-top:0;">🎓 Deep Learning Researcher Discussion</h4>
            <ul>
                <li><strong>Baseline Scratch CNN:</strong> Starts from random weight distributions. With a limited dataset, earlier convolution filters must learn basic Gabor-like edge and texture representations from scratch, making convergence slower and increasing vulnerability to overfitting.</li>
                <li><strong>Feature Extraction (MobileNetV2):</strong> Freezing the ImageNet-pretrained backbone preserves generic visual representations (edges, textures, gradients, color transitions) learned over 1.4 million natural images. Training only the classification head converges rapidly with orders of magnitude fewer trainable parameters.</li>
                <li><strong>Fine-Tuning (MobileNetV2):</strong> Unfreezing top-level convolutional layers with a low learning rate (\(10^{-5}\)) allows deep high-level filters to specialize toward fruit-specific visual patterns—such as the subtle brownish mottling of early fungal decay—yielding the highest overall discriminative performance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
