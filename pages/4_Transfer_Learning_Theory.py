"""
FreshGuard AI - Page 4: Transfer Learning Academic Theory & Syllabus Mapping
In-depth theoretical guide covering types, methodologies, mathematical formulations,
challenges, and an interactive Viva self-assessment quiz.
"""

import streamlit as st

st.set_page_config(page_title="Transfer Learning Theory | FreshGuard AI", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .theory-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4A148C;
        margin-bottom: 0.3rem;
    }
    .concept-card {
        background-color: #fdfbff;
        border: 1px solid #e1bee7;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .callout-taxonomy {
        background-color: #ede7f6;
        border-left: 5px solid #673ab7;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="theory-header">📖 Transfer Learning: Academic Theory & Taxonomy</div>', unsafe_allow_html=True)
st.caption("Comprehensive academic guide mapping university deep learning syllabi to empirical fruit classification.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Types of Transfer Learning",
    "2. Methodologies & Architectures",
    "3. Fine-Tuning Mechanics",
    "4. Challenges & Mitigations",
    "5. Interactive Viva Quiz",
])

with tab1:
    st.markdown("### 🏛️ Taxonomy of Transfer Learning (Pan & Yang, 2010)")
    st.markdown("""
    In formal transfer learning terminology, a **Domain** \(\\mathcal{D}\) consists of a feature space \(\\mathcal{X}\) 
    and a marginal probability distribution \(P(X)\): \(\\mathcal{D} = \\{\\mathcal{X}, P(X)\\}\\).
    A **Task** \(\\mathcal{T}\) consists of a label space \(\\mathcal{Y}\) and an objective predictive function \(f(\\cdot)\): \(\\mathcal{T} = \\{\\mathcal{Y}, P(Y|X)\\}\\).
    """)

    st.markdown("""
    | Transfer Learning Type | Source & Target Domains | Source & Target Tasks | Target Label Availability | FreshGuard AI Mapping |
    | :--- | :--- | :--- | :--- | :--- |
    | **Inductive Transfer Learning** | May be same or different | Different Tasks (\(\\mathcal{T}_S \\neq \\mathcal{T}_T\)) | **Labeled data available** in target domain | **✅ Yes (This Project)** |
    | **Transductive Transfer Learning** | Different Domains (\(\\mathcal{D}_S \\neq \\mathcal{D}_T\)) | Same Task (\(\\mathcal{T}_S = \\mathcal{T}_T\)) | Unlabeled target data (e.g. Domain Adaptation) | ❌ No |
    | **Unsupervised Transfer Learning** | Different Tasks | Different Tasks | **No labeled data** in either domain (Clustering/DR) | ❌ No |
    """)

    st.markdown("""
    <div class="callout-taxonomy">
        <strong>🎯 Clarification for Viva Defense:</strong> FreshGuard AI is an exact instance of 
        <strong>Inductive Transfer Learning</strong>. The source task is 1,000-class object categorization on ImageNet (\(\\mathcal{Y}_S\)), 
        while our target task is 6-class agricultural freshness classification (\(\\mathcal{Y}_T\)). 
        Target labels are provided during training to adapt the network.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### ⚙️ Deep Learning Methodologies Compared")
    st.markdown("""
    Deep Convolutional Networks learn hierarchical representations:
    - **Early Layers (Low-Level):** Detect edges, corner points, high-frequency spatial gradients, and color transitions. Highly generic across all visual tasks.
    - **Middle Layers (Mid-Level):** Form basic textures, contours, surface ripples, and geometric shapes.
    - **Deeper Layers (High-Level):** Specialize into complex semantic shapes, object parts, and class-specific motifs.
    """)

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("""
        <div class="concept-card">
            <h4>1. Training from Scratch</h4>
            <p><strong>Weights:</strong> He/Glorot Normal Random Init</p>
            <p><strong>Trainable:</strong> 100% of parameters</p>
            <p><strong>Risk:</strong> High variance, overfitting on small data, heavy compute requirements.</p>
            <p><strong>Baseline role:</strong> Proves whether pretrained features offer statistical benefits.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="concept-card">
            <h4>2. Feature Extraction</h4>
            <p><strong>Weights:</strong> Pretrained on ImageNet-1k</p>
            <p><strong>Trainable:</strong> Only new Classification Head</p>
            <p><strong>Backbone:</strong> Completely frozen (<code>trainable = False</code>)</p>
            <p><strong>Advantage:</strong> Fastest training, zero risk of destroying pretrained feature representations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown("""
        <div class="concept-card">
            <h4>3. Fine-Tuning</h4>
            <p><strong>Weights:</strong> Pretrained + partially updated</p>
            <p><strong>Trainable:</strong> Head + Top Conv Blocks</p>
            <p><strong>Learning Rate:</strong> Very low (\(10^{-5}\))</p>
            <p><strong>Advantage:</strong> Tailors high-level filters to subtle target patterns (bruises, mold) while maintaining stability.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🔧 Fine-Tuning Mechanics: Preventing Catastrophic Forgetting")
    st.markdown("""
    #### Why standard fine-tuning can fail if done incorrectly:
    1. **Catastrophic Forgetting:** If a high learning rate (\(10^{-2}\) or \(10^{-3}\)) is applied to unfrozen pretrained layers, the large gradients propagated from randomly initialized head weights will completely overwrite the delicate pretrained weights in the backbone.
    2. **Batch Normalization Instability:** Unfreezing `BatchNormalization` layers on small batches can ruin the running mean and variance computed over 1.4 million ImageNet images. In FreshGuard AI, we freeze BatchNorm layers or call the base model in inference mode (`training=False`).
    3. **Two-Stage Training Workflow:**
       - **Stage 1:** Train only the newly added top head until convergence while the backbone remains fully frozen.
       - **Stage 2:** Unfreeze the top convolutional blocks (e.g. layers 100+) and train with a reduced learning rate \(\\eta = 10^{-5}\).
    """)

with tab4:
    st.markdown("### 🛡️ Critical Deep Learning Challenges & FreshGuard Mitigations")

    challenges = [
        ("Limited Agricultural Data", "Fruit datasets are rarely as massive as ImageNet. Mitigation: Pretrained backbones + multi-axis data augmentation (rotation, zoom, translation)."),
        ("Class Imbalance", "Rotten fruits in some harvests outnumber fresh fruits. Mitigation: Inverse-frequency class weighting in categorical cross-entropy loss."),
        ("Visual Similarity & Confusing Features", "Ripening sugar spots on bananas visually mimic early rot; orange dimpling mimics fungal lesions. Mitigation: Higher resolution (224x224), deep feature maps, and Grad-CAM attention checking."),
        ("Domain Shift", "Lab/studio images with plain white backgrounds vs real-world orchard/market lighting. Mitigation: Random contrast and color jittering during training."),
        ("Confidence vs True Calibration", "Raw Softmax outputs can produce high confidence on out-of-distribution or noisy images. Mitigation: Explicit confidence thresholds and low-confidence warning banners."),
    ]

    for title, desc in challenges:
        st.markdown(f"**• {title}:** {desc}")

with tab5:
    st.markdown("### 🧠 Interactive Viva / Exam Self-Tester")
    st.caption("Answer these university viva questions to test your knowledge before defense:")

    q1 = st.radio(
        "Q1. Which Transfer Learning paradigm best describes FreshGuard AI?",
        options=[
            "Transductive Transfer Learning (unlabeled target data)",
            "Inductive Transfer Learning (different tasks, target labels present)",
            "Unsupervised Transfer Learning (no labels in either domain)",
            "Self-Supervised Contrastive Learning",
        ],
        index=0,
    )
    if st.button("Check Answer Q1"):
        if "Inductive" in q1:
            st.success("✅ Correct! Because target domain has labels and target task (fruit quality) differs from source task (ImageNet classification).")
        else:
            st.error("❌ Incorrect. FreshGuard AI uses Inductive Transfer Learning.")

    st.markdown("---")
    q2 = st.radio(
        "Q2. Why do we keep Batch Normalization layers frozen during fine-tuning?",
        options=[
            "To reduce memory usage only",
            "Because Batch Normalization does not have any trainable weights",
            "To preserve the robust population mean and variance estimated from millions of ImageNet images",
            "To speed up GPU matrix multiplication",
        ],
        index=0,
    )
    if st.button("Check Answer Q2"):
        if "preserve the robust population mean" in q2:
            st.success("✅ Correct! Updating batch statistics on small target batches can destabilize feature activations.")
        else:
            st.error("❌ Incorrect. The running statistics from ImageNet must be preserved on small datasets.")

    st.markdown("---")
    q3 = st.radio(
        "Q3. What is Catastrophic Forgetting?",
        options=[
            "When GPU runs out of VRAM memory during backpropagation",
            "When a network drastically overwrites previously learned general representations when trained on a new task with a high learning rate",
            "When images are corrupted during disk reading",
            "When validation loss decreases while training loss increases",
        ],
        index=0,
    )
    if st.button("Check Answer Q3"):
        if "drastically overwrites" in q3:
            st.success("✅ Correct! Catastrophic forgetting is prevented by layer freezing and low learning rates.")
        else:
            st.error("❌ Incorrect. Review the fine-tuning mechanics tab.")
