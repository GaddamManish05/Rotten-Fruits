# PROJECT SYNOPSIS

## 1. Project Title
**FreshGuard AI: Fresh and Rotten Fruit Classification Using Transfer Learning**

## 2. Domain & Area
- **Domain:** Artificial Intelligence & Computer Vision
- **Sub-Area:** Deep Learning, Transfer Learning, Agricultural Post-Harvest Quality Assessment

## 3. Introduction & Background
In the agricultural and retail food supply chain, post-harvest losses due to fungal decay, over-ripening, and bacterial rot represent a staggering economic and environmental challenge. Global food waste exceeds 1.3 billion tonnes annually, with fresh fruits and vegetables accounting for the highest wastage rates (approximately 45%). Traditionally, quality sorting relies on manual sensory inspection by human sorters. This process is subjective, labor-intensive, error-prone, and unsustainable for industrial-scale distribution.

Recent breakthroughs in Convolutional Neural Networks (CNNs) have demonstrated human-level image classification capabilities. However, training deep vision architectures from scratch requires massive annotated datasets (hundreds of thousands of images) and high-performance computing clusters—resources unavailable to small-scale farms, developing agricultural hubs, and academic student laboratories. **Transfer Learning** overcomes these barriers by transferring general visual representations learned on large-scale source datasets (such as ImageNet-1k) to specialized agricultural target domains with limited data.

## 4. Problem Statement
To design, implement, evaluate, and deploy an automated, lightweight, and explainable deep learning framework that accurately distinguishes fresh from rotten fruits (Apples, Bananas, and Oranges) using Inductive Transfer Learning, providing real-time inference, confidence estimation, visual attention maps (Grad-CAM), and empirical methodology comparison.

## 5. Objectives
1. **Curate and Validate Dataset:** Assemble a verified 6-class dataset (*Fresh Apples, Rotten Apples, Fresh Bananas, Rotten Bananas, Fresh Oranges, Rotten Oranges*), ensuring data hygiene, corruption filtering, and zero cross-split data leakage.
2. **Empirical Transfer Learning Comparison:** Implement and benchmark three distinct modeling paradigms:
   - *Baseline (Experiment A):* 4-block custom CNN initialized with random weights and trained from scratch.
   - *Feature Extraction (Experiment B):* Pretrained MobileNetV2 with a frozen convolutional backbone and custom classification head.
   - *Fine-Tuning (Experiment C):* Selectively unfreezing top convolutional blocks with a conservative learning rate (\(10^{-5}\)) and inference-mode batch normalization.
3. **Model Evaluation & Explainability:** Evaluate on an identical held-out test set using multi-class metrics (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix, and inference latency). Integrate Grad-CAM for visual accountability.
4. **Interactive Web Deployment:** Deploy an interactive, student-friendly Streamlit web application providing single-image upload, probability distributions, dataset exploration, and an academic viva preparation tool.

## 6. System Architecture Summary
- **Input Pipeline:** RGB fruit imagery normalized to \(224 \\times 224 \\times 3\) with architecture-specific scaling.
- **Deep Learning Framework:** TensorFlow 2.18 / Keras 3 with GPU/CPU auto-detection.
- **Backbone Architectures:** MobileNetV2 (Inverted Residuals & Linear Bottlenecks) and baseline CNN.
- **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM).
- **Presentation Layer:** Multi-page Streamlit application.

## 7. Expected Outcomes
- An end-to-end, reproducible deep learning system running efficiently on standard laptops.
- Direct empirical proof that Transfer Learning achieves higher accuracy and faster convergence compared to scratch CNNs under data-limited conditions.
- A transparent interface ensuring explainability and highlighting safety caveats regarding visual inspection limits.
