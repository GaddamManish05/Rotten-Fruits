# FreshGuard AI - Code Defense & File-by-File Explanation Guide

This guide breaks down every core source code file in **FreshGuard AI**. Use this to explain the architecture, methods, hyperparameters, and engineering design choices to an examining professor or viva committee.

---

## 1. `config.py` — Centralized System Configuration
- **Purpose:** Central source of truth for paths, class definitions, image dimensions, training hyperparameters, and thresholds.
- **Key Design Decisions:**
  - `INPUT_SHAPE = (224, 224, 3)`: Standard resolution required by MobileNetV2 and ImageNet topology.
  - `CLASS_DISPLAY_NAMES`, `CLASS_FRESHNESS`, `CLASS_FRUIT_TYPE`: Structured metadata dictionaries mapping raw class folders (e.g. `freshapples`) to human-readable strings, freshness status, and fruit species for UI display.
  - `RANDOM_SEED = 42`: Ensures deterministic dataset splitting and reproducible initialization.
  - `LR_SCRATCH = 1e-3`, `LR_FEATURE_EXTRACTION = 1e-3`, `LR_FINETUNING = 1e-5`: Fine-tuning learning rate is two orders of magnitude lower to prevent catastrophic forgetting.

---

## 2. `src/dataset_validation.py` — Data Hygiene & Leakage Prevention
- **Purpose:** Validates dataset integrity before training, detects corrupted images, checks duplicate files, and prevents cross-split data leakage.
- **Key Functions:**
  - `get_image_hash(file_path)`: Computes MD5 checksum of binary image content. If the same image appears in both train and test splits (even under different file names), it is flagged and filtered.
  - `verify_image(file_path)`: Uses Pillow (`img.verify()` and `img.convert('RGB')`) to detect corrupted file headers or truncated pixel data.
  - `check_data_leakage(train_dir, val_dir, test_dir)`: Verifies disjoint hash sets across all splits.
  - `compute_class_weights(class_counts)`: Calculates inverse-frequency class weights:
    \[
    W_k = \\frac{N_{\\text{total}}}{K \\times N_k}
    \]
    ensuring imbalanced classes contribute equally to gradient updates.

---

## 3. `src/data_preprocessing.py` — Data Pipelines & Augmentations
- **Purpose:** Constructs high-throughput `tf.data.Dataset` pipelines with on-the-fly augmentation and architecture-specific preprocessing.
- **Key Functions:**
  - `get_augmentation_layer()`: Returns a `tf.keras.Sequential` containing:
    - `RandomFlip("horizontal_and_vertical")`
    - `RandomRotation(0.15)`
    - `RandomZoom(0.15)`
    - `RandomTranslation(0.1, 0.1)`
    - `RandomContrast(0.1)`
    Only applied during training (`training=True`); validation and test sets remain unaltered.
  - `get_preprocessing_layer(model_type)`: Applies the exact normalization expected by each backbone:
    - `scratch`: Rescales pixel values from \([0, 255]\) to \([0, 1]\).
    - `mobilenet`: Calls `tf.keras.applications.mobilenet_v2.preprocess_input`, scaling to \([-1, 1]\).
  - `create_train_pipeline()`: Builds optimized pipelines with `.prefetch(tf.data.AUTOTUNE)` to prevent GPU/CPU starvation.
  - `preprocess_single_image(image)`: Resizes any uploaded PIL image to \(224 \\times 224 \\times 3\) float32 array for real-time inference.

---

## 4. `src/model_factory.py` — Model Architectures
- **Purpose:** Assembles the neural network graphs for Experiments A, B, C, and D.
- **Key Functions:**
  - `build_scratch_cnn()`:
    - 4 Conv2D blocks with Batch Normalization and MaxPooling2D.
    - Global Average Pooling, L2 kernel regularization, Dropout (0.4), Dense(128, relu), Dense(6, softmax).
    - Initialized with random weights as the empirical baseline.
  - `build_mobilenetv2_feature_extraction()`:
    - Instantiates `MobileNetV2(weights='imagenet', include_top=False)`.
    - Sets `base_model.trainable = False` to freeze all 154 layers.
    - Appends a custom classification head: GlobalAveragePooling2D, BatchNormalization, Dropout(0.3), Dense(128, relu), Dropout(0.2), Dense(6, softmax).
  - `build_mobilenetv2_finetuning()`:
    - Unfreezes deeper layers (from index 100+).
    - Freezes earlier generic layers (layers 0 to 99).
    - Freezes BatchNormalization layers to preserve ImageNet population statistics.
    - Compiles with Adam optimizer and conservative learning rate (\(10^{-5}\)).

---

## 5. `src/train_utils.py` — Standardized Training Loop & Callbacks
- **Purpose:** Standardizes training callbacks, history logging, and learning curve plotting.
- **Key Functions:**
  - `get_standard_callbacks(model_save_path)`:
    - `ModelCheckpoint`: Saves best weights based on minimum validation loss (`val_loss`).
    - `EarlyStopping`: Halts training if `val_loss` fails to improve for 4 consecutive epochs, restoring the best model weights.
    - `ReduceLROnPlateau`: Multiplies learning rate by 0.2 if `val_loss` plateaus for 2 epochs.
  - `save_training_history()`: Persists training and validation loss/accuracy across epochs into clean JSON files.
  - `plot_and_save_curves()`: Automatically generates and saves publication-quality PNG plots of accuracy and loss curves in `outputs/figures/`.

---

## 6. `src/evaluate.py` — Benchmark & Held-Out Test Evaluation
- **Purpose:** Evaluates all trained models on the exact same held-out test split under identical conditions.
- **Key Metrics Computed:**
  - Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and Weighted-F1 via Scikit-Learn.
  - Confusion Matrix saved as high-resolution Seaborn heatmaps.
  - Inference Latency (milliseconds per sample) and Model Size on disk (MB).
  - Outputs summary JSON (`outputs/metrics/model_comparison.json`) and terminal comparison tables.

---

## 7. `src/gradcam.py` — Explainability Engine
- **Purpose:** Generates Gradient-weighted Class Activation Maps (Grad-CAM) to visualize spatial regions influencing the prediction.
- **Key Functions:**
  - `generate_gradcam_heatmap(img_array, model, pred_index)`:
    - Constructs a gradient tape between the target output class logit and the last convolutional layer (`Conv_1` in MobileNetV2 or `conv4_1` in Scratch CNN).
    - Calculates spatial average of gradients to weight feature activation maps.
    - Applies ReLU to preserve only positive feature contributions.
  - `overlay_heatmap_on_image(original_img, heatmap)`:
    - Resizes heatmap, applies the Jet colormap, and blends it with the original fruit image at \(\\alpha = 0.4\).

---

## 8. `src/predict.py` — Inference & Safety Advisory Engine
- **Purpose:** Orchestrates single-image prediction with model caching, confidence scoring, probability ranking, and safety advisory generation.
- **Key Design:**
  - `_LOADED_MODELS`: In-memory dictionary cache to prevent reloading heavy models on every Streamlit interaction.
  - Confidence assessment thresholds (High \(\\ge 80\\%\\), Moderate \(50-80\\%\\), Low \(< 50\\%\)).
  - Food safety notice explaining that optical classification cannot detect internal microbial decomposition or biochemical mycotoxins.

---

## 9. `scripts/prepare_dataset.py` — Automated Dataset Ingestion
- **Purpose:** Downloads or finds the Kaggle fruit dataset, splits it into Train, Val, and Test folders with leakage prevention, and supports multiple dataset size modes:
  - `--mode quick`: ~1,800 images (ideal for fast laptop training & defense demo).
  - `--mode medium`: ~4,300 images.
  - `--mode full`: All 13,000+ Kaggle images.

---

## 10. `scripts/run_all_experiments.py` — Master Pipeline Runner
- **Purpose:** Sequential master script that runs Experiment A (Scratch), Experiment B (Feature Extraction), Experiment C (Fine-Tuning), and Step 4 (Evaluation) in a single command.
