"""
FreshGuard AI - Inference & Prediction Module
Provides robust inference on single images with confidence assessment,
class probabilities, safety advisory, and Grad-CAM explainability integration.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from src.data_preprocessing import preprocess_single_image
from src.gradcam import generate_gradcam_heatmap, overlay_heatmap_on_image

# Cached model dictionary to avoid reloading on every request in Streamlit
_LOADED_MODELS: Dict[str, tf.keras.Model] = {}


def get_available_models() -> Dict[str, Path]:
    """Returns dictionary of currently trained available models."""
    available = {}
    if config.MODEL_SCRATCH_PATH.exists():
        available["Scratch CNN (Baseline)"] = config.MODEL_SCRATCH_PATH
    if config.MODEL_FE_PATH.exists():
        available["MobileNetV2 (Feature Extraction)"] = config.MODEL_FE_PATH
    if config.MODEL_FT_PATH.exists():
        available["MobileNetV2 (Fine-Tuned)"] = config.MODEL_FT_PATH
    if config.MODEL_EFFICIENTNET_PATH.exists():
        available["EfficientNetB0 (Transfer Learning)"] = config.MODEL_EFFICIENTNET_PATH
    return available


# Streamlit resource cache decorator if available, else identity pass-through
try:
    import streamlit as st
    _cache_decorator = st.cache_resource
except Exception:
    def _cache_decorator(f):
        return f


@_cache_decorator
def _load_keras_model(file_path_str: str) -> Optional[tf.keras.Model]:
    """Loads a Keras model from disk with required custom objects."""
    try:
        custom_objects = {
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        }
        return tf.keras.models.load_model(file_path_str, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model from {file_path_str}: {e}")
        return None


def load_model_by_key(model_key: str) -> Optional[tf.keras.Model]:
    """Loads and caches model by user selection key or file path."""
    global _LOADED_MODELS
    available = get_available_models()

    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]

    target_path = None
    if model_key in available:
        target_path = available[model_key]
    else:
        # Check direct path
        p = Path(model_key)
        if p.exists():
            target_path = p

    if target_path is None or not target_path.exists():
        return None

    model = _load_keras_model(str(target_path))
    if model is not None:
        _LOADED_MODELS[model_key] = model
    return model


def predict_fruit(
    image_input: Union[Image.Image, Path, str],
    model: Optional[tf.keras.Model] = None,
    model_name_or_key: str = "MobileNetV2 (Feature Extraction)",
    class_names: Optional[List[str]] = None,
    compute_gradcam: bool = False,
) -> Dict:
    """
    Performs inference on a single image.
    Returns rich prediction dictionary with confidence, probabilities, and safety advisory.
    """
    if class_names is None:
        class_names = config.DEFAULT_CLASSES

    # Load PIL image
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Unsupported image input type.")

    # Load model if not provided
    if model is None:
        model = load_model_by_key(model_name_or_key)
        if model is None:
            # Try any available model
            available = get_available_models()
            if available:
                first_key = list(available.keys())[0]
                model = load_model_by_key(first_key)
                model_name_or_key = first_key

    if model is None:
        return {
            "error": "No trained model found. Please train a model first using the training scripts.",
            "available_models": list(get_available_models().keys()),
        }

    # Preprocess image
    img_batch = preprocess_single_image(image)

    # Predict
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

    display_name = config.CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class.title())
    freshness = config.CLASS_FRESHNESS.get(predicted_class, "Unknown")
    fruit_type = config.CLASS_FRUIT_TYPE.get(predicted_class, "Fruit")

    # Detailed probability map sorted descending
    probabilities = {}
    for idx, prob in enumerate(preds):
        cls_key = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        cls_disp = config.CLASS_DISPLAY_NAMES.get(cls_key, cls_key)
        probabilities[cls_disp] = round(float(prob), 4)

    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    # Confidence assessment
    is_low_confidence = confidence < config.CONFIDENCE_THRESHOLD_MEDIUM
    is_moderate_confidence = config.CONFIDENCE_THRESHOLD_MEDIUM <= confidence < config.CONFIDENCE_THRESHOLD_HIGH

    if confidence >= config.CONFIDENCE_THRESHOLD_HIGH:
        confidence_level = "High"
    elif is_moderate_confidence:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    # Grad-CAM visualization
    gradcam_overlay = None
    if compute_gradcam:
        heatmap = generate_gradcam_heatmap(img_batch, model, pred_index=pred_idx)
        if heatmap is not None:
            gradcam_overlay = overlay_heatmap_on_image(image, heatmap)

    result = {
        "success": True,
        "model_used": model_name_or_key,
        "predicted_class": predicted_class,
        "display_name": display_name,
        "freshness": freshness,
        "fruit_type": fruit_type,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_level": confidence_level,
        "is_low_confidence": is_low_confidence,
        "all_probabilities": sorted_probabilities,
        "gradcam_image": gradcam_overlay,
        "food_safety_advisory": (
            "⚠️ ACADEMIC & SAFETY NOTICE: This classification is based solely on surface optical features. "
            "Internal microbial contamination, invisible mycotoxins, or bacterial growth cannot be ruled out "
            "by computer vision alone. Do not rely exclusively on AI classification to establish food safety for consumption."
        ),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Fruit Freshness from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to fruit image")
    parser.add_argument("--model", type=str, default="MobileNetV2 (Feature Extraction)", help="Model name or key")
    args = parser.parse_args()

    res = predict_fruit(args.image, model_name_or_key=args.model)
    if "error" in res:
        print(f"Error: {res['error']}")
    else:
        print(f"\nResult: {res['display_name']} ({res['freshness']})")
        print(f"Confidence: {res['confidence_percentage']}% ({res['confidence_level']})")
        print(f"Model: {res['model_used']}")
        print("\nClass Probabilities:")
        for k, v in res["all_probabilities"].items():
            print(f"  {k:<25}: {v*100:>6.2f}%")
