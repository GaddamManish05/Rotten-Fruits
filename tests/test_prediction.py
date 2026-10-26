"""
FreshGuard AI - Unit Tests: Prediction & Model Integration
Validates model output format, probability summation, confidence bounds, and error handling.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.model_factory import build_scratch_cnn
from src.predict import predict_fruit


@pytest.fixture
def dummy_scratch_model():
    """Builds an un-trained scratch CNN model for inference testing."""
    model = build_scratch_cnn(num_classes=6)
    return model


def test_predict_fruit_output_structure(dummy_scratch_model):
    """Verifies that predict_fruit returns all required keys and correct types."""
    test_img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    result = predict_fruit(test_img, model=dummy_scratch_model, model_name_or_key="TestModel")

    assert result["success"] is True
    assert "predicted_class" in result
    assert "display_name" in result
    assert "freshness" in result
    assert result["freshness"] in ["Fresh", "Rotten", "Unknown"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "confidence_percentage" in result
    assert "confidence_level" in result
    assert result["confidence_level"] in ["High", "Moderate", "Low"]
    assert "all_probabilities" in result
    assert len(result["all_probabilities"]) == 6
    assert "food_safety_advisory" in result


def test_probabilities_sum_to_one(dummy_scratch_model):
    """Verifies that the predicted class probabilities form a valid probability distribution."""
    test_img = Image.new("RGB", (224, 224), color=(30, 200, 30))
    result = predict_fruit(test_img, model=dummy_scratch_model, model_name_or_key="TestModel")

    prob_sum = sum(result["all_probabilities"].values())
    assert prob_sum == pytest.approx(1.0, abs=1e-3)


def test_invalid_input_handling():
    """Verifies graceful handling of non-existent image paths."""
    with pytest.raises(Exception):
        predict_fruit("non_existent_file_path_12345.jpg")


def test_predict_fruit_with_gradcam(dummy_scratch_model):
    """Verifies predict_fruit with compute_gradcam=True executes safely without raising."""
    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    result = predict_fruit(test_img, model=dummy_scratch_model, model_name_or_key="TestModel", compute_gradcam=True)
    assert result["success"] is True
    assert "gradcam_image" in result
