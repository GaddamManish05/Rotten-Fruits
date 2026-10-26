"""
FreshGuard AI - Unit Tests: Preprocessing & Augmentation
Validates image resizing, architecture-specific normalizations, and augmentation pipelines.
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
from src.data_preprocessing import (
    get_augmentation_layer,
    get_preprocessing_layer,
    preprocess_single_image,
)


def test_augmentation_layer_shape():
    """Verifies that augmentation preserves tensor dimensions (B, 224, 224, 3)."""
    aug = get_augmentation_layer()
    sample_tensor = tf.random.uniform([4, 224, 224, 3], minval=0, maxval=255, dtype=tf.float32)
    augmented = aug(sample_tensor, training=True)
    assert augmented.shape == (4, 224, 224, 3), f"Expected (4, 224, 224, 3), got {augmented.shape}"


def test_scratch_preprocessing_normalization():
    """Verifies that Scratch CNN preprocessing scales pixel values from [0, 255] to [0, 1]."""
    prep = get_preprocessing_layer("scratch")
    sample_tensor = tf.constant([[[[0.0, 127.5, 255.0]]]])
    scaled = prep(sample_tensor).numpy()
    assert scaled[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-4)
    assert scaled[0, 0, 0, 1] == pytest.approx(0.5, abs=1e-2)
    assert scaled[0, 0, 0, 2] == pytest.approx(1.0, abs=1e-4)


def test_mobilenet_preprocessing_normalization():
    """Verifies that MobileNetV2 preprocessing maps pixel values from [0, 255] to [-1, 1]."""
    prep = get_preprocessing_layer("mobilenet")
    sample_tensor = tf.constant([[[[0.0, 127.5, 255.0]]]])
    scaled = prep(sample_tensor).numpy()
    assert scaled[0, 0, 0, 0] == pytest.approx(-1.0, abs=1e-2)
    assert scaled[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-2)
    assert scaled[0, 0, 0, 2] == pytest.approx(1.0, abs=1e-2)


def test_preprocess_single_image():
    """Verifies single PIL image preprocessing returns (1, 224, 224, 3) float32 array."""
    test_img = Image.new("RGB", (640, 480), color=(255, 100, 50))
    batch = preprocess_single_image(test_img)
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() <= 255.0
    assert 0.0 <= batch.max() <= 255.0


def test_grayscale_to_rgb_conversion():
    """Verifies single grayscale PIL image is automatically converted to 3 RGB channels."""
    gray_img = Image.new("L", (100, 100), color=128)
    batch = preprocess_single_image(gray_img)
    assert batch.shape == (1, 224, 224, 3)
