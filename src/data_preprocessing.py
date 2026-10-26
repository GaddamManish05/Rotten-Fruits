"""
FreshGuard AI - Data Preprocessing & Augmentation Pipeline
Handles dataset loading, architecture-specific normalization, augmentation,
and batch generation using tf.data pipelines.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

import config


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Creates data augmentation pipeline applied exclusively to training images.
    Includes rotation, zoom, horizontal flip, and translation.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=config.RANDOM_SEED),
        layers.RandomRotation(0.15, seed=config.RANDOM_SEED),
        layers.RandomZoom(0.15, seed=config.RANDOM_SEED),
        layers.RandomTranslation(0.1, 0.1, seed=config.RANDOM_SEED),
        layers.RandomContrast(0.1, seed=config.RANDOM_SEED),
    ], name="data_augmentation")


def get_preprocessing_layer(model_type: str = "mobilenet") -> tf.keras.layers.Layer:
    """
    Returns the appropriate preprocessing/normalization function for each model architecture.
    - scratch: Rescales [0, 255] -> [0, 1]
    - mobilenet: tf.keras.applications.mobilenet_v2.preprocess_input -> maps [0, 255] to [-1, 1]
    - efficientnet: tf.keras.applications.efficientnet.preprocess_input -> internally scales
    """
    model_type = model_type.lower()
    if model_type == "scratch":
        return layers.Rescaling(1.0 / 255.0, name="rescaling_scratch")
    elif "mobilenet" in model_type:
        return layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling_mobilenet")
    elif "efficientnet" in model_type:
        return layers.Rescaling(scale=1.0, name="rescaling_efficientnet")
    else:
        # Default fallback to 0-1 scaling
        return layers.Rescaling(1.0 / 255.0, name="default_rescaling")


def load_dataset_from_directory(
    directory: Path,
    image_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    class_names: Optional[List[str]] = None,
) -> Tuple[tf.data.Dataset, List[str]]:
    """
    Loads an image directory as a tf.data.Dataset using categorical labels.
    Returns (dataset, class_names).
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=config.RANDOM_SEED if shuffle else None,
        interpolation="bilinear",
    )
    detected_classes = dataset.class_names
    return dataset, detected_classes


def create_train_pipeline(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    apply_augmentation: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Builds optimized training and validation tf.data pipelines with caching and prefetching.
    """
    train_ds, class_names = load_dataset_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds, _ = load_dataset_from_directory(
        directory=val_dir,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names,
    )

    autotune = tf.data.AUTOTUNE

    if apply_augmentation:
        aug_layer = get_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=autotune
        )

    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def create_test_pipeline(
    test_dir: Path,
    class_names: List[str],
    batch_size: int = config.BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Builds evaluation pipeline for the held-out test set.
    Preserves exact order (shuffle=False) for reproducible metrics and confusion matrix.
    """
    test_ds, _ = load_dataset_from_directory(
        directory=test_dir,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names,
    )
    return test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def preprocess_single_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
) -> np.ndarray:
    """
    Prepares a single PIL image for model prediction.
    Resizes to target_size, converts to RGB, and returns array with shape (1, H, W, 3) in [0, 255].
    The model's internal preprocessing layer will handle architecture-specific scaling.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch
