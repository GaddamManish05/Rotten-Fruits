"""
FreshGuard AI - Model Factory Module
Constructs model architectures for:
- Experiment A: Small CNN trained from scratch (Baseline)
- Experiment B: MobileNetV2 Feature Extraction (Frozen Base)
- Experiment C: MobileNetV2 Fine-Tuning (Partial Unfreezing with low LR)
- Experiment D: EfficientNetB0 Transfer Learning (Optional Comparison)
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import config
from src.data_preprocessing import get_preprocessing_layer


def build_scratch_cnn(
    num_classes: int = len(config.DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    learning_rate: float = config.LR_SCRATCH,
) -> tf.keras.Model:
    """
    Builds a lightweight CNN architecture trained from scratch with random weight initialization.
    Serves as the empirical baseline to demonstrate the value of transfer learning.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = get_preprocessing_layer("scratch")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.4, seed=config.RANDOM_SEED, name="dropout1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Scratch_CNN_Baseline")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2_feature_extraction(
    num_classes: int = len(config.DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    learning_rate: float = config.LR_FEATURE_EXTRACTION,
) -> tf.keras.Model:
    """
    Builds a Feature Extraction model using pretrained MobileNetV2 on ImageNet.
    The convolutional backbone is completely frozen (trainable = False).
    Only newly initialized classification head layers are trained.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = get_preprocessing_layer("mobilenet")(inputs)

    # Pretrained MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze all feature extraction weights

    # Call base model with training=False to ensure BatchNorm behaves in inference mode
    x = base_model(x, training=False)

    # Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, seed=config.RANDOM_SEED, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.2, seed=config.RANDOM_SEED, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Feature_Extraction")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2_finetuning(
    base_fe_model: Optional[tf.keras.Model] = None,
    num_classes: int = len(config.DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    unfreeze_from_layer: int = 100,
    learning_rate: float = config.LR_FINETUNING,
) -> tf.keras.Model:
    """
    Builds a Fine-Tuning model starting from MobileNetV2.
    Unfreezes deeper layers (from index unfreeze_from_layer upwards)
    while keeping earlier low-level generic edge/texture detectors frozen.
    Uses a significantly lower learning rate to prevent catastrophic forgetting.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = get_preprocessing_layer("mobilenet")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True

    # Freeze layers prior to unfreeze_from_layer
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    # Freeze BatchNormalization layers across the entire base to preserve ImageNet statistics
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, seed=config.RANDOM_SEED, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.2, seed=config.RANDOM_SEED, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Fine_Tuning")

    # If trained feature extraction model is provided, copy head weights
    if base_fe_model is not None:
        for layer_name in ["head_bn", "head_dense", "classification_head"]:
            try:
                src_l = base_fe_model.get_layer(layer_name)
                tgt_l = model.get_layer(layer_name)
                tgt_l.set_weights(src_l.get_weights())
            except Exception:
                pass

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb0(
    num_classes: int = len(config.DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    learning_rate: float = config.LR_FEATURE_EXTRACTION,
) -> tf.keras.Model:
    """
    Builds an EfficientNetB0 feature extraction transfer learning model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = get_preprocessing_layer("efficientnet")(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, seed=config.RANDOM_SEED, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Transfer_Learning")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_parameters(model: tf.keras.Model) -> dict:
    """Returns total, trainable, and non-trainable parameter counts."""
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }
