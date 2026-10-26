"""
FreshGuard AI - Grad-CAM (Gradient-Weighted Class Activation Mapping)
Provides visual explainability for CNN and Transfer Learning decisions.
Visualizes which regions (e.g. rot spots, texture blemishes, fruit skin)
the model focused on to make its classification.
"""

from typing import Optional, Tuple
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import tensorflow as tf

import config


def find_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """Finds the name of the last 4D convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # Check nested models (like MobileNetV2 base)
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or "conv" in sub_layer.name.lower():
                    return sub_layer.name
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    return None


def generate_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    pred_index: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Computes Grad-CAM heatmap for an input image and a trained Keras model.
    """
    try:
        # Determine the base feature extractor sub-model or top conv layer
        mobilenet_submodel = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
                mobilenet_submodel = layer
                break

        if mobilenet_submodel is not None:
            # We construct a gradient model with the submodel's last conv layer
            last_conv_name = "Conv_1"
            if not any(l.name == last_conv_name for l in mobilenet_submodel.layers):
                for l in reversed(mobilenet_submodel.layers):
                    if "conv" in l.name.lower():
                        last_conv_name = l.name
                        break

            # Build sub-model to output conv activations
            conv_layer = mobilenet_submodel.get_layer(last_conv_name)
            sub_grad_model = tf.keras.models.Model(
                inputs=mobilenet_submodel.inputs,
                outputs=[conv_layer.output, mobilenet_submodel.output]
            )

            # Get remaining layers after submodel
            submodel_idx = model.layers.index(mobilenet_submodel)
            remaining_layers = model.layers[submodel_idx + 1:]

            # Forward pass to submodel
            # First apply preprocessing layers before submodel
            x = img_array
            for l in model.layers[:submodel_idx]:
                x = l(x)

            with tf.GradientTape() as tape:
                conv_outputs, sub_out = sub_grad_model(x)
                tape.watch(conv_outputs)
                # Pass through remaining head layers
                head_out = sub_out
                for l in remaining_layers:
                    head_out = l(head_out)
                if pred_index is None:
                    pred_index = tf.argmax(head_out[0])
                class_channel = head_out[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)

        else:
            # Scratch CNN direct architecture
            last_conv_name = "conv4_1"
            if not any(l.name == last_conv_name for l in model.layers):
                last_conv_name = find_last_conv_layer(model)
            if last_conv_name is None:
                return None

            last_conv_layer = model.get_layer(last_conv_name)
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            return None

        # Pool gradients over spatial dimensions (Global Average Pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv feature maps by corresponding gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only features that positively influence the target class
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    except Exception as e:
        print(f"Grad-CAM computation note: {e}")
        return None


def overlay_heatmap_on_image(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlays 2D Grad-CAM heatmap onto original PIL image.
    """
    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap(colormap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    # Convert to PIL
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize(original_img.size, Image.Resampling.BILINEAR)

    # Convert original to RGB
    if original_img.mode != "RGB":
        original_img = original_img.convert("RGB")

    # Blend original and heatmap
    blended = Image.blend(original_img, jet_heatmap, alpha=alpha)
    return blended
