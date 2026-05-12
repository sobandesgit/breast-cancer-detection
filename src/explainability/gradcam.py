"""
gradcam.py
----------
Implements Grad-CAM (Gradient-weighted Class Activation Mapping)
as specified in Chapter 3 (Section 3.9).

Grad-CAM generates heatmaps that highlight the regions of an input
image that contributed most to the model's prediction.

All four magnification levels are supported: 40X, 100X, 200X, 400X.

Reference:
  Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization, ICCV 2017.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, InceptionV3
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
IMG_SHAPE      = (224, 224, 3)
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

BASE_MODELS = {
    "VGG16"       : VGG16,
    "ResNet50"    : ResNet50,
    "DenseNet121" : DenseNet121,
    "InceptionV3" : InceptionV3,
}


# ── Load model from weights ───────────────────────────────────────────────────
def load_model_from_weights(model_name: str, magnification: str):
    """
    Rebuilds model architecture and loads saved weights
    for a given model and magnification level.
    """
    weights_path = (
        f"models/saved_models/"
        f"{model_name}_{magnification}_weights.weights.h5"
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    base = BASE_MODELS[model_name](
        weights=None,
        include_top=False,
        input_shape=IMG_SHAPE
    )

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.load_weights(weights_path)

    print(f"  Model loaded: {model_name} | {magnification}")
    return model


# ── Find last conv layer ──────────────────────────────────────────────────────
def get_last_conv_layer(model):
    """
    Finds the name of the last Conv2D layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


# ── Generate Grad-CAM heatmap ─────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a given image.

    Args:
        img_array            : preprocessed image (1, 224, 224, 3)
        model                : trained Keras model
        last_conv_layer_name : name of the last convolutional layer

    Returns:
        heatmap as numpy array (H, W), values 0-1
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ── Overlay heatmap on image ──────────────────────────────────────────────────
def overlay_heatmap(heatmap, img_path, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    img              = cv2.imread(img_path)
    img              = cv2.resize(img, IMG_SIZE)
    heatmap_resized  = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored  = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    superimposed = cv2.addWeighted(
        img, 1 - alpha, heatmap_colored, alpha, 0
    )
    return superimposed


# ── Full Grad-CAM pipeline ────────────────────────────────────────────────────
def save_gradcam(
    model_name: str,
    img_path: str,
    magnification: str,
    model=None,
    save_dir="results/gradcam"
):
    """
    Full Grad-CAM pipeline:
      1. Load and preprocess image
      2. Get model prediction
      3. Generate heatmap
      4. Overlay on original image
      5. Save side-by-side figure

    Args:
        model_name    : e.g. 'DenseNet121'
        img_path      : path to input image
        magnification : e.g. '400X'
        model         : optional pre-loaded model
        save_dir      : directory to save output

    Returns:
        path to saved Grad-CAM figure
    """
    os.makedirs(save_dir, exist_ok=True)

    if model is None:
        model = load_model_from_weights(model_name, magnification)

    # Preprocess image
    img       = tf.keras.preprocessing.image.load_img(
        img_path, target_size=IMG_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    pred_prob  = float(model.predict(img_array, verbose=0)[0][0])
    label      = "Malignant" if pred_prob > 0.5 else "Benign"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    # Generate heatmap
    last_conv    = get_last_conv_layer(model)
    heatmap      = make_gradcam_heatmap(img_array, model, last_conv)
    superimposed = overlay_heatmap(heatmap, img_path)

    # Plot
    original = cv2.cvtColor(
        cv2.resize(cv2.imread(img_path), IMG_SIZE),
        cv2.COLOR_BGR2RGB
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"Overlay\nPrediction: {label} ({confidence:.2%})"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Explanation — {model_name} | {magnification}",
        fontsize=13
    )
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = (
        f"{save_dir}/{model_name}_{magnification}_{base_name}_gradcam.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Grad-CAM saved : {save_path}")
    print(f"  Prediction     : {label} ({confidence:.2%})")

    return save_path


# ── Run Grad-CAM on sample images ─────────────────────────────────────────────
def run_gradcam_on_samples(
    model_name: str = "DenseNet121",
    magnification: str = "400X",
    n_samples: int = 20
):
    """
    Runs Grad-CAM on sample images from both classes
    for a given model and magnification level.
    """
    print(f"\nRunning Grad-CAM: {model_name} | {magnification}...")
    model   = load_model_from_weights(model_name, magnification)
    classes = ["benign", "malignant"]

    for cls in classes:
        img_dir = f"dataset/organised/{magnification}/{cls}"
        if not os.path.exists(img_dir):
            print(f"  [SKIP] {img_dir} not found")
            continue

        images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ][:n_samples]

        print(f"\n  Processing {cls} samples...")
        for img_file in images:
            img_path = os.path.join(img_dir, img_file)
            save_gradcam(model_name, img_path, magnification, model)

    del model
    tf.keras.backend.clear_session()
    print(f"\nGrad-CAM complete: {model_name} | {magnification}")


# ── Run all models across all magnifications ──────────────────────────────────
def run_all_gradcam(n_samples: int = 20):
    """
    Runs Grad-CAM for all 4 models across all 4 magnifications.
    Total: 16 combinations × 20 benign + 20 malignant = 640 heatmaps
    """
    for mag in MAGNIFICATIONS:
        for model_name in BASE_MODELS:
            try:
                run_gradcam_on_samples(model_name, mag, n_samples)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_gradcam(n_samples=20)