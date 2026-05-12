"""
models.py
---------
Builds transfer learning models as specified in Chapter 3 (Section 3.6).

Pre-trained models used (all trained on ImageNet):
  - VGG16
  - ResNet50
  - DenseNet121
  - InceptionV3

Implementation:
  - Lower convolutional layers are frozen to retain generic features
  - Top classification layers are replaced with custom layers
    for binary classification (benign / malignant)
  - Upper layers are fine-tuned on BreaKHis data

Note (to document in Chapter 4):
  - All models use input size 224x224 for consistency
  - InceptionV3 was originally designed for 299x299 — this will
    be flagged in the Chapter 4 discussion
  - Number of fine-tune layers and epochs are implementation
    decisions documented in Chapter 4
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, InceptionV3
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SHAPE     = (224, 224, 3)
FINE_TUNE_AT  = 4
LEARNING_RATE = 1e-4

# ── Model Registry ────────────────────────────────────────────────────────────
BASE_MODELS = {
    "VGG16"       : VGG16,
    "ResNet50"    : ResNet50,
    "DenseNet121" : DenseNet121,
    "InceptionV3" : InceptionV3,
}


def build_model(model_name: str):
    """
    Builds and compiles a transfer learning model.

    Args:
        model_name: one of 'VGG16', 'ResNet50', 'DenseNet121', 'InceptionV3'

    Returns:
        Compiled Keras model ready for training
    """
    if model_name not in BASE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(BASE_MODELS.keys())}"
        )

    # ── Step 1: Load pre-trained base model ───────────────────────────────────
    base = BASE_MODELS[model_name](
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE
    )

    # ── Step 2: Build full model first ────────────────────────────────────────
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)

    # ── Step 3: Freeze layers AFTER model is built ────────────────────────────
    # Use model.layers directly — base.layers is a different list
    # VGG16 has 19 base layers + our 6 custom layers = 25 total
    # We freeze everything except the last FINE_TUNE_AT base layers
    # and our custom classification head
    num_base_layers = len(base.layers)  # 19 for VGG16

    for i, layer in enumerate(model.layers):
        if i < num_base_layers - FINE_TUNE_AT:
            layer.trainable = False
        else:
            layer.trainable = True

    frozen    = sum(1 for l in model.layers if not l.trainable)
    trainable = sum(1 for l in model.layers if l.trainable)

    # ── Step 4: Compile AFTER freezing ────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n{'='*55}")
    print(f"  Model                 : {model_name}")
    print(f"  Input                 : {IMG_SHAPE}")
    print(f"  Base layers frozen    : {frozen}")
    print(f"  Base layers trainable : {trainable}")
    print(f"  Total parameters      : {model.count_params():,}")
    print(f"  Trainable parameters  : "
          f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
    print(f"  Non-trainable params  : "
          f"{sum(tf.size(w).numpy() for w in model.non_trainable_weights):,}")
    print(f"{'='*55}\n")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in BASE_MODELS:
        model = build_model(name)
        print(f"{name} built successfully.\n")