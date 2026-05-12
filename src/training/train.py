"""
train.py
--------
Trains all four transfer learning models on the BreaKHis dataset.

As specified in Chapter 3 (Section 3.6):
  - Models are trained on all four magnification levels
  - Each model is trained and saved separately
  - Best model per magnification is saved based on val_accuracy

Implementation decisions (to document in Chapter 4):
  - EPOCHS = 10 per model per magnification
  - Callbacks: EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from src.preprocessing.preprocess import get_data_generators
MAGNIFICATIONS = ["400X"]
from src.training.models import build_model, BASE_MODELS

# ── Constants ─────────────────────────────────────────────────────────────────
EPOCHS = 10


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(model_name: str, magnification: str):
    """
    Returns training callbacks for a given model and magnification.

    Callbacks used:
      - ModelCheckpoint : saves the best model based on val_accuracy
      - EarlyStopping   : stops training if val_loss does not improve
      - ReduceLROnPlateau: reduces learning rate when val_loss plateaus
      - CSVLogger       : logs training metrics to a CSV file
    """
    save_path = (
        f"models/saved_models/{model_name}_{magnification}_best.h5"
    )
    log_path = (
        f"results/metrics/{model_name}_{magnification}_training_log.csv"
    )

    return [
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            filename=log_path,
            append=False
        )
    ]


# ── Train single model ────────────────────────────────────────────────────────
def train_model(model_name: str, magnification: str):
    """
    Trains a single model on a single magnification level.

    Args:
        model_name    : one of 'VGG16', 'ResNet50', 'DenseNet121', 'InceptionV3'
        magnification : one of '40X', '100X', '200X', '400X'

    Returns:
        training history object
    """
    print(f"\n{'#'*60}")
    print(f"  Training : {model_name}  |  Magnification : {magnification}")
    print(f"{'#'*60}\n")

    train_gen, val_gen, _ = get_data_generators(magnification)
    model = build_model(model_name)
    callbacks = get_callbacks(model_name, magnification)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model as well
    final_path = (
        f"models/saved_models/{model_name}_{magnification}_final.h5"
    )
    model.save(final_path)
    print(f"\n  Saved : {final_path}")

    return history


# ── Train all models across all magnifications ────────────────────────────────
def train_all():
    """
    Trains all four models across all four magnification levels.
    Saves all models and training logs to results/metrics/

    Returns:
        dict of histories keyed by (model_name, magnification)
    """
    histories = {}

    for mag in MAGNIFICATIONS:
        for model_name in BASE_MODELS:
            key = f"{model_name}_{mag}"
            try:
                histories[key] = train_model(model_name, mag)
            except FileNotFoundError as e:
                print(f"\n  [SKIP] {key} — {e}")

    print(f"\n{'='*60}")
    print("  All models trained successfully.")
    print(f"{'='*60}\n")

    return histories


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_all()