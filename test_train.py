"""
test_train.py
-------------
Quick test to verify VGG16 trains correctly on 400X images
before running the full pipeline.
Trains for 2 epochs only.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.preprocessing.preprocess import get_data_generators
from src.training.models import build_model


def test_train():
    print("\n" + "="*55)
    print("  Test Training: VGG16 on 400X — 2 epochs only")
    print("="*55 + "\n")

    train_gen, val_gen, _ = get_data_generators("400X")
    model = build_model("VGG16")

    history = model.fit(
        train_gen,
        epochs=2,
        validation_data=val_gen,
        verbose=1
    )

    print("\n" + "="*55)
    print("  Test complete.")
    print(f"  Final train accuracy : {history.history['accuracy'][-1]:.4f}")
    print(f"  Final val accuracy   : {history.history['val_accuracy'][-1]:.4f}")
    print("="*55 + "\n")


if __name__ == "__main__":
    test_train()