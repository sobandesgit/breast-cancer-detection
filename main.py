"""
main.py
-------
Entry point for the full breast cancer detection pipeline.

Pipeline steps:
  1. Verify dataset and preview sample images
  2. Train all four models across all four magnification levels
  3. Evaluate all models and generate comparison results

Usage:
    python main.py

Author : Sobande Olukayode Oluwatofunmi (BU22CSC1016)
Supervisor : Dr. Abiodun Moses
Institution: Bowen University, Iwo, Osun State, Nigeria

Dataset citation:
    Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L.,
    A Dataset for Breast Cancer Histopathological Image Classification,
    IEEE Transactions on Biomedical Engineering (TBME),
    63(7):1455-1462, 2016.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.preprocessing.preprocess import (
    get_data_generators,
    plot_sample_images,
    MAGNIFICATIONS
)
from src.training.train import train_all
from src.evaluation.evaluate import compare_all_models


def main():
    print("\n" + "=" * 60)
    print("  Transfer Learning + Explainable AI")
    print("  Early Detection of Breast Cancer")
    print("  BreaKHis Dataset — All Magnification Levels")
    print("  Author: Sobande Olukayode Oluwatofunmi (BU22CSC1016)")
    print("=" * 60)

    # ── Step 1: Verify dataset ─────────────────────────────────────────────
    # Organise raw BreaKHis nested structure into flat directories
from src.preprocessing.preprocess import organise_dataset
organise_dataset()
    print("\n[Step 1] Verifying dataset and previewing samples...")
    for mag in MAGNIFICATIONS:
        print(f"\n  Magnification: {mag}")
        try:
            train_gen, val_gen, test_gen = get_data_generators(mag)
            print(f"    Train      : {train_gen.samples} images")
            print(f"    Validation : {val_gen.samples} images")
            print(f"    Test       : {test_gen.samples} images")
            print(f"    Classes    : {train_gen.class_indices}")
            plot_sample_images(train_gen, mag)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")

    # ── Step 2: Train all models ───────────────────────────────────────────
    print("\n[Step 2] Training all models across all magnification levels...")
    train_all()

    # ── Step 3: Evaluate and compare ──────────────────────────────────────
    print("\n[Step 3] Evaluating all models...")
    compare_all_models()

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Models  : models/saved_models/")
    print("  Metrics : results/metrics/")
    print("  Plots   : results/plots/")
    print("  Grad-CAM: results/gradcam/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()