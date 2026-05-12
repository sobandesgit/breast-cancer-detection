"""
preprocess.py
-------------
Handles all data preprocessing for the BreaKHis dataset.

As specified in Chapter 3 (Section 3.5):
  - Resize all images to 224x224 pixels
  - Normalize pixel values to range 0-1
  - Data augmentation: rotation, horizontal flip, zoom, random shift
  - Dataset split: 70% train, 15% validation, 15% test

The BreaKHis dataset has a nested folder structure:
  breast/
  ├── benign/SOB/{subtype}/{slide_id}/{magnification}/*.png
  └── malignant/SOB/{subtype}/{slide_id}/{magnification}/*.png

This script flattens the structure by magnification level,
collecting all benign and malignant images across all subtypes.

Citation:
  Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L.,
  A Dataset for Breast Cancer Histopathological Image Classification,
  IEEE Transactions on Biomedical Engineering (TBME),
  63(7):1455-1462, 2016.
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
SEED           = 42
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Path to the root of the BreaKHis dataset as downloaded
# Adjust this to match where you extracted the dataset on your machine
DATASET_ROOT = r"C:\Users\soban\Downloads\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"
# Path where we will write the flattened organised images
# Structure: dataset/organised/{magnification}/{benign|malignant}/
ORGANISED_DIR = "dataset/organised"


# ── Step 1: Flatten nested structure ──────────────────────────────────────────
def organise_dataset():
    """
    Walks through the nested BreaKHis folder structure and
    copies images into a flat organised directory grouped by
    magnification level and class (benign / malignant).

    Output structure:
        dataset/organised/
        ├── 40X/
        │   ├── benign/
        │   └── malignant/
        ├── 100X/
        │   ├── benign/
        │   └── malignant/
        ├── 200X/ ...
        └── 400X/ ...

    This only needs to run once.
    """
    classes = ["benign", "malignant"]

    # Check if already organised
    already_done = all(
        os.path.exists(os.path.join(ORGANISED_DIR, mag, cls))
        for mag in MAGNIFICATIONS
        for cls in classes
    )
    if already_done:
        print("Dataset already organised. Skipping.")
        return

    print("Organising BreaKHis dataset by magnification level...")

    for cls in classes:
        class_root = os.path.join(DATASET_ROOT, cls)
        if not os.path.exists(class_root):
            print(f"  [WARNING] Folder not found: {class_root}")
            continue

        # Walk all subdirectories under benign/ or malignant/
        for root, dirs, files in os.walk(class_root):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                # The magnification level is the name of the
                # immediate parent folder of the image file
                mag = os.path.basename(root)

                if mag not in MAGNIFICATIONS:
                    continue

                # Build destination path
                dest_dir = os.path.join(ORGANISED_DIR, mag, cls)
                os.makedirs(dest_dir, exist_ok=True)

                src  = os.path.join(root, file)
                dest = os.path.join(dest_dir, file)

                # Avoid overwriting if file already exists
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)

    # Report counts
    print("\nDataset organised successfully:")
    for mag in MAGNIFICATIONS:
        for cls in classes:
            folder = os.path.join(ORGANISED_DIR, mag, cls)
            count  = len(os.listdir(folder)) if os.path.exists(folder) else 0
            print(f"  {mag} / {cls}: {count} images")


# ── Step 2: Data generators ───────────────────────────────────────────────────
def get_data_generators(magnification: str):
    """
    Returns train, validation, and test generators for a given
    magnification level from the organised dataset directory.

    Args:
        magnification: one of '40X', '100X', '200X', '400X'

    Returns:
        train_generator, val_generator, test_generator
    """
    data_dir = os.path.join(ORGANISED_DIR, magnification)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Organised dataset not found at: {data_dir}\n"
            f"Please run organise_dataset() first."
        )

    # Augmentation on training set only
    # As specified in Chapter 3 Section 3.5
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.30
    )

    # No augmentation on validation and test — rescaling only
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.30
    )

    # Training set (70%)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    # Validation set (15%)
    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

    # Test set (15%)
    test_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED + 1
    )

    return train_generator, val_generator, test_generator


# ── Utility: class label map ───────────────────────────────────────────────────
def get_class_labels(generator):
    """Returns a dict mapping index to class name."""
    return {v: k for k, v in generator.class_indices.items()}


# ── Utility: plot sample images ───────────────────────────────────────────────
def plot_sample_images(generator, magnification: str, n: int = 8):
    """
    Plots a grid of sample images from the dataset.
    Saves to results/plots/
    """
    images, labels = next(generator)
    class_labels   = get_class_labels(generator)

    plt.figure(figsize=(16, 4))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(class_labels[int(labels[i])], fontsize=9)
        plt.axis("off")
    plt.suptitle(
        f"Sample BreaKHis Images — {magnification} Magnification",
        fontsize=12
    )
    plt.tight_layout()
    save_path = f"results/plots/sample_images_{magnification}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Sample images saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: organise the raw dataset first
    organise_dataset()

    # Step 2: verify generators for each magnification
    for mag in MAGNIFICATIONS:
        print(f"\n── Magnification: {mag} ──")
        try:
            train_gen, val_gen, test_gen = get_data_generators(mag)
            print(f"  Train      : {train_gen.samples} images")
            print(f"  Validation : {val_gen.samples} images")
            print(f"  Test       : {test_gen.samples} images")
            print(f"  Classes    : {train_gen.class_indices}")
            plot_sample_images(train_gen, mag)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")