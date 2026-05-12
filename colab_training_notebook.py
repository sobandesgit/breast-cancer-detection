# ── Step 1: Mount Google Drive ─────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Step 2: Set up folder structure ───────────────────────────────────────
import os

os.makedirs('/content/models/saved_models', exist_ok=True)
os.makedirs('/content/results/metrics', exist_ok=True)
os.makedirs('/content/results/plots', exist_ok=True)

# ── Step 3: Imports ────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, InceptionV3
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {tf.config.list_physical_devices('GPU')}")

# ── Step 4: Constants ──────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
IMG_SHAPE     = (224, 224, 3)
BATCH_SIZE    = 32
SEED          = 42
EPOCHS        = 30
LEARNING_RATE = 1e-4
DRIVE_BASE    = '/content/drive/MyDrive/breast_cancer'
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

BASE_MODELS = {
    'VGG16'       : VGG16,
    'ResNet50'    : ResNet50,
    'DenseNet121' : DenseNet121,
    'InceptionV3' : InceptionV3,
}

# Last 20% of layers unfrozen per model for fair comparison
FINE_TUNE_AT = {
    'VGG16'       : 4,
    'ResNet50'    : 35,
    'DenseNet121' : 85,
    'InceptionV3' : 62,
}


# ── Step 5: Data generators ────────────────────────────────────────────────
def get_data_generators(magnification: str):
    data_dir = f'{DRIVE_BASE}/{magnification}'

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

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.30
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED
    )

    test_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED + 1
    )

    return train_generator, val_generator, test_generator


# ── Step 6: Build model ────────────────────────────────────────────────────
def build_model(model_name: str):
    base = BASE_MODELS[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SHAPE
    )

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=output)

    num_base_layers = len(base.layers)
    fine_tune_at    = FINE_TUNE_AT[model_name]
    for i, layer in enumerate(model.layers):
        if i < num_base_layers - fine_tune_at:
            layer.trainable = False
        else:
            layer.trainable = True

    frozen    = sum(1 for l in model.layers if not l.trainable)
    trainable = sum(1 for l in model.layers if l.trainable)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n{'='*55}")
    print(f"  Model                 : {model_name}")
    print(f"  Base layers frozen    : {frozen}")
    print(f"  Base layers trainable : {trainable}")
    print(f"  Total parameters      : {model.count_params():,}")
    print(f"  Trainable parameters  : "
          f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
    print(f"  Non-trainable params  : "
          f"{sum(tf.size(w).numpy() for w in model.non_trainable_weights):,}")
    print(f"{'='*55}\n")

    return model


# ── Step 7: Train all models across all magnifications ────────────────────
all_histories = {}

for mag in MAGNIFICATIONS:
    dataset_dir = f'{DRIVE_BASE}/{mag}'
    if not os.path.exists(dataset_dir):
        print(f"\n[SKIP] {mag} not found in Drive")
        continue

    print(f"\n{'='*60}")
    print(f"  MAGNIFICATION: {mag}")
    print(f"{'='*60}")

    train_gen, val_gen, test_gen = get_data_generators(mag)

    # Class weights for imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_gen.classes
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Class weights: {class_weight_dict}")

    for model_name in BASE_MODELS:
        key = f"{model_name}_{mag}"
        print(f"\n{'#'*60}")
        print(f"  Training: {model_name} | {mag}")
        print(f"{'#'*60}\n")

        model = build_model(model_name)

        callbacks = [
            ModelCheckpoint(
                filepath=f'/content/models/saved_models/{model_name}_{mag}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                filename=f'/content/results/metrics/{model_name}_{mag}_training_log.csv',
                append=False
            )
        ]

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        model.save(
            f'/content/models/saved_models/{model_name}_{mag}_final.keras'
        )
        all_histories[key] = history

        best_val = max(history.history['val_accuracy'])
        print(f"\n  {key} complete.")
        print(f"  Best val accuracy: {best_val:.4f}")

        # Save to Drive immediately after each model
        import shutil
        drive_output = f'{DRIVE_BASE}/trained_models_all_magnifications'
        os.makedirs(drive_output, exist_ok=True)
        for f in os.listdir('/content/models/saved_models'):
            shutil.copy(
                f'/content/models/saved_models/{f}',
                f'{drive_output}/{f}'
            )
        print(f"  Models backed up to Drive.")

print(f"\n{'='*60}")
print("  All models trained successfully.")
print(f"{'='*60}")

# ── Step 8: Plot training curves ───────────────────────────────────────────
for key, history in all_histories.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{key} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{key} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f'/content/results/plots/{key}_training_curve.png',
        dpi=150
    )
    plt.close()
    print(f"Training curve saved: {key}")

# ── Step 9: Save all results to Google Drive ───────────────────────────────
import shutil

drive_results = f'{DRIVE_BASE}/results_all_magnifications'
os.makedirs(drive_results, exist_ok=True)
shutil.copytree('/content/results', drive_results, dirs_exist_ok=True)

print("\nAll results copied to Google Drive.")
print(f"Models  : {DRIVE_BASE}/trained_models_all_magnifications")
print(f"Results : {drive_results}")