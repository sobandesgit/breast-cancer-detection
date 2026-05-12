"""
evaluate.py
-----------
Evaluates all trained models on the test set for all magnifications.

As specified in Chapter 3 (Section 3.8), evaluation metrics are:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

Results are saved to results/plots/ and results/metrics/
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, InceptionV3
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from src.preprocessing.preprocess import get_data_generators

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SHAPE      = (224, 224, 3)
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

BASE_MODELS = {
    "VGG16"       : VGG16,
    "ResNet50"    : ResNet50,
    "DenseNet121" : DenseNet121,
    "InceptionV3" : InceptionV3,
}


# ── Rebuild model and load weights ────────────────────────────────────────────
def load_model(model_name: str, magnification: str):
    """
    Rebuilds the model architecture and loads saved weights.
    """
    weights_path = (
        f"models/saved_models/{model_name}_{magnification}_weights.weights.h5"
    )

    if not os.path.exists(weights_path):
        print(f"  [SKIP] Weights not found: {weights_path}")
        return None

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
    print(f"  Loaded: {weights_path}")

    return model


# ── Evaluate single model ─────────────────────────────────────────────────────
def evaluate_model(model_name: str, magnification: str):
    """
    Evaluates a single model on the test set for a given magnification.
    """
    print(f"\n── Evaluating: {model_name} | {magnification} ──")

    model = load_model(model_name, magnification)
    if model is None:
        return None

    _, _, test_gen = get_data_generators(magnification)
    test_gen.reset()

    y_pred_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)
    y_true      = test_gen.classes

    metrics = {
        "Model"         : model_name,
        "Magnification" : magnification,
        "Accuracy"      : round(accuracy_score(y_true, y_pred), 4),
        "Precision"     : round(precision_score(y_true, y_pred), 4),
        "Recall"        : round(recall_score(y_true, y_pred), 4),
        "F1-Score"      : round(f1_score(y_true, y_pred), 4),
        "ROC-AUC"       : round(roc_auc_score(y_true, y_pred_prob), 4),
    }

    print(f"  Accuracy  : {metrics['Accuracy']}")
    print(f"  Precision : {metrics['Precision']}")
    print(f"  Recall    : {metrics['Recall']}")
    print(f"  F1-Score  : {metrics['F1-Score']}")
    print(f"  ROC-AUC   : {metrics['ROC-AUC']}")

    plot_confusion_matrix(model_name, magnification, y_true, y_pred)
    plot_roc_curve(model_name, magnification, y_true, y_pred_prob)

    # Free memory
    del model
    tf.keras.backend.clear_session()

    return metrics


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(model_name, magnification, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.title(f"Confusion Matrix — {model_name} | {magnification}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    save_path = (
        f"results/plots/{model_name}_{magnification}_confusion_matrix.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


# ── ROC Curve ─────────────────────────────────────────────────────────────────
def plot_roc_curve(model_name, magnification, y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name} | {magnification}")
    plt.legend()
    plt.tight_layout()
    save_path = (
        f"results/plots/{model_name}_{magnification}_roc_curve.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {save_path}")


# ── Compare all models ────────────────────────────────────────────────────────
def compare_all_models():
    """
    Evaluates all models across all magnifications.
    Saves comparison table, heatmap and bar chart.
    """
    all_metrics = []

    for mag in MAGNIFICATIONS:
        for model_name in BASE_MODELS:
            result = evaluate_model(model_name, mag)
            if result:
                all_metrics.append(result)

    if not all_metrics:
        print("\nNo models evaluated. Check weights files.")
        return

    df = pd.DataFrame(all_metrics)
    print("\n── Full Model Comparison ────────────────────────────")
    print(df.to_string(index=False))

    # Save CSV
    csv_path = "results/metrics/full_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved: {csv_path}")

    # Accuracy heatmap
    pivot = df.pivot(
        index="Magnification", columns="Model", values="Accuracy"
    )
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".4f",
        cmap="YlOrRd", linewidths=0.5
    )
    plt.title("Accuracy Heatmap — All Models × All Magnifications")
    plt.tight_layout()
    plt.savefig("results/plots/accuracy_heatmap.png", dpi=150)
    plt.close()
    print("Accuracy heatmap saved.")

    # Bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for ax, metric in zip(axes, metrics_list):
        pivot_m = df.pivot(
            index="Model", columns="Magnification", values=metric
        )
        pivot_m.plot(
            kind="bar", ax=ax, colormap="Set2",
            legend=(metric == "Accuracy")
        )
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=15, fontsize=8
        )

    plt.suptitle(
        "Model Performance Across All Magnifications", fontsize=14
    )
    plt.tight_layout()
    plt.savefig("results/plots/full_comparison_chart.png", dpi=150)
    plt.close()
    print("Full comparison chart saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_all_models()