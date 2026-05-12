# ── Full Evaluation: All 16 Models ────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, InceptionV3
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import os, shutil

DRIVE_BASE     = '/content/drive/MyDrive/breast_cancer'
MAGNIFICATIONS = ['40X', '100X', '200X', '400X']
IMG_SIZE       = (224, 224)
IMG_SHAPE      = (224, 224, 3)
BATCH_SIZE     = 32
SEED           = 42

BASE_MODELS = {
    'VGG16'       : VGG16,
    'ResNet50'    : ResNet50,
    'DenseNet121' : DenseNet121,
    'InceptionV3' : InceptionV3,
}

os.makedirs('/content/eval_results/plots', exist_ok=True)
os.makedirs('/content/eval_results/metrics', exist_ok=True)


# ── Rebuild model and load weights ────────────────────────────────────────────
def load_model(model_name, magnification):
    model_path = (
        f'{DRIVE_BASE}/trained_models_all_magnifications/'
        f'{model_name}_{magnification}_best.keras'
    )
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    model = tf.keras.models.load_model(model_path)
    return model


# ── Get test generator ────────────────────────────────────────────────────────
def get_test_generator(magnification):
    data_dir = f'{DRIVE_BASE}/{magnification}'

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.30
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
    return test_generator


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(model_name, mag, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.title(f'Confusion Matrix — {model_name} | {mag}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = f'/content/eval_results/plots/{model_name}_{mag}_confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.close()


# ── Plot ROC curve ────────────────────────────────────────────────────────────
def plot_roc_curve(model_name, mag, y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name} | {mag}')
    plt.legend()
    plt.tight_layout()
    path = f'/content/eval_results/plots/{model_name}_{mag}_roc_curve.png'
    plt.savefig(path, dpi=150)
    plt.close()


# ── Evaluate all models ───────────────────────────────────────────────────────
all_metrics = []

for mag in MAGNIFICATIONS:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {mag}")
    print(f"{'='*60}")

    test_gen = get_test_generator(mag)

    for model_name in BASE_MODELS:
        print(f"\n  ── {model_name} | {mag} ──")

        model = load_model(model_name, mag)
        if model is None:
            continue

        test_gen.reset()
        y_pred_prob = model.predict(test_gen, verbose=1).flatten()
        y_pred      = (y_pred_prob > 0.5).astype(int)
        y_true      = test_gen.classes

        metrics = {
            'Model'         : model_name,
            'Magnification' : mag,
            'Accuracy'      : round(accuracy_score(y_true, y_pred), 4),
            'Precision'     : round(precision_score(y_true, y_pred), 4),
            'Recall'        : round(recall_score(y_true, y_pred), 4),
            'F1-Score'      : round(f1_score(y_true, y_pred), 4),
            'ROC-AUC'       : round(roc_auc_score(y_true, y_pred_prob), 4),
        }

        print(f"    Accuracy  : {metrics['Accuracy']}")
        print(f"    Precision : {metrics['Precision']}")
        print(f"    Recall    : {metrics['Recall']}")
        print(f"    F1-Score  : {metrics['F1-Score']}")
        print(f"    ROC-AUC   : {metrics['ROC-AUC']}")

        all_metrics.append(metrics)

        plot_confusion_matrix(model_name, mag, y_true, y_pred)
        plot_roc_curve(model_name, mag, y_true, y_pred_prob)

        # Free memory
        del model
        tf.keras.backend.clear_session()


# ── Save comparison table ─────────────────────────────────────────────────────
df = pd.DataFrame(all_metrics)
print("\n── Full Model Comparison ──────────────────────────────────")
print(df.to_string(index=False))

csv_path = '/content/eval_results/metrics/full_model_comparison.csv'
df.to_csv(csv_path, index=False)
print(f"\nFull comparison table saved: {csv_path}")


# ── Accuracy heatmap across models and magnifications ─────────────────────────
pivot = df.pivot(index='Magnification', columns='Model', values='Accuracy')
plt.figure(figsize=(10, 5))
sns.heatmap(
    pivot, annot=True, fmt='.4f', cmap='YlOrRd',
    linewidths=0.5
)
plt.title('Accuracy Heatmap — All Models × All Magnifications')
plt.tight_layout()
plt.savefig('/content/eval_results/plots/accuracy_heatmap.png', dpi=150)
plt.close()
print("Accuracy heatmap saved.")


# ── Bar chart comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for ax, metric in zip(axes, metrics_to_plot):
    pivot_m = df.pivot(index='Model', columns='Magnification', values=metric)
    pivot_m.plot(kind='bar', ax=ax, colormap='Set2', legend=(metric == 'Accuracy'))
    ax.set_title(metric)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=8)

plt.suptitle('Model Performance Across All Magnifications', fontsize=14)
plt.tight_layout()
plt.savefig('/content/eval_results/plots/full_comparison_chart.png', dpi=150)
plt.close()
print("Full comparison chart saved.")


# ── Copy all results to Google Drive ──────────────────────────────────────────
drive_eval = f'{DRIVE_BASE}/results_all_magnifications/evaluation'
os.makedirs(drive_eval, exist_ok=True)
shutil.copytree('/content/eval_results', drive_eval, dirs_exist_ok=True)
print(f"\nAll evaluation results copied to Drive: {drive_eval}")