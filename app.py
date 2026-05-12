"""
app.py
------
Flask web application for the breast cancer detection system.
Implements Objective v from Chapter 1: deploy model as a web application.

DenseNet121 is used as the deployed model based on comparative analysis
showing it achieved the highest accuracy across all magnification levels.

Features:
  - Upload a histopathological image
  - Select magnification level (40X, 100X, 200X, 400X)
  - Get a prediction (Benign / Malignant) with confidence score
  - View Grad-CAM heatmap explanation of the prediction

Usage:
    python app.py
    Then open: http://127.0.0.1:5000
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from src.explainability.gradcam import save_gradcam, load_model_from_weights

# ── App config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
IMG_SIZE           = (224, 224)

# Best performing model from comparative analysis
MODEL_NAME     = "DenseNet121"
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# DenseNet121 accuracy at each magnification from evaluation
MODEL_ACCURACY = {
    "40X" : "76.25%",
    "100X": "87.50%",
    "200X": "88.56%",
    "400X": "87.82%",
}

# Cache loaded models to avoid reloading on every request
model_cache = {}


# ── Get model ──────────────────────────────────────────────────────────────────
def get_model(magnification: str):
    """
    Returns cached DenseNet121 model for given magnification.
    Loads and caches it on first request.
    """
    if magnification not in model_cache:
        print(f"  Loading DenseNet121 | {magnification}...")
        model_cache[magnification] = load_model_from_weights(
            MODEL_NAME, magnification
        )
        print(f"  Loaded successfully.")
    return model_cache[magnification]


# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(img_path: str) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=IMG_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        magnifications=MAGNIFICATIONS,
        model_accuracy=MODEL_ACCURACY,
        model_name=MODEL_NAME
    )

@app.route("/health")
def health():
    import os
    status = {}
    for mag in MAGNIFICATIONS:
        path = f"models/saved_models/DenseNet121_{mag}_weights.weights.h5"
        status[mag] = os.path.exists(path)
    return jsonify({
        "status": "running",
        "weights_found": status,
        "cwd": os.getcwd()
    })
    
@app.route("/predict", methods=["POST"])
def predict():
    magnification = request.form.get("magnification", "400X")

    if magnification not in MAGNIFICATIONS:
        return jsonify({
            "error": f"Invalid magnification: {magnification}"
        }), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type. Please upload PNG, JPG, or TIF."
        }), 400

    # Save uploaded file
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    # Load DenseNet121 for selected magnification
    try:
        model = get_model(magnification)
    except FileNotFoundError:
        return jsonify({
            "error": (
                f"Model weights not found for "
                f"{MODEL_NAME} | {magnification}"
            )
        }), 500

    # Predict
    img_array  = preprocess_image(save_path)
    pred_prob  = float(model.predict(img_array, verbose=0)[0][0])
    label      = "Malignant" if pred_prob > 0.5 else "Benign"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    # Generate Grad-CAM
    os.makedirs("static/uploads/gradcam", exist_ok=True)
    gradcam_path = save_gradcam(
        MODEL_NAME,
        save_path,
        magnification,
        model,
        save_dir="static/uploads/gradcam"
    )
    gradcam_url = "/static/uploads/gradcam/" + os.path.basename(gradcam_path)

    return jsonify({
        "prediction"   : label,
        "confidence"   : f"{confidence:.2%}",
        "gradcam_url"  : gradcam_url,
        "model_used"   : MODEL_NAME,
        "magnification": magnification,
        "model_accuracy": MODEL_ACCURACY[magnification]
    })


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    print("\n" + "=" * 60)
    print("  Breast Cancer Detection — Web Application")
    print("  Author: Sobande Olukayode Oluwatofunmi (BU22CSC1016)")
    print("  Model: DenseNet121 (Best performing model)")
    print("=" * 60)
    print("\n  DenseNet121 accuracy by magnification:")
    for mag, acc in MODEL_ACCURACY.items():
        print(f"    {mag} : {acc}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)