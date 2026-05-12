# Transfer Learning Model Incorporating Explainable AI for Early Detection of Breast Cancer

**Author:** Sobande Olukayode Oluwatofunmi (BU22CSC1016)
**Supervisor:** Dr. Abiodun Moses
**Institution:** Bowen University, Iwo, Osun State, Nigeria

## Project Overview
This project develops a transfer learning model for early breast cancer detection
using histopathological images from the BreaKHis dataset, incorporating
Explainable AI (Grad-CAM) for interpretability.

## Models
- VGG16
- ResNet50
- DenseNet121
- InceptionV3

## Dataset
- BreaKHis Dataset
- Magnifications: 40X, 100X, 200X, 400X
- Classes: Benign, Malignant
- Citation: Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L.,
  A Dataset for Breast Cancer Histopathological Image Classification,
  IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.

## Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## Tools
Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Flask

## Project Structure
- dataset/     → BreaKHis images organised by magnification
- src/         → All source code modules
- models/      → Saved trained model files
- results/     → Plots, Grad-CAM heatmaps, metrics
- static/      → Flask web app assets
- templates/   → Flask HTML templates
- notebooks/   → Jupyter notebooks

## How to Run
1. Place BreaKHis images in dataset/breakhis/{40X,100X,200X,400X}/{benign,malignant}/
2. Install dependencies: pip install -r requirements.txt
3. Run full pipeline: python main.py
4. Run web app: python app.py