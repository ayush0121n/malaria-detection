<div align="center">

# 🔬 MalariaScope — AI-Powered Malaria Cell Detection

**Deep learning system for detecting malaria parasites in blood-smear microscopic images**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-~93%25-brightgreen)](README.md)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Demo](#-demo)
- [Models](#-models)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [API Reference](#-api-reference)
- [Configuration](#️-configuration)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🧬 Overview

**MalariaScope** is a complete end-to-end deep learning project that classifies blood-smear microscopic cell images as either **Parasitized** (infected with malaria) or **Uninfected** using convolutional neural networks..

The project includes:
- A **full ML training pipeline** (`malaria_detection.py`) covering data loading, preprocessing, model building, training, evaluation, and inference
- A **Flask web application** (`app.py`) providing a beautiful dark-mode UI and REST API for real-time predictions
- Three model architectures: **Custom CNN**, **MobileNetV2** (transfer learning), **EfficientNetB0** (transfer learning)

> ⚠️ **Medical Disclaimer:** This system is a screening tool for research and educational purposes. Results must be verified by qualified medical personnel. Not a substitute for professional medical diagnosis.

---

## 📁 Project Structure

```
malaria/
│
├── 📄 app.py                          # Flask web application & REST API
├── 📄 malaria_detection.py            # Full ML pipeline (Activities 2.1–2.7)
├── 📄 download_dataset.py             # Standalone Kaggle dataset downloader
├── 📄 requirements.txt                # Python dependencies
├── 📄 .gitignore                      # Git exclusions
│
├── 📁 templates/
│   └── index.html                     # Frontend UI (dark-mode, drag & drop)
│
├── 📁 static/                         # Static assets (CSS, JS, images)
│
├── 🤖 CustomCNN_best.h5               # Best checkpoint – Custom CNN (~93% acc)
├── 🤖 MobileNetV2_best.h5             # Best checkpoint – MobileNetV2 (~91% acc)
├── 🤖 EfficientNetB0_best.h5          # Best checkpoint – EfficientNetB0
│
├── 📊 CustomCNN_training_history.png  # Training curves
├── 📊 CustomCNN_confusion_matrix.png  # Confusion matrix
├── 📊 CustomCNN_roc_curve.png         # ROC-AUC curve
└── 📊 MobileNetV2_training_history.png
```

---

## 🎬 Demo

The web app runs at **http://127.0.0.1:5000**

| Feature | Description |
|---------|-------------|
| 🖼️ **Drag & Drop Upload** | Drop any PNG/JPG/JPEG blood cell image |
| ⚡ **Real-time Inference** | Results in < 3 seconds |
| 📊 **Diagnostic Report** | Confidence score, probability breakdown, timestamps |
| 🌙 **Dark Mode UI** | Premium glassmorphism design with particle animations |
| 🔁 **Analyze More** | One-click reset to analyze another sample |

---

## 🤖 Models

### Architecture 1 — Custom CNN

A purpose-built 4-block convolutional neural network:

```
Input (64×64×3)
  → Conv2D(32) + BN + MaxPool + Dropout(0.25)
  → Conv2D(64) + BN + MaxPool + Dropout(0.25)
  → Conv2D(128) + BN + MaxPool + Dropout(0.25)
  → Conv2D(256) + BN + MaxPool + Dropout(0.25)
  → GlobalAveragePooling2D
  → Dense(512) + BN + Dropout(0.5)
  → Dense(1, sigmoid)        ← binary output
```

### Architecture 2 — MobileNetV2 (Transfer Learning)

```
Input (64×64×3)
  → MobileNetV2(weights='imagenet', frozen)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.5)
  → Dense(1, sigmoid)
```

### Architecture 3 — EfficientNetB0 (Transfer Learning)

```
Input (64×64×3)
  → EfficientNetB0(weights='imagenet', frozen)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.5)
  → Dense(1, sigmoid)
```

> **Class mapping:** `sigmoid < 0.5` → Parasitized (class 0), `sigmoid >= 0.5` → Uninfected (class 1)

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |
| **Author** | Lister Hill National Center (US NIH) |
| **Classes** | Parasitized / Uninfected |
| **Total images** | ~27,558 (13,779 per class) |
| **Image format** | PNG, JPEG |
| **Used in training** | Up to 6,000 (3,000 per class, configurable) |

The dataset is automatically downloaded from Kaggle via `kagglehub` when you run `malaria_detection.py`.

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 – 3.13 |
| pip | Latest |
| Kaggle Account | Required for dataset download |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/malaria-detection.git
cd malaria-detection
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Kaggle API Credentials

> Only needed if you want to **retrain** the models. Skip if using the pre-trained `.h5` files.

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. This downloads `kaggle.json`
3. Place it at:
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`
4. Set permissions (macOS/Linux only): `chmod 600 ~/.kaggle/kaggle.json`

The file should look like:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

---

## 🚀 Running the Project

### Option A — Run the Web App (Pre-trained Models)

This uses the saved `.h5` model files. No dataset download needed.

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

Upload a blood cell image and click **Analyze Sample** to get a prediction.

### Option B — Run the Full ML Training Pipeline

This downloads the dataset from Kaggle, trains all 3 models, and saves checkpoints.

```bash
python malaria_detection.py
```

This will:
1. Download the dataset (~350 MB) via KaggleHub
2. Train **CustomCNN**, **MobileNetV2**, and **EfficientNetB0**
3. Save best checkpoints as `*_best.h5`
4. Generate and save training history plots, confusion matrices, ROC curves

> ⏱️ **Estimated training time:** ~15–30 minutes on CPU | ~5 minutes on GPU

### Option C — Download Dataset Only

```bash
python download_dataset.py
```

---

## 📡 API Reference

The Flask app exposes 4 REST endpoints:

### `GET /`
Returns the main web UI.

---

### `GET /health`
Returns server and model status.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "model_name": "CustomCNN",
  "model_path": "/absolute/path/to/CustomCNN_best.h5",
  "error": null
}
```

---

### `GET /models`
Returns information about the loaded model.

**Response:**
```json
{
  "loaded": "CustomCNN",
  "input_shape": [null, 64, 64, 3],
  "classes": ["Parasitized", "Uninfected"]
}
```

---

### `POST /predict`
Accepts an image file and returns the malaria prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image: PNG | JPG | JPEG | BMP | TIFF>
```

**Response (success):**
```json
{
  "success": true,
  "prediction": "Parasitized",
  "confidence": 94.27,
  "model": "CustomCNN",
  "details": {
    "parasitized_probability": 94.27,
    "uninfected_probability": 5.73
  },
  "timestamp": "2026-03-06 18:08:00"
}
```

**Response (error):**
```json
{
  "success": false,
  "error": "No file uploaded"
}
```

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@path/to/cell_image.png"
```

**Python Example:**
```python
import requests

with open("cell_image.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"file": f}
    )

data = response.json()
print(f"Prediction: {data['prediction']}")
print(f"Confidence: {data['confidence']}%")
```

---

## 🎛️ Configuration

Edit the `Config` class in `malaria_detection.py` to tune training:

```python
class Config:
    IMG_SIZE              = (64, 64)   # Image dimensions — increase for better accuracy
    BATCH_SIZE            = 64         # Samples per gradient update
    EPOCHS                = 10         # Training epochs — increase to 30–50 for full run
    LEARNING_RATE         = 1e-4       # Adam learning rate
    VALIDATION_SPLIT      = 0.2        # 20% of data for validation
    SEED                  = 42         # Reproducibility seed
    DATASET_ID            = "iarunava/cell-images-for-detecting-malaria"
    CLASSES               = ["Parasitized", "Uninfected"]
    MAX_SAMPLES_PER_CLASS = 3000       # Subsample limit — use None for full dataset
```

| Parameter | Fast CPU | Full Quality |
|-----------|----------|--------------|
| `IMG_SIZE` | `(64, 64)` | `(128, 128)` |
| `EPOCHS` | `10` | `40` |
| `MAX_SAMPLES_PER_CLASS` | `3000` | `None` |

---

## 📊 Results

| Model | Validation Accuracy | ROC-AUC | Notes |
|-------|-------------------|---------|-------|
| **CustomCNN** | ~93% | ~0.97 | Best overall – fast & accurate |
| **MobileNetV2** | ~91% | ~0.96 | Good transfer learning baseline |
| **EfficientNetB0** | ~50% | ~0.50 | Needs fine-tuning (frozen base too restrictive at 64×64) |

> Results are from a 6,000 image subset (3,000/class) trained for 10 epochs at 64×64.
> Using the full dataset at 128×128 for 40 epochs typically achieves **95%+ accuracy**.

### Training Curves
The following charts are automatically generated during training:

| File | Description |
|------|-------------|
| `CustomCNN_training_history.png` | Accuracy & Loss over epochs |
| `CustomCNN_confusion_matrix.png` | True/False Positive/Negative heatmap |
| `CustomCNN_roc_curve.png` | Receiver Operating Characteristic curve |
| `MobileNetV2_training_history.png` | MobileNetV2 training curves |
| `MobileNetV2_confusion_matrix.png` | MobileNetV2 confusion matrix |

---

## 🔧 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow 2.x / Keras |
| **Transfer Learning** | MobileNetV2, EfficientNetB0 (ImageNet) |
| **Web Framework** | Flask 3.x |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Image Processing** | Pillow (PIL) |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset Access** | KaggleHub |
| **Frontend** | HTML5, Vanilla CSS (dark-mode, glassmorphism) |
| **Fonts** | Google Fonts – Inter |

---


## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License  ©  2026  Ayush Narkhede
```

---

## 🙏 Acknowledgements

- Dataset by [Lister Hill National Center for Biomedical Communication (US NIH)](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image/malaria-dataSet.html)
- Kaggle dataset published by [iarunava](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- Transfer learning models from [TensorFlow Model Garden](https://github.com/tensorflow/models)

---

<div align="center">

**🔬 MalariaScope — Saving Lives Through Technology**


</div>
