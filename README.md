# 🦟 Malaria Cell Image Detection

A deep-learning pipeline that classifies blood-smear cell images as **Parasitized** or **Uninfected**, plus a Flask web app for live inference.

---

## Project Structure

```
malaria/
├── malaria_detection.py   # Full ML pipeline (Activities 2.1 – 2.7)
├── app.py                 # Flask inference web app
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Activities

| # | Activity | Description |
|---|----------|-------------|
| 2.1 | Importing Libraries | numpy, pandas, matplotlib, seaborn, sklearn, scipy, tensorflow, kagglehub |
| 2.2 | Reading the Dataset | Download via `kagglehub`, build image DataFrame |
| 2.3 | Data Preprocessing | Train/val split, `ImageDataGenerator` with augmentation |
| 2.4 | Model Building | CustomCNN, MobileNetV2 (transfer learning), EfficientNetB0 |
| 2.5 | Model Training | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |
| 2.6 | Evaluation | Classification report, Confusion matrix, ROC-AUC curve |
| 2.7 | Inference | `predict_image()` helper function |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API credentials

Create `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`):
```json
{"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_API_KEY"}
```
Get your key from https://www.kaggle.com/settings → **Create New Token**.

### 3. Run the training pipeline

```bash
python malaria_detection.py
```

This will:
- Download the Kaggle malaria dataset (~350 MB)
- Train three models (CustomCNN, MobileNetV2, EfficientNetB0)
- Save the best checkpoint for each model (`*_best.h5`)
- Plot and save training history, confusion matrices, and ROC curves

### 4. Run the Flask web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, upload a cell image, and click **Analyse Cell**.

---

## Dataset

- **Source:** [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Classes:** Parasitized (13,779 images) | Uninfected (13,779 images)
- **Total:** ~27,558 cell images

---

## Model Results (sample from screenshots)

| Model | Val Accuracy |
|-------|-------------|
| CustomCNN | ~93 % |
| MobileNetV2 | ~91 % |
| EfficientNetB0 | ~50 % (underfitting – needs fine-tuning) |
