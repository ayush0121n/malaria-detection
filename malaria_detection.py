import sys
import io

# Force UTF-8 output — safe on Windows (Python 3.x, including 3.13)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
Malaria Cell Detection using Deep Learning
===========================================
Dataset: Cell Images for Detecting Malaria
Source : Kaggle - iarunava/cell-images-for-detecting-malaria

Activities:
  2.1 - Importing the Libraries
  2.2 - Reading the Dataset
  2.3 - Data Preprocessing & Augmentation
  2.4 - Model Building (CustomCNN, MobileNetV2, EfficientNetB0)
  2.5 - Model Training
  2.6 - Evaluation & Visualisation
  2.7 - Prediction / Inference
"""

# =============================================================================
# Activity 2.1 — Importing the Libraries
# =============================================================================
print("=" * 60)
print("Activity 2.1: Importing Libraries")
print("=" * 60)

# ── Standard Library ─────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Numerical & Data Handling ─────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# ── Image Processing ──────────────────────────────────────────────────────────
from PIL import Image

# ── Scikit-learn Utilities ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder

# ── SciPy ─────────────────────────────────────────────────────────────────────
from scipy import stats

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# ── KaggleHub  ────────────────────────────────────────────────────────────────
import kagglehub
from kagglehub import KaggleDatasetAdapter

print("✅  All libraries imported successfully.\n")
print(f"   TensorFlow version : {tf.__version__}")
print(f"   Keras version      : {keras.__version__}")
print(f"   NumPy version      : {np.__version__}")
print(f"   Pandas version     : {pd.__version__}")


# =============================================================================
# Activity 2.2 — Reading the Dataset
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.2: Reading the Dataset")
print("=" * 60)


# ── Configuration ─────────────────────────────────────────────────────────────
class Config:
    IMG_SIZE             = (64, 64)    # 64x64 for fast CPU training (change to 128,128 for full run)
    BATCH_SIZE           = 64          # larger batch = fewer steps per epoch
    EPOCHS               = 10          # quick demo; increase to 40 for full training
    LEARNING_RATE        = 1e-4
    VALIDATION_SPLIT     = 0.2
    SEED                 = 42
    DATASET_ID           = "iarunava/cell-images-for-detecting-malaria"
    CLASSES              = ["Parasitized", "Uninfected"]
    MAX_SAMPLES_PER_CLASS = 3000       # cap per class -> ~6000 total (fast CPU run)

config = Config()


# ── Helper: build a DataFrame of (filepath, label) from a directory tree ──────
def build_image_dataframe(root_dir: str) -> pd.DataFrame:
    """
    Walk *root_dir* recursively and return a DataFrame with columns
    ['filepath', 'label'].  Only .png / .jpg / .jpeg / .bmp files are included.
    The label is taken from the immediate parent folder name.
    """
    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
    records = []
    for dirpath, _, filenames in os.walk(root_dir):
        label = os.path.basename(dirpath)
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in VALID_EXT:
                records.append(
                    {"filepath": os.path.join(dirpath, fname), "label": label}
                )
    return pd.DataFrame(records)


# ── Download dataset via kagglehub ────────────────────────────────────────────
print("\n[1] Downloading dataset from Kaggle …")
print(f"    Dataset : {config.DATASET_ID}")

dataset_path = kagglehub.dataset_download(config.DATASET_ID)
print(f"    Downloaded to : {dataset_path}")


# ── Locate the 'cell_images' root  ────────────────────────────────────────────
print("\n[2] Locating cell_images directory …")
cell_images_dir = None
for root, dirs, _ in os.walk(dataset_path):
    if "cell_images" in dirs:
        cell_images_dir = os.path.join(root, "cell_images")
        break
    if os.path.basename(root) == "cell_images":
        cell_images_dir = root
        break

if cell_images_dir is None:
    # Fallback: use the downloaded root directly
    cell_images_dir = str(dataset_path)
else:
    cell_images_dir = str(cell_images_dir)

print(f"    cell_images dir : {cell_images_dir}")

# Show top-level contents
print("\n    Directory structure:")
for item in os.listdir(cell_images_dir):
    full = os.path.join(cell_images_dir, item)
    if os.path.isdir(full):
        count = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
        print(f"      📁 {item}/  ({count} files)")
    else:
        print(f"      📄 {item}")


# ── Build DataFrame ───────────────────────────────────────────────────────────
print("\n[3] Building image DataFrame …")
df = build_image_dataframe(cell_images_dir)

# Keep only the two target classes
df = df[df["label"].isin(config.CLASSES)].reset_index(drop=True)

# ── Subsample for fast CPU training ───────────────────────────────────────────
sampled_parts = []
for cls in config.CLASSES:
    cls_df = df[df["label"] == cls]
    n = min(config.MAX_SAMPLES_PER_CLASS, len(cls_df))
    sampled_parts.append(cls_df.sample(n=n, random_state=config.SEED))
df = pd.concat(sampled_parts).reset_index(drop=True)

print(f"\n    Total images (after sampling) : {len(df):,}")
print(f"\n    Class distribution :")
print(df["label"].value_counts().to_string())




# ── Quick EDA ─────────────────────────────────────────────────────────────────
print("\n[4] DataFrame head (first 5 rows):")
print(df.head().to_string(index=False))

print("\n[5] DataFrame info:")
print(f"    Shape  : {df.shape}")
print(f"    Dtypes :\n{df.dtypes.to_string()}")
print(f"    Nulls  :\n{df.isnull().sum().to_string()}")


# =============================================================================
# Activity 2.3 — Data Preprocessing & Augmentation
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.3: Data Preprocessing & Augmentation")
print("=" * 60)

# ── Train / Validation split ──────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df,
    test_size=config.VALIDATION_SPLIT,
    random_state=config.SEED,
    stratify=df["label"],
)
print(f"    Training samples   : {len(train_df):,}")
print(f"    Validation samples : {len(val_df):,}")

# ── Image Data Generators ─────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode="binary",
    seed=config.SEED,
    shuffle=True,
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="label",
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode="binary",
    seed=config.SEED,
    shuffle=False,
)

print(f"\n    Class indices : {train_gen.class_indices}")


# =============================================================================
# Activity 2.4 — Model Building
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.4: Model Building")
print("=" * 60)


def create_custom_cnn(img_size):
    """Custom CNN architecture."""
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=(*img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 4
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Classifier head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CustomCNN",
    )
    return model


def create_transfer_model(base_name: str, img_size):
    """Transfer-learning model using MobileNetV2 or EfficientNetB0."""
    input_shape = (*img_size, 3)
    if base_name == "MobileNetV2":
        base = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif base_name == "EfficientNetB0":
        base = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_name}")

    # Freeze base layers
    base.trainable = False

    inputs  = keras.Input(shape=input_shape)
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name=base_name)


# =============================================================================
# Activity 2.5 — Model Training
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.5: Model Training")
print("=" * 60)


def get_callbacks(model_name: str):
    return [
        ModelCheckpoint(
            f"{model_name}_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_model(model, model_name, train_gen, val_gen, epochs, lr):
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print("=" * 50)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    return history, model


# ── Train all three models ────────────────────────────────────────────────────
models_to_train = {
    "CustomCNN":     create_custom_cnn(config.IMG_SIZE),
    "MobileNetV2":   create_transfer_model("MobileNetV2",   config.IMG_SIZE),
    "EfficientNetB0": create_transfer_model("EfficientNetB0", config.IMG_SIZE),
}

results = {}

for model_name, model in models_to_train.items():
    print(f"\n{model_name} Summary:")
    model.summary()

    history, trained_model = train_model(
        model, model_name,
        train_gen, val_gen,
        config.EPOCHS, config.LEARNING_RATE,
    )
    results[model_name] = {"history": history, "model": trained_model}


# =============================================================================
# Activity 2.6 — Evaluation & Visualisation
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.6: Evaluation & Visualisation")
print("=" * 60)


def plot_training_history(history, model_name):
    """Plot accuracy and loss curves for one model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14)

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def evaluate_model(model, model_name, val_gen):
    """Evaluate and print metrics + confusion matrix."""
    val_gen.reset()
    y_true  = val_gen.classes
    y_prob  = model.predict(val_gen, verbose=0).flatten()
    y_pred  = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'─' * 40}")
    print(f"  {model_name}  —  Val Accuracy : {acc:.4f}")
    print(f"{'─' * 40}")
    print(classification_report(
        y_true, y_pred,
        target_names=list(val_gen.class_indices.keys()),
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(val_gen.class_indices.keys()),
        yticklabels=list(val_gen.class_indices.keys()),
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return acc, roc_auc


# ── Plot and evaluate each model ─────────────────────────────────────────────
for model_name, res in results.items():
    plot_training_history(res["history"], model_name)
    val_gen.reset()
    acc, roc_auc = evaluate_model(res["model"], model_name, val_gen)
    results[model_name]["val_accuracy"] = acc
    results[model_name]["roc_auc"]      = roc_auc


# ── 4. Find best model ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

comparison_df = pd.DataFrame({
    'Model':               list(results.keys()),
    'Validation Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
    'ROC AUC':             [results[m]['roc_auc']      for m in results.keys()],
})
print("\n", comparison_df)

if not comparison_df.empty:
    best_model_name = comparison_df.loc[comparison_df['Validation Accuracy'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Validation Accuracy: {results[best_model_name]['val_accuracy']:.4f}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")


# =============================================================================
# Activity 2.7 — Prediction / Inference helper
# =============================================================================
print("\n" + "=" * 60)
print("Activity 2.7: Prediction / Inference Helper")
print("=" * 60)


def predict_image(image_path: str, model, img_size=None):
    """
    Load a single cell image, preprocess it, and predict whether it is
    Parasitized or Uninfected.

    Parameters
    ----------
    image_path : str   – absolute or relative path to the image file.
    model      : keras Model  – trained binary classifier.
    img_size   : tuple – (height, width) used during training (optional).

    Returns
    -------
    label : str   – 'Parasitized' or 'Uninfected'
    prob  : float – model confidence (probability of Parasitized)
    """
    if img_size is None:
        # Avoid shape mismatch by pulling expected size directly from the model
        img_size = (model.input_shape[1], model.input_shape[2])

    img  = Image.open(image_path).convert("RGB").resize(img_size)
    arr  = np.array(img, dtype="float32") / 255.0
    arr  = np.expand_dims(arr, axis=0)           # (1, H, W, 3)

    prob = float(model.predict(arr, verbose=0)[0][0])

    # Class mapping depends on generator class_indices:
    # train_gen.class_indices -> {'Parasitized': 0, 'Uninfected': 1}
    # sigmoid output < 0.5  -> class 0 (Parasitized)
    # sigmoid output >= 0.5 -> class 1 (Uninfected)
    label = "Uninfected" if prob >= 0.5 else "Parasitized"
    return label, prob


print("\n✅  predict_image() helper function defined.")
print("    Usage example:")
print("      label, prob = predict_image('path/to/cell.png', best_model)")
print("      print(f'Prediction: {label}  |  Confidence: {prob:.4f}')")

print("\n✅  Malaria Detection pipeline complete.")
