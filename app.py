"""
app.py  –  Flask inference web-app for Malaria Detection
==========================================================
Fixes applied
  • UTF-8 stdout/stderr wrapper (Windows-safe)
  • Smart model selection: prefer CustomCNN_best.h5, then any *_best.h5,
    then any .h5 found in the project directory.
  • Robust image preprocessing that reads target size from model.input_shape.
  • /models  endpoint so the UI can show which model is loaded.
  • Cleaner error messages returned to the frontend.
"""

import sys
import io

# ── Force UTF-8 output — Windows-safe, works whether run directly or imported ──
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Globals ────────────────────────────────────────────────────────────────────
model       = None
MODEL_ERROR = None
MODEL_PATH  = None
MODEL_NAME  = None
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ── TensorFlow / Pillow import ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MALARIA DETECTION  –  LOADING MODEL")
print("=" * 70)

try:
    import tensorflow as tf
    from PIL import Image
    print(f"[OK] TensorFlow {tf.__version__}")
    print(f"[OK] Pillow {Image.__version__}")
except Exception as e:
    MODEL_ERROR = f"Import failed: {e}"
    print(f"[ERR] {MODEL_ERROR}")

# ── Model discovery & loading ──────────────────────────────────────────────────
PREFERRED_MODELS = [
    "CustomCNN_best.h5",
    "MobileNetV2_best.h5",
    "EfficientNetB0_best.h5",
]

if MODEL_ERROR is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[OK] Working directory: {current_dir}")

    # Build candidate list: preferred order first, then any remaining .h5
    h5_all = [f for f in os.listdir(current_dir) if f.endswith(".h5")]
    candidates = [f for f in PREFERRED_MODELS if f in h5_all]
    candidates += [f for f in h5_all if f not in candidates]

    print(f"     Found .h5 files: {h5_all}")
    print(f"     Load order     : {candidates}")

    for candidate in candidates:
        try:
            path = os.path.join(current_dir, candidate)
            model = tf.keras.models.load_model(path, compile=False)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            MODEL_PATH = path
            MODEL_NAME = candidate.replace("_best.h5", "").replace(".h5", "")
            print(f"[OK] Loaded: {candidate}")
            print(f"     Input : {model.input_shape}")
            print(f"     Output: {model.output_shape}")
            break
        except Exception as e:
            print(f"[WARN] Could not load {candidate}: {e}")

    if model is None:
        MODEL_ERROR = "No valid .h5 model could be loaded."
        print(f"[ERR] {MODEL_ERROR}")


# ── Image preprocessing ────────────────────────────────────────────────────────
def prepare_image(image_file):
    """Preprocess an uploaded file object into a numpy array the model expects."""
    if model is None:
        return None, "Model not loaded"

    try:
        img = Image.open(image_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        input_shape = model.input_shape  # e.g. (None, 64, 64, 3)

        if len(input_shape) == 4:
            # Standard CNN – (batch, H, W, C)
            h, w = input_shape[1], input_shape[2]
            img = img.resize((w, h))
            arr = np.array(img, dtype="float32") / 255.0
            return np.expand_dims(arr, axis=0), None

        elif len(input_shape) == 2:
            # Flat / MLP input – resize to 64x64 then flatten
            img = img.resize((64, 64))
            arr = np.array(img, dtype="float32") / 255.0
            gray = np.mean(arr, axis=2).flatten()
            expected = input_shape[1]
            if len(gray) < expected:
                gray = np.pad(gray, (0, expected - len(gray)))
            else:
                gray = gray[:expected]
            return np.expand_dims(gray, axis=0), None

        else:
            return None, f"Unsupported model input shape: {input_shape}"

    except Exception as e:
        return None, f"Image processing error: {e}"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status"      : "running",
        "model_loaded": model is not None,
        "model_name"  : MODEL_NAME,
        "model_path"  : MODEL_PATH,
        "error"       : MODEL_ERROR,
    })


@app.route("/models")
def models_info():
    return jsonify({
        "loaded"     : MODEL_NAME,
        "input_shape": list(model.input_shape) if model else None,
        "classes"    : CLASS_NAMES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": MODEL_ERROR or "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Validate extension
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"success": False, "error": f"Unsupported file type: {ext}"}), 400

    img_array, err = prepare_image(file)
    if img_array is None:
        return jsonify({"success": False, "error": err}), 500

    try:
        prediction = model.predict(img_array, verbose=0)
        print(f"Raw prediction: {prediction}")

        prob = float(prediction[0][0])           # sigmoid output in [0, 1]
        # class_indices: Parasitized → 0, Uninfected → 1
        # prob < 0.5 → class 0 (Parasitized), prob >= 0.5 → class 1 (Uninfected)
        if prob >= 0.5:
            pred_class = 1   # Uninfected
            confidence = prob * 100
        else:
            pred_class = 0   # Parasitized
            confidence = (1.0 - prob) * 100

        result           = CLASS_NAMES[pred_class]
        parasitized_prob = (1.0 - prob) * 100
        uninfected_prob  = prob * 100

        print(f"Result: {result}  |  Confidence: {confidence:.2f}%")

        return jsonify({
            "success"   : True,
            "prediction": result,
            "confidence": round(confidence, 2),
            "model"     : MODEL_NAME,
            "details"   : {
                "parasitized_probability": round(parasitized_prob, 2),
                "uninfected_probability" : round(uninfected_prob,  2),
            },
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static",    exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Model : {'LOADED  ->  ' + MODEL_NAME if model else 'NOT LOADED'}")
    if model:
        print(f"  Input : {model.input_shape}")
    if MODEL_ERROR:
        print(f"  Error : {MODEL_ERROR}")
    print(f"  URL   : http://127.0.0.1:5000")
    print("=" * 70 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
