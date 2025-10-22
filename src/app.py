# src/app.py
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
import json
import tensorflow as tf
from PIL import Image, ImageOps
import tensorflow_datasets as tfds
from src.api_integration import get_calories

app = FastAPI()

# Load your trained model at startup
MODEL_PATH = "models/mobilenetv2_food101.h5"
model = load_model(MODEL_PATH)


def _iter_all_layers(layer):
    # Recursively iterate through all sublayers
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            yield from _iter_all_layers(sub)
    yield layer


def _model_has_rescaling(m) -> bool:
    try:
        for lyr in _iter_all_layers(m):
            if isinstance(lyr, tf.keras.layers.Rescaling):
                return True
        return False
    except Exception:
        return False


# Detect whether model already rescales (1./255) internally
MODEL_DOES_RESCALING = _model_has_rescaling(model)

# Load class names from saved mapping if present; otherwise from TFDS
_labels_path = os.path.join(os.path.dirname(__file__), "..", "models", "class_names.json")
_labels_path = os.path.normpath(_labels_path)
if os.path.exists(_labels_path):
    with open(_labels_path, "r") as f:
        labels = json.load(f)
else:
    _info = tfds.builder("food101").info
    labels = _info.features["label"].names

def preprocess_image(img_bytes, normalize: bool):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    if normalize:
        img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def _tta_batch_from_bytes(img_bytes: bytes, normalize: bool):
    # Build a small TTA batch: original, hflip, rotate +10, rotate -10
    base = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    variants = [
        base,
        ImageOps.mirror(base),
        base.rotate(10, resample=Image.BILINEAR),
        base.rotate(-10, resample=Image.BILINEAR),
    ]
    arrays = []
    for im in variants:
        arr = image.img_to_array(im)
        if normalize:
            arr = arr / 255.0
        arrays.append(arr)
    batch = np.stack(arrays, axis=0)
    return batch

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Normalize only if model does not include its own Rescaling layer
    do_norm = not MODEL_DOES_RESCALING

    # Test-time augmentation for more robust predictions
    batch = _tta_batch_from_bytes(contents, normalize=do_norm)
    preds = model.predict(batch)
    mean_preds = preds.mean(axis=0)

    top_indices = np.argsort(mean_preds)[-3:][::-1]
    top = [
        {"label": labels[i], "probability": float(mean_preds[i])}
        for i in top_indices
    ]

    pred_idx = int(top_indices[0])
    pred_label = labels[pred_idx]
    confidence = float(mean_preds[pred_idx])

    calories = get_calories(pred_label.replace("_", " "))
    return {
        "predicted_food": pred_label,
        "confidence": confidence,
        "top3": top,
        "calories": calories,
    }
