# src/app.py
"""
Main FastAPI backend for the Food Calorie Estimator.
Handles image uploads, food recognition, and calorie estimation.
"""

import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

from src.api_integration import get_calories  #

# ====================================
#  CONFIGURATION & INITIALIZATION
# ====================================
load_dotenv()

app = FastAPI(
    title="Food Recognition API",
    description="Recognize foods and estimate calories using MobileNetV2 + Edamam API",
    version="2.0",
)

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/mobilenetv2_food101_after55.keras")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "../models/food101_classes.txt")


# ====================================
# ✅ LOAD MODEL & LABELS
# ====================================
print(f"🔄 Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded and compiled successfully!")

if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ Loaded {len(LABELS)} Food-101 class names locally.")
else:
    raise FileNotFoundError(f"⚠️ Missing class label file: {LABELS_FILE}")


# ====================================
# ✅ HELPER FUNCTION — Prediction
# ====================================
def predict_food(image_bytes: bytes) -> str:
    """Run inference on an uploaded image and return the predicted food label."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        model_input_shape = model.input_shape[1:3]
        image = image.resize(model_input_shape)
        img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = model.predict(img_array)
        idx = int(np.argmax(preds))
        predicted_food = LABELS[idx]
        confidence = float(np.max(preds)) * 100
        print(f"🔍 Prediction index={idx} → {predicted_food} ({confidence:.2f}%)")
        return predicted_food
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ====================================
# ✅ ROUTES
# ====================================
@app.get("/")
def root():
    return {"message": "🍽️ Food Recognition API is running successfully!"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image → return food prediction + calorie info."""
    try:
        contents = await file.read()
        predicted_food = predict_food(contents)
        calories = get_calories(predicted_food)  # ✅ Unified lookup

        # Optional price heuristic (can adjust later)
        price = round(calories * 0.015, 2)

        return {
            "predicted_food": predicted_food,
            "calories": calories,
            "price": price
        }

    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
