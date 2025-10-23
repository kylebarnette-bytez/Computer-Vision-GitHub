# src/app.py
# ------------------------------------------------------------
# Food Calorie Estimator Backend (FastAPI + TensorFlow)
# ------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
import numpy as np
import io
import os

# ------------------------------------------------------------
# Initialize FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Food Calorie Estimator")

# ------------------------------------------------------------
# CORS so frontend (React) can talk to backend
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Root route (test)
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome To The Food Calorie Estimator!"}

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "mobilenetv2_food101_20251020-104403-ft30_e03.keras"
)

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    model_loaded = False
    print(f"⚠️ Could not load model: {e}")

# ------------------------------------------------------------
# Food-101 label subset (replace with full set if available)
# ------------------------------------------------------------
labels = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
    "banana", "apple", "orange", "broccoli", "carrot", "strawberry"
]

# ------------------------------------------------------------
# Mock calorie/price database
# ------------------------------------------------------------
food_data = {
    "apple": {"calories": 52, "price": 0.5},
    "banana": {"calories": 89, "price": 0.3},
    "orange": {"calories": 62, "price": 0.4},
    "broccoli": {"calories": 55, "price": 0.8},
    "carrot": {"calories": 41, "price": 0.6},
    "strawberry": {"calories": 33, "price": 1.0},
    "apple_pie": {"calories": 237, "price": 2.5},
    "baby_back_ribs": {"calories": 350, "price": 5.0},
    "baklava": {"calories": 300, "price": 3.0},
    "beef_carpaccio": {"calories": 150, "price": 4.0},
}

# ------------------------------------------------------------
# Pydantic model for typed food input
# ------------------------------------------------------------
class FoodName(BaseModel):
    name: str

# ------------------------------------------------------------
# Image preprocessing helper
# ------------------------------------------------------------
def preprocess_image(img_bytes):
    """Preprocess image bytes to model input format."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalize
    return img_array

# ------------------------------------------------------------
# POST: upload image and get prediction
# ------------------------------------------------------------
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    if model_loaded and model is not None:
        # Preprocess and predict
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_label = labels[pred_idx % len(labels)]  # guard if labels shorter
    else:
        pred_label = "apple"  # fallback if model not loaded

    key = pred_label.split("_")[0].lower()
    data = food_data.get(key, {"calories": 100, "price": 1.0})

    return {
        "predicted_food": pred_label,
        "calories": data["calories"],
        "price": data["price"]
    }

# ------------------------------------------------------------
# POST: get calories/price for typed food name
# ------------------------------------------------------------
@app.post("/get-info")
async def get_info(food: FoodName):
    key = food.name.lower()
    data = food_data.get(key, {"calories": 100, "price": 1.0})
    return {
        "predicted_food": food.name,
        "calories": data["calories"],
        "price": data["price"]
    }
