from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import io
from PIL import Image
import requests
import os

app = FastAPI()

# ============================================================
# 🔹 1️⃣ Load your NEW fine-tuned model
# ============================================================
# Make sure this file exists (check models/ folder)
MODEL_PATH = "models/mobilenetv2_food101_after55.keras"
model = load_model(MODEL_PATH, compile=False)

# Automatically detect input size (e.g., (160,160))
MODEL_INPUT_SHAPE = model.input_shape[1:3]

# ============================================================
# 🔹 2️⃣ Load class names
# ============================================================
with open("src/class_names.txt") as f:
    class_names = [line.strip() for line in f]

# Optional label cleanup map
label_map = {
    "spaghetti_bolognese": "spaghetti bolognese",
    "peking_duck": "roast duck",
    "beef_tartare": "beef tartare",
    "apple_pie": "apple pie",
    "macarons": "macarons dessert",
    "chocolate_cake": "chocolate cake",
    "fried_rice": "fried rice",
    "pizza_margherita": "pizza margherita",
}

# ============================================================
# 🔹 3️⃣ Preprocess uploaded image
# ============================================================
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    # ✅ Resize to model's expected input (auto-detected)
    img = cv2.resize(img, MODEL_INPUT_SHAPE)
    img = preprocess_input(img)  # same normalization as training
    return np.expand_dims(img, axis=0)

# ============================================================
# 🔹 4️⃣ Prediction route
# ============================================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = prepare_image(image_bytes)

    preds = model.predict(img)
    top_indices = preds[0].argsort()[-5:][::-1]
    top_labels = [class_names[i] for i in top_indices]
    top_scores = [float(preds[0][i]) for i in top_indices]

    # Load API keys from environment
    app_id = os.getenv("EDAMAM_APP_ID")
    app_key = os.getenv("EDAMAM_APP_KEY")
    url = "https://api.edamam.com/api/food-database/v2/parser"

    def get_best_food_match(top_labels):
        """Return first label that Edamam actually recognizes."""
        for lbl in top_labels:
            name = label_map.get(lbl, lbl.replace("_", " "))
            params = {"ingr": name, "app_id": app_id, "app_key": app_key}
            res = requests.get(url, params=params).json()
            if "parsed" in res and res["parsed"]:
                return name
            if "hints" in res and res["hints"]:
                return name
        # fallback to first guess
        return label_map.get(top_labels[0], top_labels[0].replace("_", " "))

    food_name = get_best_food_match(top_labels)

    # ============================================================
    # 🔹 Fetch calories from Edamam (with fallback)
    # ============================================================
    params = {"ingr": food_name, "app_id": app_id, "app_key": app_key}
    calories = None

    try:
        response = requests.get(url, params=params, timeout=6)
        response.raise_for_status()
        data = response.json()

        if "parsed" in data and data["parsed"]:
            calories = data["parsed"][0]["food"]["nutrients"].get("ENERC_KCAL", None)
        elif "hints" in data and data["hints"]:
            calories = data["hints"][0]["food"]["nutrients"].get("ENERC_KCAL", None)
        else:
            print(f"⚠️ No nutrition data found for {food_name}")

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"⚠️ Edamam API error for {food_name}: {e}")

    # 🔸 Fallback if API fails or returns None
    if calories is None:
        calories = 0.0  # or use a string "N/A"


    return result
