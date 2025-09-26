# src/app.py
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load your trained model at startup
MODEL_PATH = "models/mobilenetv2_food101.h5"
model = load_model(MODEL_PATH)

# Food-101 labels (simplified example, you’ll need the full mapping)
labels = ["apple_pie", "baby_back_ribs", "baklava", "... up to 101"]

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0                  # normalize like training
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = labels[pred_idx]
    return {"predicted_food": pred_label}
