# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
import numpy as np
import io
import os


# Initialize FastAPI
app = FastAPI()

# Enable CORS so React frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; in production, use ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome To The Food Calorie Estimator!"}

# Load your trained model
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mobilenetv2_food101.h5")
#model = load_model(MODEL_PATH)

# Food-101 labels (simplified example; replace with full 101 labels)
labels = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
    "banana", "apple", "orange", "broccoli", "carrot", "strawberry"
]

class FoodName(BaseModel):
    name: str

# Mock data for calories & price
mock_data = {
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

# Helper: preprocess image for model
#def preprocess_image(img_bytes):
    #img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    #img = img.resize((224, 224))
    #img_array = image.img_to_array(img)
    #img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    #img_array = img_array / 255.0  # normalize like training
    #return img_array

# POST endpoint for image upload
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    
    data = mock_data["apple"]
    
    #contents = await file.read()
    #img_array = preprocess_image(contents)
    #preds = model.predict(img_array)
    #pred_idx = np.argmax(preds, axis=1)[0]
    #pred_label = labels[pred_idx]

    # Lookup calories & price using first word of label
    #key = pred_label.split("_")[0].lower()
    #data = food_data.get(key, {"calories": 100, "price": 1.0})

    return {
         "predicted_food": "apple",
        "calories": data["calories"],
        "price": data["price"]
    }

# Pydantic model for typed food name
class FoodName(BaseModel):
    name: str

# POST endpoint for typed name
@app.post("/get-info")
async def get_info(food: FoodName):
    #key = food.name.lower()
    data = mock_data.get(food.name.lower(), {"calories": 100, "price": 1.0})
    return {
        "predicted_food": food.name,
        "calories": data["calories"],
        "price": data["price"]
    }
