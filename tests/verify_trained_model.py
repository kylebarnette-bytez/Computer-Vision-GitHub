"""
verify_trained_model.py — Confirms that your fine-tuned MobileNetV2 model loads,
evaluates correctly, and can make predictions on a sample image.
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# -------------------------------
# 1️⃣ Load the trained model
# -------------------------------
MODEL_PATH = "models/mobilenetv2_food101_after10.keras"  # change if renamed
print(f"🔍 Loading model from: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!\n")

# -------------------------------
# 2️⃣ Print architecture summary
# -------------------------------
model.summary()

# -------------------------------
# 3️⃣ Optional: evaluate on validation data
# -------------------------------
try:
    from src.data_preprocessing import get_datasets
    train_data, val_data = get_datasets(batch_size=32)
    print("\n📊 Evaluating on validation data...")
    results = model.evaluate(val_data)
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
except Exception as e:
    print("\n⚠️ Skipping validation evaluation (dataset not found or import issue).")
    print(f"Reason: {e}\n")

# -------------------------------
# 4️⃣ Predict on a test image
# -------------------------------
SAMPLE_IMG = "tests/sample_food.jpg"  # drop any Food-101 image here
if os.path.exists(SAMPLE_IMG):
    print(f"🍽️ Running sample prediction on: {SAMPLE_IMG}")
    img = image.load_img(SAMPLE_IMG, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    preds = model.predict(x)
    pred_index = np.argmax(preds)
    confidence = np.max(preds)

    try:
        from src.utils import load_class_names
        class_names = load_class_names()
        print(f"✅ Prediction: {class_names[pred_index]} ({confidence:.2%} confidence)")
    except Exception:
        print(f"✅ Prediction index: {pred_index} ({confidence:.2%} confidence)")
else:
    print("⚠️ No test image found. Place one at 'tests/sample_food.jpg' to test predictions.")
