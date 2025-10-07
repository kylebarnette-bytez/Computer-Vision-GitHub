import os
import tensorflow as tf
from src.model import build_model

# Number of classes in Food-101
num_classes = 101

# Path to save the model
model_path = "models/mobilenetv2_food101.h5"

print("🔨 Building model...")
model = build_model(num_classes, save_path=model_path)

print(" Model built successfully")

# Save and check if file exists
if os.path.exists(model_path):
    print(f" Model file saved at: {model_path}")
else:
    print(" Model file not found!")

# Reload the model to verify it works
print(" Reloading model...")
reloaded_model = tf.keras.models.load_model(model_path)

print(" Reloaded model summary:")
reloaded_model.summary()
