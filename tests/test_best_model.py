"""
tests/test_best_model.py
Evaluate mobilenetv2_food101_best.keras on the Food-101 validation set.
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

# -------------------------------------------------------------
# 1. Find the model file no matter where the script is executed
# -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mobilenetv2_food101_best.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

print(f"📦 Loading model from: {MODEL_PATH}")

# -------------------------------------------------------------
# 2. Load and compile the model
# -------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    ]
)

print("✅ Model loaded and compiled successfully!\n")

# -------------------------------------------------------------
# 3. Build the Food-101 validation dataset
# -------------------------------------------------------------
print("📂 Loading Food-101 validation set (first time may take a few minutes)...")

_, val_ds = tfds.load("food101", split=["train", "validation"], as_supervised=True)

def preprocess(image, label):
    image = tf.image.resize(image, (160, 160))  # ✅ match model input shape
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
print("✅ Validation dataset ready!\n")

# -------------------------------------------------------------
# 4. Evaluate model performance
# -------------------------------------------------------------
print("🚀 Evaluating model on validation data...")
loss, acc, top5 = model.evaluate(val_ds, verbose=2)

print("\n📊 Evaluation Results:")
print(f"  Validation Loss:  {loss:.4f}")
print(f"  Top-1 Accuracy:   {acc * 100:.2f}%")
print(f"  Top-5 Accuracy:   {top5 * 100:.2f}%")
print("\n✅ Evaluation complete.")
