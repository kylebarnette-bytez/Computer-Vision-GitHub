"""
Resume fine-tuning from the previously trained head model.
Safe for Apple M1 — uses TFDS Food-101 (160x160).
"""

import os
import sys

# --- Make sure Python can find your src/ directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Disable GPU/Metal for M1 safety ---
os.environ["TF_METAL_ENABLE"] = "0"
os.environ["APPLE_ENABLE_METAL"] = "NO"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Now import everything else ---
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import compile_model, fine_tune_model


# ============================================================
# Load dataset
# ============================================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

print("📦 Loading Food-101 dataset from TensorFlow Datasets...")
(ds_train, ds_val), ds_info = tfds.load(
    "food101",
    split=["train", "validation"],
    as_supervised=True,
    shuffle_files=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = (
    ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ============================================================
# Load the head model and resume fine-tuning
# ============================================================
print("🧠 Loading trained head model ...")
model = tf.keras.models.load_model("models/mobilenetv2_food101_tfds_head_e05.keras", compile=False)
compile_model(model, learning_rate=1e-5)  # low LR for fine-tuning

fine_tune_model(
    model,
    train_ds,
    val_ds,
    fine_tune_at=80,
    epochs=5,
    save_path="models/mobilenetv2_food101_tfds_finetuned_e05.keras"
)

print("\n✅ Fine-tuning complete! Saved updated model in /models/")
