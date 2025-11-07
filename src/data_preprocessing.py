# src/data_preprocessing.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# Data augmentation pipeline
# =====================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

import tensorflow as tf

def get_augmentation_layer():
    """Return stronger data augmentation for Food-101."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ], name="augmentation")


# =====================================================
# Image Preprocessing
# =====================================================
def preprocess_image(image, label, img_size=(224, 224)):
    """Resize and normalize image (used for debugging only)."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def _to_xy(example, img_size=(224, 224)):
    """Convert TFDS dict to (image, label) pair for model input."""
    img = tf.image.resize(example["image"], img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # Keep 0..255; model applies preprocess_input
    y = tf.cast(example["label"], tf.int32)
    return img, y

# =====================================================
# Dataset Loader
# =====================================================
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

def _preprocess_example(example, img_size):
    """
    Convert TFDS example dict → (image, label) and apply preprocessing.
    """
    image = example["image"]
    label = example["label"]
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def load_data(batch_size=8, img_size=(160, 160), shuffle_buffer=256, limit_samples=False):
    """
    Load and prepare the Food-101 dataset (TFDS) with low-memory defaults.
    Returns (train_ds, val_ds, info).
    """
    print(f" Loading Food-101 dataset from TensorFlow Datasets at size {img_size}...")
    (train_raw, val_raw), info = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
        as_supervised=False     # keep dict so we can access "image" / "label"
    )

    # Map preprocessing with parallel calls
    train_ds = (
        train_raw
        .map(lambda ex: _preprocess_example(ex, img_size), num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buffer)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_raw
        .map(lambda ex: _preprocess_example(ex, img_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Optional small-sample mode for testing stability
    if limit_samples:
        train_ds = train_ds.take(2000)
        val_ds   = val_ds.take(500)
        print("️ Using limited dataset sample for quick testing (2000/500).")

    print(f" TFDS Food-101 ready: {info.splits['train'].num_examples} training images, "
          f"{info.splits['validation'].num_examples} validation images.")
    return train_ds, val_ds, info


# =====================================================
# Compatibility Wrapper
# =====================================================
def get_datasets(batch_size=32):
    """Return (train_ds, test_ds, class_names) for legacy/test code."""
    train_ds, test_ds, info = load_data(batch_size=batch_size)
    class_names = info.features["label"].names
    return train_ds, test_ds, class_names

# =====================================================
# Debug Run
# =====================================================
if __name__ == "__main__":
    train_ds, test_ds, info = load_data()
    print(" Preprocessing pipeline ready")
    print("Classes:", info.features["label"].names[:10])

    # Inspect a few batches
    for i, (images, labels) in enumerate(train_ds.take(2)):
        print(f"Batch {i+1}: {images.shape}, {labels.shape}")
