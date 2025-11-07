# src/data_preprocessing.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# 🌱 Data Augmentation (for training only)
# =====================================================
def get_augmentation_layer(strength="medium"):
    """Return a tuned data augmentation layer for Food-101."""
    if strength == "strong":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.15)
        ], name="augmentation_strong")
    else:
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
        ], name="augmentation_medium")

# =====================================================
# 🧩 Preprocessing
# =====================================================
def preprocess_image(image, label=None, img_size=(224, 224)):
    """
    Resize & preprocess image for MobileNetV2.
    If label is None, return only image (for inference compatibility).
    """
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    if label is not None:
        return image, tf.cast(label, tf.int32)
    return image

def _preprocess_example(example, img_size):
    """Convert TFDS example dict → (image, label)."""
    return preprocess_image(example["image"], example["label"], img_size)

# =====================================================
# 📦 Dataset Loader
# =====================================================
def load_data(batch_size=32, img_size=(224, 224), limit_samples=False):
    """Load Food-101 with preprocessing and optional sample limit."""
    print(f"📥 Loading Food-101 dataset at {img_size}...")
    (train_raw, val_raw), info = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
        as_supervised=False,
    )

    # Map preprocessing + shuffle/batch/prefetch
    train_ds = (
        train_raw
        .map(lambda ex: _preprocess_example(ex, img_size), num_parallel_calls=AUTOTUNE)
        .shuffle(512)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw
        .map(lambda ex: _preprocess_example(ex, img_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    if limit_samples:
        train_ds = train_ds.take(2000)
        val_ds = val_ds.take(500)
        print("⚠️ Using limited dataset sample for quick tests (2000 train / 500 val).")

    print(f"✅ Loaded Food-101: {info.splits['train'].num_examples} training, "
          f"{info.splits['validation'].num_examples} validation images.")
    return train_ds, val_ds, info

# =====================================================
# 🧪 Legacy Wrapper
# =====================================================
def get_datasets(batch_size=32):
    train_ds, val_ds, info = load_data(batch_size=batch_size)
    return train_ds, val_ds, info.features["label"].names

# =====================================================
# 🧾 Debug
# =====================================================
if __name__ == "__main__":
    train_ds, val_ds, info = load_data(limit_samples=True)
    print("Classes:", info.features["label"].names[:5])
    for i, (imgs, labels) in enumerate(train_ds.take(1)):
        print(f"Batch {i+1}: {imgs.shape}, {labels.numpy()[:5]}")
