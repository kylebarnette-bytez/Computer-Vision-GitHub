# src/data_preprocessing.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# Convenience accessor so other modules (e.g., model building) can attach the
# same augmentation pipeline without importing symbols directly.
def get_augmentation_layer():
    return data_augmentation

# =========================
# Preprocessing Functions
# =========================
def preprocess_image(image, label, img_size=(224, 224)):
    """Resize and normalize image."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def _to_xy(example, img_size=(224, 224)):
    img = tf.image.resize(example["image"], img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # keep 0..255; model applies preprocess_input
    y = tf.cast(example["label"], tf.int32) # sparse int labels
    return img, y

    train_ds = prepare_datasets(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    test_ds = prepare_datasets(test_ds, batch_size=batch_size, shuffle=False, augment=False)

    return train_ds, test_ds, info


def get_datasets(batch_size=32):
    """Compatibility wrapper expected by tests.

    Returns:
        train_ds, test_ds, class_names
    """
    train_ds, test_ds, info = load_data(batch_size=batch_size)
    class_names = info.features["label"].names
    return train_ds, test_ds, class_names

# =========================
# Debug Run
# =========================
if __name__ == "__main__":
    train_ds, test_ds, info = load_data()
    print(" Preprocessing pipeline ready")
    print("Classes:", info.features["label"].names[:10])

    # Take multiple batches to verify iteration
    for i, (images, labels) in enumerate(train_ds.take(3)):
        print(f"Batch {i + 1}: {images.shape}, {labels.shape}")
