# src/data_preprocessing.py

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Augmentation pipeline
# =========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# =========================
# Preprocessing functions
# =========================
def preprocess_image(image, label, img_size=(224, 224)):
    """Resize and normalize image."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    return image, label


import tensorflow as tf


def prepare_datasets(dataset, batch_size=32, shuffle=False, augment=False):
    """Preprocess, batch, and prefetch datasets."""
    # Apply resize + normalization
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle if training
    if shuffle:
        dataset = dataset.shuffle(1000)

    # Apply augmentation if requested
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Batch + prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset




def load_data(batch_size=32):
    dataset, info = tfds.load("food101", with_info=True, as_supervised=True)
    train_ds = dataset["train"]
    test_ds = dataset["validation"]

    train_ds = prepare_datasets(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    test_ds = prepare_datasets(test_ds, batch_size=batch_size, shuffle=False, augment=False)

    return train_ds, test_ds, info

def get_datasets(batch_size=32):
    """
    Public function for teammates:
    Returns train_ds, test_ds, class_names
    """
    train_ds, test_ds, info = load_data(batch_size=batch_size)
    class_names = info.features["label"].names
    return train_ds, test_ds, class_names

def get_datasets(batch_size=32):
    """
    Public function for teammates.
    Loads Food-101, preprocesses, augments train, and returns ready-to-train datasets.
    Returns:
        train_ds, test_ds, class_names
    """
    train_ds, test_ds, info = load_data(batch_size=batch_size)
    class_names = info.features["label"].names
    return train_ds, test_ds, class_names

def get_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

# =========================
# Quick debug (run directly)
# =========================
if __name__ == "__main__":
    train_ds, test_ds, info = load_data()
    print("Classes:", info.features["label"].names[:10])
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape, labels.shape)
