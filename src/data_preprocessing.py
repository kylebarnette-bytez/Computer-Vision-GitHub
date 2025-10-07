import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# =========================
# Data Augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# =========================
# Preprocessing Functions
# =========================
def preprocess_image(image, label, img_size=(224, 224)):
    """Resize and normalize image."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_datasets(dataset, batch_size=32, shuffle=False, augment=False):
    """Prepare dataset with shuffle, augmentation, batching, prefetching."""
    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_data(batch_size=32):
    """
    Load Food-101 dataset, preprocess, and return train/test splits.
    """
    dataset, info = tfds.load("food101", with_info=True, as_supervised=True)
    train_ds = dataset["train"]
    test_ds = dataset["validation"]

    train_ds = prepare_datasets(train_ds, batch_size=batch_size, shuffle=True, augment=True)
    test_ds = prepare_datasets(test_ds, batch_size=batch_size, shuffle=False, augment=False)

    return train_ds, test_ds, info

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