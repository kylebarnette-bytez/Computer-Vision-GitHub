import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32


# 🔹 Load Food-101 dataset
def load_data():
    dataset, info = tfds.load("food101", with_info=True, as_supervised=True)
    train_ds, test_ds = dataset["train"], dataset["validation"]
    return train_ds, test_ds, info


# 🔹 Preprocess: resize + normalize
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# 🔹 Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# 🔹 Prepare datasets with batching, shuffling, and prefetching
def prepare_datasets(train_ds, test_ds):
    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .map(lambda x, y: (data_augmentation(x), y))
                .shuffle(1000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))
    return train_ds, test_ds
