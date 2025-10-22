# src/data_preprocessing.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

def get_augmentation_layer(img_size=(224, 224)):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augment")

def _to_xy(example, img_size=(224, 224)):
    img = tf.image.resize(example["image"], img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # keep 0..255; model applies preprocess_input
    y = tf.cast(example["label"], tf.int32) # sparse int labels
    return img, y

def _prepare(ds, batch_size, shuffle, img_size=(224, 224)):
    ds = ds.map(lambda e: _to_xy(e, img_size), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def load_data(batch_size=64, img_size=(224, 224)):
    splits, info = tfds.load(
        "food101",
        split=["train", "validation"],
        with_info=True,
        as_supervised=False
    )
    ds_train, ds_val = splits  # <-- unpack the list of splits
    train_ds = _prepare(ds_train, batch_size, shuffle=True,  img_size=img_size)
    test_ds  = _prepare(ds_val,   batch_size, shuffle=False, img_size=img_size)
    return train_ds, test_ds, info