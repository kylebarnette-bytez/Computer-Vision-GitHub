import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_preprocessing import get_augmentation_layer

os.environ["KERAS_HOME"] = os.path.expanduser("~/.keras")


def build_model(num_classes, save_path=None, use_augmentation=True):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = inputs
    if use_augmentation:
        x = get_augmentation_layer()(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def compile_model(model, learning_rate=1e-4):
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_default_callbacks(save_path, patience=5):
    """Return a list of robust default callbacks."""
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )
    return [checkpoint, early_stop, reduce_lr]


def train_model(model: tf.keras.Model,
                train_ds,
                val_ds,
                epochs: int,
                save_path: str,
                callbacks=None):
    """Train the model with optional callbacks and save it."""
    if callbacks is None:
        callbacks = get_default_callbacks(save_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"✅ Best model checkpoint saved to {save_path}")
    return history
