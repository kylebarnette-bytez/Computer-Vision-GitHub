# src/model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
os.environ["KERAS_HOME"] = os.path.expanduser("~/.keras")

def build_model(num_classes, save_path=None):
    """Build a MobileNetV2-based model for Food-101."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),   # keep your dense layer
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Optionally save model structure/weights
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        model.save(save_path)
        print(f"✅ Model saved to {save_path}")

    return model


def compile_model(model, learning_rate=1e-4):
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(model, train_ds, test_ds, epochs=5):
    """Train the model on train_ds and validate on test_ds."""
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )
    return history
