"""
Model definition and training utilities for the Food-101 Calorie Estimator.

This module builds a MobileNetV2-based image classifier using TensorFlow / Keras.
It provides modular functions for:
  • Building and compiling the model
  • Training with callbacks and saving checkpoints
  • Fine-tuning deeper layers for higher accuracy
  • Consistent preprocessing with inference pipeline

Author: Kyle T. Barnette et al. (CS 4337 – Official Food App)
Refactored: 2025-10-29 for final 55-epoch model alignment.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from src.data_preprocessing import get_augmentation_layer

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
os.environ["KERAS_HOME"] = os.path.expanduser("~/.keras")
IMG_SIZE = (160, 160)        # ✅ match fine-tuned model
BASE_LR = 3e-4               # initial LR for head training
FINE_TUNE_LR = 1e-5          # smaller LR for fine-tuning


# ------------------------------------------------------------
# Model construction
# ------------------------------------------------------------
def build_model(num_classes: int, use_augmentation: bool = True, img_size=IMG_SIZE) -> tf.keras.Model:
    """
    Build a MobileNetV2-based classifier.
    img_size should match the training and inference pipelines (default: 160×160).
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Start frozen

    inputs = layers.Input(shape=(*img_size, 3))
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    # ✅ identical preprocessing to predict.py
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)

    # Head layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="food101_mobilenetv2")
    return model


# ------------------------------------------------------------
# Compilation
# ------------------------------------------------------------
def compile_model(model: tf.keras.Model, learning_rate: float = BASE_LR, one_hot_labels=False) -> tf.keras.Model:
    """
    Compile the model with the correct loss depending on label format.
    If one_hot_labels=True, uses categorical_crossentropy.
    """
    loss_fn = "categorical_crossentropy" if one_hot_labels else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )
    return model


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
def get_default_callbacks(save_path: str, patience: int = 6):
    """Return improved callbacks for checkpointing and LR scheduling."""
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="max",
        verbose=1
    )
    return [checkpoint, early_stop, reduce_lr]


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_model(model: tf.keras.Model,
                train_ds,
                val_ds,
                epochs: int,
                save_path: str,
                callbacks=None):
    """Train the model and automatically save the best checkpoint."""
    if callbacks is None:
        callbacks = get_default_callbacks(save_path)

    print("\n🚀 Starting head-training (base frozen)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"✅ Best model checkpoint saved to {save_path}")
    return history


# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------
def fine_tune_model(model: tf.keras.Model,
                    train_ds,
                    val_ds,
                    fine_tune_at: int = 80,
                    epochs: int = 10,
                    save_path: str | None = None):
    """Unfreeze deeper layers of MobileNetV2 for fine-tuning."""
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name:
            base_model = layer
            break
    if base_model is None:
        raise ValueError("❌ Could not find MobileNetV2 base model in loaded model.")

    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )

    print(f"\n🔧 Fine-tuning MobileNetV2 from layer {fine_tune_at} onwards...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_default_callbacks(save_path)
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"💾 Fine-tuned model saved to {save_path}")

    return history


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def summarize_model(model: tf.keras.Model):
    """Print a concise model summary."""
    print("\n🧩 Model Summary:")
    model.summary(line_length=100)


if __name__ == "__main__":
    # Debug run
    m = build_model(num_classes=101)
    compile_model(m)
    summarize_model(m)
    print("✅ Model built successfully.")
