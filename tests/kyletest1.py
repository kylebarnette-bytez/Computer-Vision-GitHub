import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_MODEL = os.path.join(BASE_DIR, "../models/my_very_own_food101_model_v3.keras")
NEW_MODEL = os.path.join(BASE_DIR, "../models/mobilenetv2_food101_clean.h5")

# Rebuild the correct MobileNetV2 head
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"   # or None if you want random init
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
out = tf.keras.layers.Dense(101, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

# Try to copy usable weights from the old file
try:
    tmp = tf.keras.models.load_model(OLD_MODEL, compile=False)
    for layer, old in zip(model.layers, tmp.layers):
        if layer.weights and old.weights:
            layer.set_weights(old.get_weights())
    print("✅ Copied overlapping weights from old model")
except Exception as e:
    print("⚠️ Could not load weights from old model:", e)

# Save in legacy HDF5 format (very stable)
model.save(NEW_MODEL)
print("✅ Saved clean model:", NEW_MODEL)
