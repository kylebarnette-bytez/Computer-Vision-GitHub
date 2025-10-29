##### FIRST TRAINING OF 5 EPOCHS KYLE'S MODEL####


# """
# Train and fine-tune MobileNetV2 on Food-101 (TFDS), M1-safe baseline.
# """
#
# # ============================================================
# # 0️⃣ Environment setup (disable GPU on M1 for stability)
# # ============================================================
# import os
# os.environ["TF_METAL_ENABLE"] = "0"     # disable Apple Metal
# os.environ["APPLE_ENABLE_METAL"] = "NO"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no CUDA (harmless on Mac)
#
# # ============================================================
# # 1️⃣ Imports & path setup
# # ============================================================
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#
# import tensorflow as tf
# from src.model import build_model, compile_model, train_model, fine_tune_model
# from src.data_preprocessing import load_data   # ✅ use your refactored loader
#
# # ============================================================
# # 2️⃣ Load Food-101 dataset
# # ============================================================
# IMG_SIZE = (160, 160)       # smaller image -> lower memory use
# BATCH_SIZE = 8              # safe batch size for 16GB M1
#
# train_ds, val_ds, info = load_data(
#     batch_size=BATCH_SIZE,
#     img_size=IMG_SIZE,
#     shuffle_buffer=256,
#     limit_samples=False      # ✅ change to True for quick 2000/500 test run
# )
#
# NUM_CLASSES = info.features["label"].num_classes
# print(f"🧠 Building MobileNetV2 for {NUM_CLASSES} classes ...")
#
# # ============================================================
# # 3️⃣ Build, Compile, Train, Fine-tune
# # ============================================================
# os.makedirs("models", exist_ok=True)
#
# # ✅ Pass img_size explicitly to match dataset
# model = build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
# compile_model(model)
#
# # --- Stage 1: Head Training ---
# print("\n🚀 Starting initial head training (CPU-safe config)...")
# train_model(
#     model,
#     train_ds,
#     val_ds,
#     epochs=5,  # start smaller; increase once stable
#     save_path="models/mobilenetv2_food101_tfds_head_e05.keras"
# )
#
# # --- Stage 2: Fine-tuning (optional; only if Stage 1 is stable) ---
# print("\n🔧 Fine-tuning deeper layers ...")
# fine_tune_model(
#     model,
#     train_ds,
#     val_ds,
#     fine_tune_at=80,   # unfreeze last 80 layers
#     epochs=5,          # reduce to stay memory-safe
#     save_path="models/mobilenetv2_food101_tfds_finetuned_e05.keras"
# )
#
# print("\n✅ Training complete! Models saved in /models/")

"""
Resume fine-tuning MobileNetV2 on Food-101 from existing checkpoint.
Safe defaults for Apple M1 (160x160, batch size 8, Metal disabled).
"""

import os
import sys

# ============================================================
# 0️⃣ Environment setup for stability on M1
# ============================================================
os.environ["TF_METAL_ENABLE"] = "0"
os.environ["APPLE_ENABLE_METAL"] = "NO"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Ensure src/ directory can be imported ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# 1️⃣ Load dataset (M1-safe config)
# ============================================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

print("📦 Loading Food-101 dataset...")
(ds_train, ds_val), ds_info = tfds.load(
    "food101",
    split=["train", "validation"],
    as_supervised=True,
    shuffle_files=True,
    with_info=True
)
NUM_CLASSES = ds_info.features["label"].num_classes
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    label = tf.one_hot(label, depth=NUM_CLASSES)   # 👈 restore one-hot encoding
    return image, label


train_ds = (
    ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ============================================================
# 2️⃣ Load the existing model at 5 epochs
# ============================================================
MODEL_PATH = "models/mobilenetv2_food101_tfds_finetuned_e05.keras"
print(f"🧠 Loading checkpoint from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ============================================================
# 3️⃣ Unfreeze last 80 layers for fine-tuning
# ============================================================
fine_tune_at = len(model.layers) - 80
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in model.layers[fine_tune_at:]:
    layer.trainable = True

# ============================================================
# 4️⃣ Compile model (low LR for gentle fine-tune)
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
)

# ============================================================
# 5️⃣ Callbacks
# ============================================================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("models/mobilenetv2_food101_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
]

# ============================================================
# 6️⃣ Resume training (5 → 10 epochs)
# ============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=55,               # total target epochs
    initial_epoch=40,         # resume from epoch 5
    callbacks=callbacks
)

# ============================================================
# 7️⃣ Save new model
# ============================================================
SAVE_PATH = "models/mobilenetv2_food101_after10.keras"
model.save(SAVE_PATH)
print(f"✅ Fine-tuning complete — saved to {SAVE_PATH}")
