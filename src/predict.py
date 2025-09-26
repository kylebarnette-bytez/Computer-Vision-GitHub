# src/predict.py
import tensorflow as tf
import numpy as np
from src.data_preprocessing import preprocess_image
from src.api_integration import query_calories

def load_class_names():
    import tensorflow_datasets as tfds
    _, info = tfds.load("food101", with_info=True, as_supervised=True)
    return info.features["label"].names

def predict_image(model, image_path, class_names, top_k=3):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img, _ = preprocess_image(img, 0)
    img = tf.expand_dims(img, 0)

    preds = model.predict(img)
    top_indices = np.argsort(preds[0])[-top_k:][::-1]

    results = []
    for idx in top_indices:
        food_name = class_names[idx].replace("_", " ")
        calories = query_calories(food_name)
        results.append({
            "food": food_name,
            "probability": float(preds[0][idx]),
            "calories": calories
        })
    return results
