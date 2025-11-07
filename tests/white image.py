import tensorflow as tf, numpy as np
model = tf.keras.models.load_model("models/mobilenetv2_food101_20251020-104403-ft30_e03.keras", compile=False)

# Black image with one bright pixel
img = np.zeros((1, 224, 224, 3), dtype=np.float32)
img[0, 100, 100, :] = 1.0
pred_spot = model(img)
print("Predicted index on SINGLE BRIGHT SPOT image:", int(np.argmax(pred_spot)))

