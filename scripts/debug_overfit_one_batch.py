import os, numpy as np, tensorflow as tf
from src.data_preprocessing import load_data
from src.model import build_model, compile_model

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    train_ds, _, info = load_data(batch_size=64)
    num_classes = info.features["label"].num_classes
    it = iter(train_ds)
    x, y = next(it)  # one batch
    print("batch shape:", x.shape, y.shape, "labels min/max:", int(tf.reduce_min(y)), int(tf.reduce_max(y)))

    model = build_model(num_classes, use_augmentation=False)
    model = compile_model(model, lr=3e-3)

    history = model.fit(
        x, y,
        epochs=20,
        verbose=2
    )
    print("Final batch acc:", history.history["accuracy"][-1])

if __name__ == "__main__":
    main()