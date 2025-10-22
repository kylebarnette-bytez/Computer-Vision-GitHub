# scripts/eval_model.py
import os, argparse, tensorflow as tf
from src.data_preprocessing import load_data
from src.model import build_model, compile_model

def resolve_model_path(name):
    root = os.path.dirname(os.path.dirname(__file__))
    p = name if os.path.isabs(name) else os.path.join(root, name)
    if os.path.exists(p): return p
    q = os.path.join(root, "models", name)
    if os.path.exists(q): return q
    raise FileNotFoundError(name)

def pick(d, *keys):
    for k in keys:
        if k in d: return d[k]
    return None

def compute_top5(model, ds, num_classes):
    m = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    for x, y in ds:
        y1h = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)
        m.update_state(y1h, model(x, training=False))
    return float(m.result().numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--as-weights", dest="as_weights", action="store_true")
    args = ap.parse_args()

    path = resolve_model_path(args.model)
    _, test_ds, info = load_data(batch_size=args.batch)
    num_classes = info.features["label"].num_classes

    if args.as_weights or path.lower().endswith(".weights.h5"):
        model = build_model(num_classes, use_augmentation=False)
        model = compile_model(model)  # use repo’s defaults
        model.load_weights(path)
    else:
        try:
            model = tf.keras.models.load_model(path, compile=False)
            model = compile_model(model)  # standardize metrics/loss
        except Exception:
            model = build_model(num_classes, use_augmentation=False)
            model = compile_model(model)
            model.load_weights(path)

    results = model.evaluate(test_ds, verbose=1)
    if not isinstance(results, (list, tuple)):
        results = [results]
    names = model.metrics_names  # starts with 'loss'
    metrics = dict(zip(names, results))

    # Try common names for top-1 and top-5
    top1 = pick(metrics, "accuracy", "sparse_categorical_accuracy", "categorical_accuracy")
    top5 = pick(metrics, "top5", "top_k_categorical_accuracy")

    # If top-5 wasn't part of compiled metrics, compute it now
    if top5 is None:
        top5 = compute_top5(model, test_ds, num_classes)

    root = os.path.dirname(os.path.dirname(__file__))
    rel = os.path.relpath(path, root)
    print(f"{rel} | loss={metrics.get('loss', results[0]):.4f} top1={float(top1):.4f} top5={float(top5):.4f}")

if __name__ == "__main__":
    main()