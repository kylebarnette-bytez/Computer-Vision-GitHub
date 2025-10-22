# scripts/train.py
import os, argparse, numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data_preprocessing import load_data
from src.model import build_model, compile_model  # your existing model.py

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def find_mobilenet(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and "mobilenetv2" in l.name.lower():
            return l
    return None

def ensure_compiled(model, lr):
    # Try using the project's compile_model with lr kwarg (if supported)
    try:
        return compile_model(model, lr=lr)
    except TypeError:
        # Fallback: either compile_model(model) without lr, or fully recompile
        try:
            model = compile_model(model)
            # Try to set lr on the returned optimizer
            try:
                model.optimizer.learning_rate.assign(lr)
                return model
            except Exception:
                pass
        except Exception:
            pass
        # Final fallback: compile ourselves with sane defaults
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
        )
        return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--val-steps", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr-head", type=float, default=3e-3)
    ap.add_argument("--ft-epochs", type=int, default=0)
    ap.add_argument("--unfreeze-from", type=int, default=-30)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""

    print("📦 Loading dataset...")
    train_ds, test_ds, info = load_data(batch_size=args.batch)
    num_classes = info.features["label"].num_classes

    # Peek a batch to sanity-check labels and shapes
    x0, y0 = next(iter(train_ds))
    print(f"🔎 Sample batch: x={tuple(x0.shape)} y={tuple(y0.shape)} "
          f"labels[min,max]=[{int(tf.reduce_min(y0))},{int(tf.reduce_max(y0))}]")

    print("🔨 Building & compiling model...")
    model = build_model(num_classes, use_augmentation=not args.no_augment)
    model = ensure_compiled(model, lr=args.lr_head)

    # Trainable params sanity
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"🧮 Trainable params: {trainable_params}")
    model.summary(line_length=120)

    # Checkpoint pattern (epoch number + timestamp + tag)
    ckpt_pattern = os.path.join(
        MODELS_DIR,
        f"mobilenetv2_food101_{run_id}{tag}_e{{epoch:02d}}.keras"
    )

    callbacks = [
        ModelCheckpoint(
            filepath=ckpt_pattern,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    print(f"🚀 Training: epochs={args.epochs}, steps={args.steps}, val_steps={args.val_steps}, "
          f"augment={'off' if args.no_augment else 'on'}")
    hist_main = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_steps=args.val_steps,
        callbacks=callbacks,
        verbose=1
    )

    total_epochs = len(hist_main.history.get("loss", []))

    # Optional fine-tune phase
    if args.ft_epochs > 0:
        base = find_mobilenet(model)
        if base is not None:
            print(f"🪄 Fine-tuning top of {base.name}: unfreezing from index {args.unfreeze_from}...")
            base.trainable = True
            if args.unfreeze_from is not None:
                cutoff = args.unfreeze_from if args.unfreeze_from >= 0 else len(base.layers) + args.unfreeze_from
                for i, layer in enumerate(base.layers):
                    layer.trainable = (i >= cutoff)
            model = ensure_compiled(model, lr=args.lr_ft)
            hist_ft = model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=args.ft_epochs,
                callbacks=callbacks,
                verbose=1
            )
            total_epochs += len(hist_ft.history.get("loss", []))
        else:
            print("⚠️ Could not locate MobileNetV2 submodel for fine-tuning; skipping FT phase.")

    # Save final artifacts with total epoch count in name
    final_base = f"mobilenetv2_food101_{run_id}{tag}_e{total_epochs:02d}"
    final_model_path = os.path.join(MODELS_DIR, f"{final_base}.keras")
    final_weights_path = os.path.join(MODELS_DIR, f"{final_base}.weights.h5")

    print("💾 Saving final artifacts...")
    model.save(final_model_path)
    model.save_weights(final_weights_path)
    print(f"✅ Saved final model:   {final_model_path}")
    print(f"✅ Saved final weights: {final_weights_path}")

if __name__ == "__main__":
    main()