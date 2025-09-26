# scripts/train.py
import os
from src.data_preprocessing import load_data
from src.model import build_model, compile_model, train_model

# Resolve project root (directory above /scripts)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mobilenetv2_food101.h5")

def main():
    print("📦 Loading dataset...")
    train_ds, test_ds, info = load_data(batch_size=32)

    # Print some info for debugging
    print(f"Training batches: {len(train_ds)} | Test batches: {len(test_ds)}")
    num_classes = info.features["label"].num_classes
    print(f"Detected {num_classes} classes")

    print("🔨 Building model...")
    model = build_model(num_classes)
    model = compile_model(model)

    print("🚀 Starting training (1 epoch test)...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    history = train_model(
        model,
        train_ds,
        test_ds,
        epochs=1,                # increase later
        save_path=MODEL_PATH,
        callbacks=[]
    )

    if os.path.exists(MODEL_PATH):
        print(f"✅ Training complete. Model saved at: {MODEL_PATH}")
    else:
        print("❌ Training finished but model file not found!")

if __name__ == "__main__":
    main()
