from src.model import build_model, compile_model, train_model
from src.data_preprocessing import get_datasets

# Get datasets (small batch size for speed)
train_ds, test_ds, class_names = get_datasets(batch_size=16)

# Build + compile model
model = build_model(num_classes=len(class_names))
model = compile_model(model)

# Train 1 epoch just to check pipeline works
print("🚀 Starting quick training test (1 epoch)...")
history = train_model(model, train_ds, test_ds, epochs=1)

print("✅ Training loop ran successfully")
