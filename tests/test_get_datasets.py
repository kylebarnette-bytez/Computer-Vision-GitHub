# tests/test_get_datasets.py
from src.data_preprocessing import get_datasets

# Load dataset
train_ds, test_ds, class_names = get_datasets(batch_size=32)

print("✅ Dataset ready for model training")
print("Number of classes:", len(class_names))
print("First 10 classes:", class_names[:10])

# Inspect 1 batch
for images, labels in train_ds.take(1):
    print("Train batch shape:", images.shape, labels.shape)
    assert images.shape[1:] == (224, 224, 3), "Images not resized correctly"
    assert images.numpy().min() >= 0.0 and images.numpy().max() <= 1.0, "Images not normalized"
    break
