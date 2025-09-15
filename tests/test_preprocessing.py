import matplotlib.pyplot as plt
from src.data_preprocessing import load_data, prepare_datasets

# Load and preprocess
train_ds, test_ds, info = load_data()
train_ds, test_ds = prepare_datasets(train_ds, test_ds)

# Print dataset info
print("Classes:", info.features["label"].names[:10])
print("Train batches:", train_ds)

# Visualize a batch
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(info.features["label"].int2str(labels[i].numpy()))
        plt.axis("off")
    plt.show()
