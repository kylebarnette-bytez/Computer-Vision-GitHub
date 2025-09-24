# tests/test_preprocessing.py
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data

# Load the dataset
train_ds, test_ds, info = load_data()

print("Classes:", info.features["label"].names[:10])

# Take 1 batch and visualize a few images
for images, labels in train_ds.take(1):
    print("Batch shape:", images.shape, labels.shape)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(info.features["label"].int2str(labels[i].numpy()))
        plt.axis("off")
    plt.show()
