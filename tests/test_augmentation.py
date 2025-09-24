# tests/test_augmentation.py
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data, data_augmentation

# Load a small batch (no augmentation yet)
train_ds, _, info = load_data(batch_size=9)

# Take 1 batch
for images, labels in train_ds.take(1):
    plt.figure(figsize=(12, 6))

    for i in range(9):
        # Left: original
        ax = plt.subplot(3, 6, 2*i + 1)
        plt.imshow(images[i].numpy())
        plt.title("Orig")
        plt.axis("off")

        # Right: augmented
        augmented_img = data_augmentation(images[i], training=True)
        ax = plt.subplot(3, 6, 2*i + 2)
        plt.imshow(augmented_img.numpy())
        plt.title("Aug")
        plt.axis("off")

    plt.suptitle("Original vs Augmented Images", fontsize=14)
    plt.show()
    break
