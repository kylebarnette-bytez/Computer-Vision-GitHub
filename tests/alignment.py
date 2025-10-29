import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_PATH = os.path.join(BASE_DIR, "models", "food101_classes.txt")

print(f"🔍 Checking label file at: {LABEL_PATH}")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"⚠️ Label file not found at {LABEL_PATH}")

with open(LABEL_PATH) as f:
    labels = [l.strip() for l in f.readlines() if l.strip()]

print("✅ Label count:", len(labels))
print("First 5:", labels[:5])
print("Last 5:", labels[-5:])
