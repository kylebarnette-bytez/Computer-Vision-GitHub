import os
import requests
import pandas as pd

# ✅ correct endpoint from your Swagger docs
API_URL = "http://127.0.0.1:8000/upload-image"

results = []
# ✅ relative path to your test_images folder
root = os.path.join(os.path.dirname(__file__), "../test_images")
root = os.path.abspath(root)

for label in os.listdir(root):
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # skip non-images
        if not (file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png")):
            continue

        with open(path, "rb") as img:
            try:
                r = requests.post(API_URL, files={"file": img})
                data = r.json()
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")
                continue

        # ✅ match backend response keys
        results.append({
            "filename": file,
            "true_label": label,
            "pred_label": data.get("predicted_food"),
            "calories": data.get("calories"),
            "price": data.get("price")
        })

# ✅ save + print results
df = pd.DataFrame(results)
df["correct"] = df.true_label == df.pred_label
print(df.head())
print(f"\nModel accuracy: {df.correct.mean() * 100:.2f}%")

df.to_csv("test_results.csv", index=False)
print("✅ Results saved to test_results.csv")
