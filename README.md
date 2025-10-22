# Food Calorie Estimator

## 📌 Objective

Recognize different foods from images (Food-101 dataset) and estimate calorie
content using the Edamam API.

## 📂 Project Structure

- `src/` → preprocessing pipeline, model code (to be added)
- `tests/` → test scripts
- `notebooks/` → experiments & visualization
- `requirements.txt` → dependencies
- `.gitignore` → ignores virtual environment, IDE files, datasets

## ⚙️ Setup

````bash
git clone https://github.com/<YourUsername>/FoodCalorieEstimator.git
cd FoodCalorieEstimator
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

## 🚀 Current Progress

- ✅ Preprocessing pipeline (resize, normalize, augment)
- ✅ Dataset loader (Food-101 train/test splits)
- ✅ Model build (MobileNetV2 backbone with transfer learning)
- ✅ Basic training test script (runs 1 epoch)
- ✅ Gitignore updated (ignores models, logs, datasets)

Next steps:
- [ ] Train full model on Food-101
- [ ] Save checkpoints & logs
- [ ] Integrate API for calorie estimation

## ▶️ Run the API locally

1) Ensure a trained model exists at `models/mobilenetv2_food101.h5`.
   - You can produce a quick sanity model by running:
     ```bash
     python scripts/train.py
     ```

2) Set Edamam credentials (required for calorie lookup):
   ```bash
   export EDAMAM_APP_ID=your_app_id
   export EDAMAM_APP_KEY=your_app_key
````

3. Start the FastAPI server:
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```

4. Test the endpoint with an image:
   ```bash
   curl -F "file=@/path/to/food.jpg" http://localhost:8000/predict
   ```

The response includes the predicted Food-101 label and an estimated calorie
value when available.
