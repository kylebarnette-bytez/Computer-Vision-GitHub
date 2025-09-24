# Food Calorie Estimator

## 📌 Objective
Recognize different foods from images (Food-101 dataset) and estimate calorie content using the Edamam API.

## 📂 Project Structure
- `src/` → preprocessing pipeline, model code (to be added)
- `tests/` → test scripts
- `notebooks/` → experiments & visualization
- `requirements.txt` → dependencies
- `.gitignore` → ignores virtual environment, IDE files, datasets

## ⚙️ Setup
```bash
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
