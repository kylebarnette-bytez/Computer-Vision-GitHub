# 🍽️ Food Calorie Estimator

## 📌 Objective
Recognize different foods from images (using the **Food-101 dataset**) and estimate calorie content via the **Edamam API**.  
This project combines **Computer Vision** and **Deep Learning** with practical nutrition applications.  

---

## 📂 Project Structure
- `src/` → core source code  
  - `data_preprocessing.py` – dataset setup, normalization, augmentation  
  - `model.py` – MobileNetV2 model definition & training  
  - `api_integration.py` – Edamam API requests for calorie info  
  - `app.py` – Flask app for demo interface  
  - `utils.py` – shared helper functions  
- `tests/` → unit tests for preprocessing, model, and API  
- `models/` → trained `.h5` models (ignored in Git)  
- `notebooks/` → Jupyter notebooks for exploration & experiments  
- `requirements.txt` → Python dependencies  
- `.gitignore` → ignores virtual environment, IDE files, datasets  

---

## 👥 Team Roles
- **Person A – Data & Preprocessing Lead**  
  - Dataset download, preprocessing, augmentation, train/test split  

- **Person B – Model Training Lead**  
  - MobileNetV2 training, fine-tuning, evaluation, saving `.h5` model  

- **Person C – API & Integration Lead**  
  - Edamam API integration, building demo Flask app, final integration  

---

## 📅 Milestones
- ✅ **Sept 20 – Oct 1**: Dataset setup & preprocessing complete  
- ✅ **Oct 5**: Data augmentation & final train/test split  
- ⏳ **Oct – Nov**: Model training & API integration  
- 🎤 **Nov 13**: Final presentation (demo + slides)  
- 📦 **Nov 25**: Final project submission  

---

## ⚙️ Setup
```bash
# Clone the repository
git clone git@github.com:<YourUsername>/FoodCalorieEstimator.git
cd FoodCalorieEstimator

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
