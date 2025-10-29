# src/api_integration.py
"""
Edamam API integration and calorie lookup utilities.

Provides multi-tiered calorie estimation with automatic fallback
to local heuristics when the Edamam API rate limit or errors occur.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv


# ------------------------------------------------------------
# 🔑 Credential Loader
# ------------------------------------------------------------
def get_edamam_credentials():
    """Return (app_id, app_key) from .env or environment."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return os.getenv("EDAMAM_APP_ID"), os.getenv("EDAMAM_APP_KEY")


# ------------------------------------------------------------
# 🧩 API Helper Functions
# ------------------------------------------------------------
def _request_json(url, params=None, method="GET", json=None):
    """Unified request handler with 429 detection and safe JSON parsing."""
    try:
        if method == "POST":
            r = requests.post(url, params=params, json=json, timeout=10)
        else:
            r = requests.get(url, params=params, timeout=10)

        if r.status_code == 429:
            print("⚠️  Edamam rate limit reached.")
            return "RATE_LIMIT"

        if r.status_code != 200:
            print(f"⚠️  Edamam request failed: {r.status_code}")
            return None

        return r.json()
    except Exception as e:
        print(f"⚠️  Network error: {e}")
        return None


def _get_food_database_calories(food_name, app_id, app_key):
    url = "https://api.edamam.com/api/food-database/v2/parser"
    params = {"app_id": app_id, "app_key": app_key, "ingr": food_name}
    data = _request_json(url, params)
    if data in (None, "RATE_LIMIT"):
        return data
    for path in [
        ("parsed", 0, "food", "nutrients", "ENERC_KCAL"),
        ("hints", 0, "food", "nutrients", "ENERC_KCAL"),
    ]:
        try:
            d = data
            for p in path:
                d = d[p]
            return d
        except Exception:
            continue
    return None


def _get_nutrition_data_calories(food_name, app_id, app_key):
    url = "https://api.edamam.com/api/nutrition-data"
    params = {"app_id": app_id, "app_key": app_key, "ingr": f"1 serving {food_name}"}
    data = _request_json(url, params)
    if data in (None, "RATE_LIMIT"):
        return data
    return data.get("calories")


def _get_nutrition_data_calories_with_variants(food_name, app_id, app_key):
    """Try multiple phrasing variants for better hit rate."""
    name = food_name.replace("_", " ").strip().lower()
    canonical_map = {
        "pizza": ["1 slice pizza"],
        "ramen": ["1 bowl ramen"],
        "sushi": ["6 pieces sushi"],
        "taco": ["2 tacos"],
        "ice cream": ["1 cup ice cream"],
        "spaghetti bolognese": ["1 serving spaghetti bolognese"],
    }

    candidates = [f"1 serving {name}", name]
    if name in canonical_map:
        candidates += canonical_map[name]

    for phrase in candidates:
        cals = _get_nutrition_data_calories(phrase, app_id, app_key)
        if cals == "RATE_LIMIT":
            return "RATE_LIMIT"
        if isinstance(cals, (int, float)) and cals > 0:
            print(f"✅  Found via nutrition-data: {phrase} → {cals} kcal")
            return cals
    return None


def _get_nutrition_details_calories(food_name, app_id, app_key):
    """Use Nutrition Details API (POST)."""
    url = f"https://api.edamam.com/api/nutrition-details?app_id={app_id}&app_key={app_key}"
    headers = {"Content-Type": "application/json"}
    name = food_name.replace("_", " ").strip()
    for phrase in [f"1 serving {name}", f"1 {name}", name]:
        data = _request_json(url, method="POST", json={"ingr": [phrase]})
        if data == "RATE_LIMIT":
            return "RATE_LIMIT"
        if data and isinstance(data.get("calories"), (int, float)) and data["calories"] > 0:
            print(f"✅  Found via nutrition-details: {phrase} → {data['calories']} kcal")
            return data["calories"]
    return None


# ------------------------------------------------------------
# 🍽️  Main Public Function
# ------------------------------------------------------------
def get_calories(food_name: str):
    app_id, app_key = get_edamam_credentials()
    if not app_id or not app_key:
        print("⚠️  Missing Edamam credentials.")
        return 250  # generic fallback

    # Primary sequence of lookups
    for fn in (
        _get_nutrition_data_calories_with_variants,
        _get_nutrition_details_calories,
        _get_food_database_calories,
    ):
        cals = fn(food_name, app_id, app_key)
        if cals == "RATE_LIMIT":
            print("⚠️  Rate-limit fallback engaged.")
            return _local_fallback(food_name)
        if isinstance(cals, (int, float)) and cals > 0:
            return round(cals)

    # All failed → fallback
    print("⚠️  No valid API result, using heuristic fallback.")
    return _local_fallback(food_name)


# ------------------------------------------------------------
# 🧭 Local heuristic fallback
# ------------------------------------------------------------
def _local_fallback(food_name: str):
    fallback = {
        "hamburger": 354,
        "pizza": 285,
        "hot dog": 151,
        "lasagna": 350,
        "french fries": 365,
        "sushi": 300,
        "ramen": 500,
        "taco": 170,
        "spaghetti bolognese": 350,
        "ice cream": 273,
        "chicken salad": 350,
        "peking duck": 335,
    }
    return fallback.get(food_name.replace("_", " ").lower(), 250)


# ------------------------------------------------------------
# Backward-compat alias
# ------------------------------------------------------------
def query_calories(food_name: str):
    """Compatibility alias for legacy imports."""
    return get_calories(food_name)
