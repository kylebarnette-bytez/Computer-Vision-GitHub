import os
from pathlib import Path
import requests
from dotenv import load_dotenv


def get_edamam_credentials():
    """Return Edamam app id/key from environment variables.

    Expected variables:
      - EDAMAM_APP_ID
      - EDAMAM_APP_KEY
    """
    # Load .env once per process (safe to call multiple times)
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return os.getenv("EDAMAM_APP_ID"), os.getenv("EDAMAM_APP_KEY")


def _get_food_database_calories(food_name: str, app_id: str, app_key: str):
    url = "https://api.edamam.com/api/food-database/v2/parser"
    params = {"app_id": app_id, "app_key": app_key, "ingr": food_name}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        # First try strict parsed
        try:
            return data["parsed"][0]["food"]["nutrients"]["ENERC_KCAL"]
        except (KeyError, IndexError, TypeError):
            pass
        # Then try hints (common for generic phrases)
        try:
            return data["hints"][0]["food"]["nutrients"]["ENERC_KCAL"]
        except (KeyError, IndexError, TypeError):
            return None
    except Exception:
        return None


def _get_nutrition_data_calories(food_name: str, app_id: str, app_key: str):
    # Nutrition Analysis API (simple): returns overall calories for an ingredient line
    url = "https://api.edamam.com/api/nutrition-data"
    # Prefix with a generic serving to improve parsing
    params = {"app_id": app_id, "app_key": app_key, "ingr": f"1 serving {food_name}"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        # 'calories' is a top-level field in nutrition-data API
        return data.get("calories")
    except Exception:
        return None


def _get_nutrition_data_calories_with_variants(food_name: str, app_id: str, app_key: str):
    """Try multiple phrasing variants to improve Nutrition API matching."""
    name = food_name.replace("_", " ").strip().lower()

    # Canonical phrasing for some common items
    canonical_map = {
        "hamburger": ["1 hamburger sandwich", "1 cheeseburger"],
        "hot dog": ["1 hot dog"],
        "french fries": ["1 serving french fries"],
        "pizza": ["1 slice pizza"],
        "ramen": ["1 bowl ramen"],
        "sushi": ["6 pieces sushi"],
        "taco": ["2 tacos"],
        "tacos": ["2 tacos"],
        "ice cream": ["1 cup ice cream"],
        "spaghetti bolognese": ["1 serving spaghetti bolognese"],
        "chicken salad": ["1 serving chicken salad"],
        "lasagna": ["1 piece meat lasagna", "1 slice lasagna", "1 serving lasagna"],
    }

    # Synonyms that help Edamam parsing
    synonyms = {
        "lasagna": ["lasagne", "meat lasagna", "lasagna with meat sauce", "cheese lasagna"],
        "hamburger": ["beef burger", "cheeseburger"],
    }

    candidate_names = {name}
    for base, syns in synonyms.items():
        if name == base:
            candidate_names.update(syns)

    candidates: list[str] = []
    if name in canonical_map:
        candidates.extend(canonical_map[name])

    # Generic variants across candidate names
    measure_variants = [
        "1 serving {}",
        "1 piece {}",
        "1 slice {}",
        "1 bowl {}",
        "1 cup {}",
        "1 {}",
        "{}",
        "1 serving of {}",
        "1 piece of {}",
        "1 slice of {}",
    ]
    for n in candidate_names:
        for tmpl in measure_variants:
            candidates.append(tmpl.format(n))

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for phrase in unique_candidates:
        cals = _get_nutrition_data_calories(phrase, app_id, app_key)
        if isinstance(cals, (int, float)) and cals > 0:
            return cals
    return None


def _get_nutrition_details_calories(food_name: str, app_id: str, app_key: str):
    """Nutrition Details API via POST with ingredient list, more robust parsing."""
    url = f"https://api.edamam.com/api/nutrition-details?app_id={app_id}&app_key={app_key}"
    name = food_name.replace("_", " ").strip()
    candidates = [
        f"1 serving {name}",
        f"1 {name}",
        name,
    ]
    headers = {"Content-Type": "application/json"}
    for phrase in candidates:
        try:
            payload = {"ingr": [phrase]}
            response = requests.post(url, json=payload, headers=headers, timeout=12)
            if response.status_code != 200:
                continue
            data = response.json()
            cals = data.get("calories")
            if isinstance(cals, (int, float)) and cals > 0:
                return cals
        except Exception:
            continue
    return None


def get_calories(food_name: str):
    app_id, app_key = get_edamam_credentials()
    if not app_id or not app_key:
        return None

    # Prefer Nutrition Analysis (variants + details) for broader access; fall back to Food DB
    calories = _get_nutrition_data_calories_with_variants(food_name, app_id, app_key)
    if calories is None:
        calories = _get_nutrition_details_calories(food_name, app_id, app_key)
    if calories is None:
        calories = _get_food_database_calories(food_name, app_id, app_key)
    if calories is not None:
        return calories

    # Last-resort heuristic fallback for common dishes (approximate per serving)
    fallback = {
        "hamburger": 354,
        "pizza": 285,              # per slice
        "hot dog": 151,
        "lasagna": 350,            # per piece
        "french fries": 365,
        "sushi": 300,              # assorted pieces
        "ramen": 500,
        "taco": 170,
        "spaghetti bolognese": 350,
        "ice cream": 273,
        "chicken salad": 350,
    }
    key = food_name.replace("_", " ").strip().lower()
    return fallback.get(key)


# Backward-compatible alias for predict utilities that may import query_calories
def query_calories(food_name: str):
    return get_calories(food_name)
