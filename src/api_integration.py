import requests

def get_calories(food_name: str):
    url = "https://api.edamam.com/api/food-database/v2/parser"
    params = {
        "app_id": "YOUR_APP_ID",
        "app_key": "YOUR_APP_KEY",
        "ingr": food_name
    }
    r = requests.get(url, params=params)
    data = r.json()
    try:
        return data["parsed"][0]["food"]["nutrients"]["ENERC_KCAL"]
    except (KeyError, IndexError):
        return None
