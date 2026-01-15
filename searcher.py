import os
import requests
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

search_id = os.getenv('SEARCH_ID')
api_key = os.getenv('API_KEY')


def google_search(query, search_id: str, api_key: str) -> dict:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': search_id,
        'q': query,
        'num': 5
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    results = google_search("Погода в Спб", search_id, api_key)
