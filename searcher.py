import os
import requests
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

SEARCH_ID = os.getenv('SEARCH_ID')
API_KEY = os.getenv('API_KEY')


def google_search(query, search_id: str, api_key: str) -> dict:
    """
    Выполняет поисковый запрос к Google Custom Search API и возвращает результаты поиска

    Args:
        query: Поисковый запрос
        search_id: Идентификатор поисковой системы
        api_key: API-ключ

    Returns:
        dict: JSON-объект, преобразованный в словарь

    """
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
    print(results)
