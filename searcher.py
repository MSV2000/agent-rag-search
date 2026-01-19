import os
import re
import requests
import trafilatura
from dotenv import load_dotenv
from html import unescape

# Загружаем переменные из .env файла
load_dotenv()

SEARCH_ID = os.getenv('SEARCH_ID')
API_KEY = os.getenv('API_KEY')


def fetch_html(url: str) -> str | None:
    """
    Загружает HTML-страницу по указанному URL

    Args:
        url: URL страницы для загрузки

    Returns:
        str | None: HTML-код страницы в виде строки или None при ошибке
    """
    try:
        r = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def normalize_text(text: str) -> str:
    """
    Нормализует текст, удаляя HTML-сущности и лишние пробелы

    Args:
        text: Исходный текст

    Returns:
        str: Нормализованный текст
    """
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_main_text(html: str) -> str | None:
    """
    Извлекает основной текст статьи из HTML-документа

    Args:
        html: HTML-код страницы

    Returns:
        str | None: Основной текст страницы или None, если извлечение не удалось
    """
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    )
    if not text:
        return None
    return normalize_text(text)


def collect_for_llm(search_response: dict) -> str:
    """
    Формирует текст из результатов поисковой выдачи для передачи в LLM

    Args:
        search_response: Ответ поискового API

    Returns:
        str: Объединённый текст, пригодный для передачи в LLM
    """
    documents = []

    for item in search_response.get("items", []):
        url = item.get("link")
        if not url:
            continue

        html = fetch_html(url)
        if not html:
            continue

        text = extract_main_text(html)
        if not text:
            continue

        documents.append(
            f"Источник: {url}\n{text}"
        )

    return "\n===========\n".join(documents)


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
    results = google_search("Погода в Спб", SEARCH_ID, API_KEY)
    print(results)
