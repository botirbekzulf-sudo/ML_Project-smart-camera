"""
google_search.py
Ищет характеристики и цену устройства по названию модели.

Два режима работы:

1) С Google Custom Search API (рекомендуется для реального сайта):
   - Создай Custom Search Engine: https://programmablesearchengine.google.com/
   - Получи API-ключ: https://console.cloud.google.com/apis/credentials
   - Установи переменные окружения GOOGLE_API_KEY и GOOGLE_CSE_ID

2) Без ключей (режим по умолчанию для разработки):
   - Просто формирует ссылку на google.com/search?q=... — ровно то, что
     описано в задании ("сайт открывает Google с характеристиками") —
     и открывается на фронтенде в новой вкладке через window.open().
   - Функция search_specs() при этом возвращает пустой текст характеристик,
     и preprocess.py подставит разумные значения по умолчанию, чтобы
     ML-модель всё равно могла сделать предсказание.
"""

from __future__ import annotations
import os
import urllib.parse
import requests

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")


def build_google_search_url(model_name: str) -> str:
    """Формирует ссылку на обычный поиск Google по модели устройства."""
    query = urllib.parse.quote(f"{model_name} характеристики цена")
    return f"https://www.google.com/search?q={query}"


def search_specs(model_name: str) -> dict:
    """
    Возвращает {"raw_text": str, "source_url": str, "items": [...]}.

    raw_text — сырой текст сниппетов, который пойдёт в
    preprocess.specs_text_to_features() для извлечения признаков.
    """
    search_url = build_google_search_url(model_name)

    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        # Без ключей просто отдаём ссылку — фронтенд откроет Google сам.
        return {"raw_text": "", "source_url": search_url, "items": []}

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"{model_name} характеристики цена specifications price",
        "num": 5,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"raw_text": "", "source_url": search_url, "items": [], "error": str(exc)}

    items = data.get("items", [])
    raw_text = " ".join(
        f"{item.get('title', '')}. {item.get('snippet', '')}" for item in items
    )

    return {
        "raw_text": raw_text,
        "source_url": search_url,
        "items": [
            {"title": i.get("title"), "snippet": i.get("snippet"), "link": i.get("link")}
            for i in items
        ],
    }
