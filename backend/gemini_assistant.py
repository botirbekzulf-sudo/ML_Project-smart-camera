"""
gemini_assistant.py
Gemini 3.6 Flash используется как ФИЛЬТР поискового поля: перед тем как
сайт откроет Google и запустит ML-предсказание ценового класса, Gemini
проверяет, что введённый текст похож на название электронного устройства
(смартфон, ноутбук, ПК, планшет, умные часы и т.п.), а не случайный текст.

Сами тексты промптов вынесены в gemini_prompts.py — правь их там, не здесь.

Настройка:
    1. Получи API-ключ: https://aistudio.google.com/apikey
    2. Установи переменную окружения GEMINI_API_KEY
    (опционально GEMINI_MODEL, по умолчанию "gemini-3.6-flash")

РЕЖИМ БЕЗ КЛЮЧА ("fail-open"): если GEMINI_API_KEY не задан или Gemini
недоступен, проверка пропускается и поиск выполняется как обычно —
сайт не должен ломаться из-за отсутствия ключа. checked=False в ответе
показывает, что проверка не выполнялась.
"""

from __future__ import annotations
import base64
import os
import re
import requests

import gemini_prompts

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

VALID_CATEGORIES = {"phone", "laptop", "pc", "tablet", "watch", "other_electronics"}


def describe_device_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Отправляет кадр с камеры в Gemini 3.6 (мультимодально) и возвращает
    короткое текстовое описание устройства — этот текст выводится ПРЯМО
    на сайте, в реальном времени, на каждый кадр сканера.

    Возвращает {"answer": str} или {"error": str}.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY не задан на сервере — см. README."}

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": gemini_prompts.DEVICE_PHOTO_PROMPT},
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3},
    }

    try:
        response = requests.post(
            GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=payload, timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return {"error": "Gemini не вернул ответ (возможно, сработал фильтр безопасности)."}
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return {"answer": text or "Не удалось получить ответ."}
    except requests.RequestException as exc:
        return {"error": f"Ошибка запроса к Gemini: {exc}"}


def validate_device_query(query: str) -> dict:
    """
    Возвращает:
      {"is_device": True,  "category": "phone", "checked": True}
      {"is_device": False, "reason": "...",      "checked": True}
      {"is_device": True,  "checked": False}   -- Gemini недоступен, пропускаем проверку
    """
    query = (query or "").strip()
    if not query:
        return {"is_device": False, "reason": "Пустой запрос.", "checked": True}

    if not GEMINI_API_KEY:
        return {"is_device": True, "checked": False}

    prompt = gemini_prompts.DEVICE_VALIDATION_PROMPT.format(query=query)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 30, "temperature": 0},
    }

    try:
        response = requests.post(
            GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=payload, timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return {"is_device": True, "checked": False}

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return _parse_validation_response(text)

    except requests.RequestException:
        # Сеть/квота подвели — не блокируем пользователя из-за этого.
        return {"is_device": True, "checked": False}


def _parse_validation_response(text: str) -> dict:
    """Разбирает ответ вида 'ДА|phone' или 'НЕТ|это не устройство'."""
    verdict, _, rest = text.partition("|")
    verdict = verdict.strip().upper()
    rest = rest.strip()

    if verdict.startswith("ДА") or verdict.startswith("YES"):
        category = rest.lower() if rest.lower() in VALID_CATEGORIES else "other_electronics"
        return {"is_device": True, "category": category, "checked": True}

    if verdict.startswith("НЕТ") or verdict.startswith("NO"):
        return {"is_device": False, "reason": rest or "Не похоже на устройство.", "checked": True}

    # Модель ответила не в ожидаемом формате — не блокируем пользователя.
    return {"is_device": True, "checked": False, "raw_response": text}


def estimate_device_specs(model_name: str) -> dict:
    """
    Просит Gemini оценить типичные характеристики устройства по названию,
    используя его знания (а не поиск в реальном времени). Это заменяет
    платный Google Custom Search API как источник различающихся
    характеристик для ML-модели — без него все запросы получали бы
    одинаковый набор значений "по умолчанию" и, соответственно, всегда
    один и тот же ценовой класс.

    Возвращает {"ram_gb": float, "battery_mah": float, "storage_gb": float,
    "camera_mp": float, "checked": True} при успехе, или
    {"checked": False} если Gemini недоступен/квота исчерпана — тогда
    вызывающий код должен тихо откатиться на значения по умолчанию.
    """
    model_name = (model_name or "").strip()
    if not model_name or not GEMINI_API_KEY:
        return {"checked": False}

    prompt = gemini_prompts.DEVICE_SPECS_ESTIMATE_PROMPT.format(query=model_name)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 40, "temperature": 0.2},
    }

    try:
        response = requests.post(
            GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=payload, timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return {"checked": False}
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return _parse_specs_estimate(text)
    except requests.RequestException:
        # Сеть/квота/лимит подвели — молча откатываемся на значения по
        # умолчанию, не ломая остальной сайт.
        return {"checked": False}


def _parse_specs_estimate(text: str) -> dict:
    """Разбирает 'RAM_GB=8;BATTERY_MAH=4500;STORAGE_GB=128;CAMERA_MP=108'."""
    result: dict = {"checked": False}
    patterns = {
        "ram_gb": r"RAM_GB\s*=\s*([\d.]+)",
        "battery_mah": r"BATTERY_MAH\s*=\s*([\d.]+)",
        "storage_gb": r"STORAGE_GB\s*=\s*([\d.]+)",
        "camera_mp": r"CAMERA_MP\s*=\s*([\d.]+)",
    }
    found_any = False
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))
            found_any = True
    result["checked"] = found_any
    return result
