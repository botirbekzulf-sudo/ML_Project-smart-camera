"""
gemini_assistant.py
Gemini 3.6 Flash используется в двух местах:

1) describe_device_photo() — сканер камеры (мультимодально): что на фото.
2) check_and_estimate_specs() — ОДИН запрос на каждый поиск по названию,
   который сразу и проверяет "это устройство?", и оценивает типичные
   характеристики (объединено специально, чтобы тратить вдвое меньше
   квоты, чем два отдельных запроса).

Сами тексты промптов вынесены в gemini_prompts.py — правь их там, не здесь.

Настройка:
    1. Получи API-ключ: https://aistudio.google.com/apikey
    2. Установи переменную окружения GEMINI_API_KEY
    (опционально GEMINI_MODEL, по умолчанию "gemini-3.6-flash")

РЕЖИМ БЕЗ КЛЮЧА / ПРИ ИСЧЕРПАННОЙ КВОТЕ ("fail-open"): если
GEMINI_API_KEY не задан, или Gemini вернул ошибку (в т.ч. 429 Too Many
Requests при исчерпанной бесплатной квоте), проверка/оценка тихо
пропускается — checked=False — и остальной сайт продолжает работать на
значениях по умолчанию, вместо того чтобы ломаться.
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


def check_and_estimate_specs(query: str) -> dict:
    """
    ОДИН запрос к Gemini вместо двух: проверяет, что запрос похож на
    название электронного устройства, И если да — сразу оценивает его
    типичные характеристики (RAM/батарея/память/камера). Экономит
    квоту API вдвое по сравнению с раздельными вызовами.

    Возвращает:
      {"is_device": True,  "category": "phone", "checked": True,
       "ram_gb": 8.0, "battery_mah": 4500.0, "storage_gb": 128.0, "camera_mp": 108.0}
      {"is_device": False, "reason": "...", "checked": True}
      {"is_device": True,  "checked": False}   -- Gemini недоступен/квота исчерпана
    """
    query = (query or "").strip()
    if not query:
        return {"is_device": False, "reason": "Пустой запрос.", "checked": True}

    if not GEMINI_API_KEY:
        return {"is_device": True, "checked": False}

    prompt = gemini_prompts.DEVICE_CHECK_AND_SPECS_PROMPT.format(query=query)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 60, "temperature": 0},
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
        return _parse_check_and_specs_response(text)

    except requests.RequestException as exc:
        # Сеть/квота/лимит (включая 429) подвели — не блокируем
        # пользователя, просто помечаем как непроверено.
        print(f"[gemini_assistant] check_and_estimate_specs недоступен: {exc}")
        return {"is_device": True, "checked": False}


def _parse_check_and_specs_response(text: str) -> dict:
    """
    Разбирает либо 'НЕТ|причина', либо
    'ДА|phone|RAM_GB=8;BATTERY_MAH=4500;STORAGE_GB=128;CAMERA_MP=108'.
    """
    verdict, _, rest = text.partition("|")
    verdict = verdict.strip().upper()
    rest = rest.strip()

    if verdict.startswith("НЕТ") or verdict.startswith("NO"):
        return {"is_device": False, "reason": rest or "Не похоже на устройство.", "checked": True}

    if verdict.startswith("ДА") or verdict.startswith("YES"):
        category_part, _, specs_part = rest.partition("|")
        category = category_part.strip().lower()
        if category not in VALID_CATEGORIES:
            category = "other_electronics"

        result = {"is_device": True, "category": category, "checked": True}
        result.update(_parse_specs_fields(specs_part))
        return result

    # Модель ответила не в ожидаемом формате — не блокируем пользователя,
    # но и характеристик у нас нет.
    return {"is_device": True, "checked": False, "raw_response": text}


def _parse_specs_fields(text: str) -> dict:
    """Разбирает 'RAM_GB=8;BATTERY_MAH=4500;STORAGE_GB=128;CAMERA_MP=108'."""
    fields = {}
    patterns = {
        "ram_gb": r"RAM_GB\s*=\s*([\d.]+)",
        "battery_mah": r"BATTERY_MAH\s*=\s*([\d.]+)",
        "storage_gb": r"STORAGE_GB\s*=\s*([\d.]+)",
        "camera_mp": r"CAMERA_MP\s*=\s*([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields[key] = float(match.group(1))
    return fields
