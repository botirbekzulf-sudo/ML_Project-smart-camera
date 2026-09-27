"""
gemini_assistant.py
Gemini 3.6 Flash используется ТОЛЬКО для сканера камеры (мультимодально):
что видно на кадре. Поиск по названию больше НЕ использует Gemini —
характеристики теперь ищутся локально, через device_lookup.py, без
внешних API и без лимитов квоты.
"""

from __future__ import annotations
import base64
import os
import requests

import gemini_prompts

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


def describe_device_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """Отправляет кадр с камеры в Gemini 3.6 и возвращает {"answer": str} или {"error": str}."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY не задан на сервере — см. README."}

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": gemini_prompts.DEVICE_PHOTO_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3},
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=payload, timeout=15)
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
