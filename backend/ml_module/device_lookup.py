"""
device_lookup.py
Сопоставляет введённое пользователем название устройства с локальной
базой характеристик (data/device_specs_lookup.csv) — БЕЗ обращения к
каким-либо внешним API. Никаких лимитов, задержек или ключей — просто
поиск по названию в CSV-файле.

Это и есть источник разных характеристик для разных устройств: раньше
эту роль играл Gemini (с лимитами бесплатной квоты) или пустой ответ
Google Custom Search — теперь это простая локальная база, которую
можно свободно расширять, просто дописывая строки в CSV.

Как добавить больше устройств: открой data/device_specs_lookup.csv в
Excel/Google Таблицах и допиши строки в формате
name,ram_gb,battery_mah,storage_gb,camera_mp — файл читается заново
при каждом перезапуске сервера.
"""

from __future__ import annotations
import csv
import difflib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "..", "data", "device_specs_lookup.csv")

_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is None:
        rows = []
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        _cache = rows
    return _cache


def find_specs(query: str) -> dict | None:
    """
    Ищет наиболее подходящее устройство по названию. Возвращает
    {"ram_gb", "battery_mah", "storage_gb", "camera_mp", "matched_name"}
    или None, если ничего похожего не нашлось.
    """
    query_norm = (query or "").strip().lower()
    if not query_norm:
        return None

    rows = _load()
    if not rows:
        return None

    names_norm = [row["name"].lower() for row in rows]

    # 1) точное совпадение — высший приоритет
    if query_norm in names_norm:
        return _row_to_specs(rows[names_norm.index(query_norm)])

    # 2) совпадение по подстроке в любую сторону — предпочитаем САМОЕ
    #    ДЛИННОЕ (более специфичное) совпадение, чтобы "S24 Ultra" не
    #    ошибочно схлопывалось до просто "S24".
    best_substring_match = None
    for name_norm, row in zip(names_norm, rows):
        if query_norm in name_norm or name_norm in query_norm:
            if best_substring_match is None or len(name_norm) > len(best_substring_match[0]):
                best_substring_match = (name_norm, row)
    if best_substring_match:
        return _row_to_specs(best_substring_match[1])

    # 2) нечёткое совпадение (опечатки, немного другой порядок слов)
    close = difflib.get_close_matches(query_norm, names_norm, n=1, cutoff=0.6)
    if close:
        idx = names_norm.index(close[0])
        return _row_to_specs(rows[idx])

    return None


def _row_to_specs(row: dict) -> dict:
    return {
        "ram_gb": float(row["ram_gb"]),
        "battery_mah": float(row["battery_mah"]),
        "storage_gb": float(row["storage_gb"]),
        "camera_mp": float(row["camera_mp"]),
        "matched_name": row["name"],
    }
