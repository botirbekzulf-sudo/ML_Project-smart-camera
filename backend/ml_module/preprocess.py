"""
preprocess.py
Приводит "сырые" характеристики устройства (полученные из Google/парсинга
или введённые вручную) к тому набору признаков, на котором обучена модель.

Схема признаков соответствует классическому датасету
Kaggle "Mobile Price Classification" (iabhishekofficial):
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

FEATURE_ORDER — порядок признаков ОБЯЗАН совпадать с тем, что видела модель
при обучении (train_price_model.py использует этот же список).
"""

from __future__ import annotations
import re
from typing import Any

FEATURE_ORDER = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]

# Значения "по умолчанию", если что-то не удалось извлечь из текста
# характеристик — модель всё равно должна суметь сделать предсказание.
#
# ВАЖНО: эти значения должны попадать в диапазон признаков реального
# датасета Kaggle "Mobile Price Classification", на котором обучена
# модель (train_price_model.py) — иначе модель получает нереалистичную,
# "выходящую за пределы обучающих данных" комбинацию и почти всегда
# скатывается в один и тот же класс, независимо от того, что реально
# ввёл пользователь. Диапазоны в этом датасете:
#   battery_power 501-1998, clock_speed 0.5-3.0, fc 0-19, int_memory 2-64,
#   m_dep 0.1-1.0, mobile_wt 80-200, n_cores 1-8, pc 0-20,
#   px_height 0-1960, px_width 500-1998, ram 256-3998, sc_h 5-19,
#   sc_w 0-18, talk_time 2-20.
# Ниже — средние/типичные значения ИЗ ЭТИХ диапазонов, а не из старого
# синтетического тестового датасета.
DEFAULTS: dict[str, Any] = {
    "battery_power": 1238, "blue": 1, "clock_speed": 1.5, "dual_sim": 1,
    "fc": 5, "four_g": 1, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 140,
    "n_cores": 4, "pc": 10, "px_height": 800, "px_width": 1200,
    "ram": 2000, "sc_h": 12, "sc_w": 8, "talk_time": 10, "three_g": 1,
    "touch_screen": 1, "wifi": 1,
}

# Настройка для UI-шкал характеристик (заполненность в % от практического
# максимума на рынке смартфонов). Порядок здесь = порядок отображения баров.
FEATURE_DISPLAY: dict[str, dict] = {
    "ram":            {"label": "Оперативная память", "unit": "МБ", "max": 12288},
    "battery_power":  {"label": "Батарея",             "unit": "мА·ч", "max": 6000},
    "int_memory":     {"label": "Встроенная память",   "unit": "ГБ", "max": 512},
    "pc":             {"label": "Основная камера",     "unit": "МП", "max": 108},
    "clock_speed":    {"label": "Частота процессора",  "unit": "ГГц", "max": 4.0},
    "n_cores":        {"label": "Ядра процессора",     "unit": "", "max": 8},
    "px_width":       {"label": "Разрешение экрана",   "unit": "px (ширина)", "max": 2400},
    "talk_time":      {"label": "Время работы",        "unit": "ч", "max": 24},
}


def features_to_display(features: dict) -> list[dict]:
    """
    Превращает признаки модели в готовый для UI список шкал:
    [{"key", "label", "unit", "value", "percent"}, ...]
    percent — заполненность шкалы (0-100), округлённая, ограничена сверху 100.
    """
    rows = []
    for key, cfg in FEATURE_DISPLAY.items():
        value = features.get(key, DEFAULTS.get(key, 0))
        percent = 0 if cfg["max"] <= 0 else round(min(value / cfg["max"], 1.0) * 100)
        rows.append({
            "key": key,
            "label": cfg["label"],
            "unit": cfg["unit"],
            "value": value,
            "percent": percent,
        })
    return rows


def _extract_number(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def specs_text_to_features(raw_specs: str) -> dict:
    """
    Пытается вытащить числовые характеристики из свободного текста,
    который вернул поиск (например, сниппет Google / карточка товара).

    Это эвристический разбор "best effort": он не обязан быть точным —
    его задача дать модели разумные признаки, когда структурированных
    данных нет. Для продакшена лучше брать данные из API/парсера с
    чёткой структурой, а не из произвольного текста.
    """
    text = raw_specs or ""
    features = dict(DEFAULTS)

    ram_gb = _extract_number(text, r"(\d+(?:[.,]\d+)?)\s*(?:gb|гб)\s*ram")
    if ram_gb:
        features["ram"] = int(ram_gb * 1024)  # в датасете ram в МБ

    battery_mah = _extract_number(text, r"(\d{3,5})\s*mah")
    if battery_mah:
        features["battery_power"] = int(battery_mah)

    storage_gb = _extract_number(text, r"(\d+(?:[.,]\d+)?)\s*(?:gb|гб)\s*(?:storage|rom|памят)")
    if storage_gb:
        features["int_memory"] = int(storage_gb)

    camera_mp = _extract_number(text, r"(\d+(?:[.,]\d+)?)\s*(?:mp|мп)")
    if camera_mp:
        features["pc"] = int(camera_mp)

    ghz = _extract_number(text, r"(\d+(?:[.,]\d+)?)\s*ghz")
    if ghz:
        features["clock_speed"] = ghz

    weight_g = _extract_number(text, r"(\d{2,4})\s*(?:g|г)\b")
    if weight_g:
        features["mobile_wt"] = int(weight_g)

    features["four_g"] = 1 if re.search(r"\b4g\b|lte", text, re.IGNORECASE) else features["four_g"]
    features["three_g"] = 1 if re.search(r"\b3g\b", text, re.IGNORECASE) else features["three_g"]
    features["wifi"] = 1 if re.search(r"wi-?fi", text, re.IGNORECASE) else features["wifi"]
    features["blue"] = 1 if re.search(r"bluetooth", text, re.IGNORECASE) else features["blue"]
    features["dual_sim"] = 1 if re.search(r"dual sim|две sim", text, re.IGNORECASE) else features["dual_sim"]
    features["touch_screen"] = 1 if re.search(r"touch|сенсор", text, re.IGNORECASE) else features["touch_screen"]

    return features


def form_to_features(form: dict) -> dict:
    """
    Приводит данные, введённые пользователем вручную на сайте
    (форма с понятными полями типа "RAM, GB", "Batteries, mAh"),
    к признакам модели. Более надёжный путь, чем разбор текста.
    """
    features = dict(DEFAULTS)
    mapping_gb_to_mb = {"ram"}
    for key, value in form.items():
        if key not in FEATURE_ORDER or value in (None, ""):
            continue
        value = float(value)
        if key in mapping_gb_to_mb and value < 64:  # ввели в GB, а не MB
            value = value * 1024
        features[key] = value
    return features


def to_ordered_vector(features: dict) -> list[float]:
    """Возвращает список значений строго в порядке FEATURE_ORDER."""
    return [float(features.get(name, DEFAULTS[name])) for name in FEATURE_ORDER]
