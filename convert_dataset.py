"""
convert_dataset.py
Конвертирует скачанный с Kaggle датасет реальных смартфонов в формат,
который использует backend/ml_module/device_lookup.py:
    name,ram_gb,battery_mah,storage_gb,camera_mp

Как использовать:
    1. Скачай CSV с Kaggle (например:
       https://www.kaggle.com/datasets/shraddha4ever20/smartphone-specifications-dataset)
    2. Положи скачанный файл в data/raw_smartphone_dataset.csv
    3. Запусти: python convert_dataset.py
       (из корня проекта, где лежит папка data/)
    4. Скрипт перезапишет data/device_specs_lookup.csv

Если автоопределение столбцов не сработает (скрипт выведет ошибку и
список найденных столбцов) — пришли мне этот список, поправим сопоставление.
"""

import csv
import os
import re

RAW_PATH = os.path.join("data", "raw_smartphone_dataset.csv")
OUT_PATH = os.path.join("data", "device_specs_lookup.csv")

# Ключевые слова для автоопределения нужных столбцов (без учёта регистра).
# Проверяются по порядку — первое совпадение побеждает.
COLUMN_KEYWORDS = {
    "brand": ["brand", "manufacturer", "company"],
    "model": ["model", "name", "phone_name", "device"],
    "ram_gb": ["ram"],
    "battery_mah": ["battery"],
    "storage_gb": ["storage", "rom", "internal_memory"],
    "camera_mp": ["camera", "rear_camera", "main_camera", "primary_camera"],
}


def find_column(headers: list[str], keywords: list[str]) -> str | None:
    headers_lower = {h.lower(): h for h in headers}
    for keyword in keywords:
        for h_lower, h_original in headers_lower.items():
            if keyword in h_lower:
                return h_original
    return None


def extract_number(value: str) -> float | None:
    """Достаёт первое число из строки вида '8 GB' / '128GB' / '4500 mAh'."""
    if value is None:
        return None
    match = re.search(r"[\d.]+", str(value).replace(",", "."))
    return float(match.group()) if match else None


def main():
    if not os.path.exists(RAW_PATH):
        print(f"Не найден файл {RAW_PATH}. Скачай датасет с Kaggle и положи туда.")
        return

    with open(RAW_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    print(f"Найдено столбцов: {len(headers)}")
    print(f"Столбцы: {headers}")

    col_brand = find_column(headers, COLUMN_KEYWORDS["brand"])
    col_model = find_column(headers, COLUMN_KEYWORDS["model"])
    col_ram = find_column(headers, COLUMN_KEYWORDS["ram_gb"])
    col_battery = find_column(headers, COLUMN_KEYWORDS["battery_mah"])
    col_storage = find_column(headers, COLUMN_KEYWORDS["storage_gb"])
    col_camera = find_column(headers, COLUMN_KEYWORDS["camera_mp"])

    print(f"\nОпределены столбцы:")
    print(f"  brand    -> {col_brand}")
    print(f"  model    -> {col_model}")
    print(f"  ram_gb   -> {col_ram}")
    print(f"  battery  -> {col_battery}")
    print(f"  storage  -> {col_storage}")
    print(f"  camera   -> {col_camera}")

    missing = [name for name, col in [
        ("model/name", col_model), ("ram", col_ram),
        ("battery", col_battery), ("storage", col_storage), ("camera", col_camera),
    ] if col is None]
    if missing:
        print(f"\nНЕ УДАЛОСЬ найти столбцы: {missing}")
        print("Пришли список столбцов выше — поправим сопоставление вручную.")
        return

    out_rows = []
    seen_names = set()
    for row in rows:
        model = (row.get(col_model) or "").strip()
        brand = (row.get(col_brand) or "").strip() if col_brand else ""
        if not model:
            continue

        # Если модель не содержит имя бренда — добавляем ("Galaxy S24" -> "Samsung Galaxy S24")
        name = model if (not brand or brand.lower() in model.lower()) else f"{brand} {model}"

        ram_gb = extract_number(row.get(col_ram))
        battery_mah = extract_number(row.get(col_battery))
        storage_gb = extract_number(row.get(col_storage))
        camera_mp = extract_number(row.get(col_camera))

        if None in (ram_gb, battery_mah, storage_gb, camera_mp):
            continue  # пропускаем неполные строки

        if name in seen_names:
            continue
        seen_names.add(name)

        out_rows.append([name, ram_gb, battery_mah, storage_gb, camera_mp])

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "ram_gb", "battery_mah", "storage_gb", "camera_mp"])
        writer.writerows(out_rows)

    print(f"\nГотово! Записано {len(out_rows)} устройств в {OUT_PATH}")


if __name__ == "__main__":
    main()
