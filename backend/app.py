"""
app.py
Flask backend для сайта "Определение цены товара техники и ценовой класс
смартфона".

Запуск:
    pip install -r requirements.txt
    cd backend
    python app.py
Сайт будет на http://localhost:5000

Роуты:
    GET  /                      -> отдаёт frontend/index.html
    GET  /scan.html             -> отдаёт frontend/scan.html
    POST /api/gemini-vision     -> кадр с камеры -> текстовый ответ Gemini о том, что на фото
    POST /api/lookup            -> название модели -> локальный поиск характеристик + ссылка на Google
    POST /api/predict-price     -> характеристики -> ценовой класс (ML)
"""

from __future__ import annotations
import os
import sys
import joblib
from flask import Flask, request, jsonify, send_from_directory

sys.path.append(os.path.join(os.path.dirname(__file__), "ml_module"))

from preprocess import (  # noqa: E402
    specs_text_to_features, form_to_features, to_ordered_vector, features_to_display,
)
from train_price_model import PRICE_CLASS_NAMES  # noqa: E402
import device_lookup  # noqa: E402
import google_search  # noqa: E402
import gemini_assistant  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "ml_module", "price_model.pkl")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Модель не обучена. Запусти: python ml_module/train_price_model.py"
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/scan.html")
def scan_page():
    return send_from_directory(FRONTEND_DIR, "scan.html")


@app.route("/api/gemini-vision", methods=["POST"])
def gemini_vision():
    """Кадр с камеры -> Gemini 3.6 (мультимодально) -> что на фото."""
    if "image" not in request.files:
        return jsonify({"error": "Файл 'image' не найден в запросе"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    result = gemini_assistant.describe_device_photo(image_bytes, mime_type=file.mimetype or "image/jpeg")
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)


@app.route("/api/lookup", methods=["POST"])
def lookup():
    """
    Принимает {"model_name": "iPhone 13"}.

    Характеристики ищутся ЛОКАЛЬНО в data/device_specs_lookup.csv по
    названию (device_lookup.py) — без внешних API и без лимитов квоты.
    Это и есть источник разных характеристик для разных устройств: без
    него все запросы получали бы одни и те же значения "по умолчанию" и
    один и тот же ценовой класс, независимо от того, что ввёл пользователь.

    Отдельно сайт также открывает обычный поиск Google по названию
    (google_search.py) — просто ссылка, как и описано в задании.
    """
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    if not model_name:
        return jsonify({"error": "Не указано название модели (model_name)"}), 400

    matched = device_lookup.find_specs(model_name)

    result = google_search.search_specs(model_name)
    result["specs_estimate"] = {
        "checked": matched is not None,
        "ram_gb": matched["ram_gb"] if matched else None,
        "battery_mah": matched["battery_mah"] if matched else None,
        "storage_gb": matched["storage_gb"] if matched else None,
        "camera_mp": matched["camera_mp"] if matched else None,
        "matched_name": matched["matched_name"] if matched else None,
    }
    return jsonify(result)


@app.route("/api/predict-price", methods=["POST"])
def predict_price():
    """
    Принимает ОДИН из вариантов:
      {"raw_text": "..."}                              -- сырой текст характеристик
      {"form": {"ram": 8, "battery_power": 4500, ...}}  -- ручной ввод / данные из lookup
    Возвращает ценовой класс.
    """
    data = request.get_json(silent=True) or {}

    if "form" in data and data["form"]:
        features = form_to_features(data["form"])
    else:
        features = specs_text_to_features(data.get("raw_text", ""))

    vector = [to_ordered_vector(features)]

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    model, scaler = bundle["model"], bundle["scaler"]
    vector_scaled = scaler.transform(vector)
    pred_class = int(model.predict(vector_scaled)[0])
    proba = model.predict_proba(vector_scaled)[0].tolist()

    return jsonify({
        "price_class_id": pred_class,
        "price_class_name": PRICE_CLASS_NAMES[pred_class],
        "probabilities": {
            PRICE_CLASS_NAMES[i]: round(p, 3) for i, p in enumerate(proba)
        },
        "features_used": features,
        "features_display": features_to_display(features),
    })


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
