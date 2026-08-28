"""
train_price_model.py
Обучает классификатор ценового класса смартфона по характеристикам.

Датасет: Kaggle "Mobile Price Classification" (iabhishekofficial)
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
(зеркало на HuggingFace: FeatEng/Data, kheejay88/phone_price_classification_train)

Как использовать с РЕАЛЬНЫМ датасетом:
1. Скачай train.csv с Kaggle (нужен только столбец price_range как таргет).
2. Положи файл в data/mobile_price_train.csv
3. Запусти: python train_price_model.py
   (без файла скрипт автоматически возьмёт data/sample_mobile_prices.csv —
    сгенерированный тестовый датасет той же структуры, просто чтобы
    пайплайн можно было проверить сразу).

Результат: backend/ml_module/price_model.pkl — обученная модель,
готовая к использованию в app.py.
"""

from __future__ import annotations
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from preprocess import FEATURE_ORDER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data")
REAL_DATA = os.path.join(DATA_DIR, "mobile_price_train.csv")
SAMPLE_DATA = os.path.join(DATA_DIR, "sample_mobile_prices.csv")
MODEL_OUT = os.path.join(HERE, "price_model.pkl")

PRICE_CLASS_NAMES = {
    0: "Бюджетный",
    1: "Средний",
    2: "Премиум",
    3: "Флагман",
}


def load_dataset() -> pd.DataFrame:
    if os.path.exists(REAL_DATA):
        print(f"[data] Использую реальный датасет: {REAL_DATA}")
        df = pd.read_csv(REAL_DATA)
    elif os.path.exists(SAMPLE_DATA):
        print(f"[data] ВНИМАНИЕ: реальный датасет не найден, использую "
              f"тестовый сгенерированный: {SAMPLE_DATA}")
        print("[data] Скачай настоящий с Kaggle и положи в "
              "data/mobile_price_train.csv для реального обучения.")
        df = pd.read_csv(SAMPLE_DATA)
    else:
        raise FileNotFoundError(
            "Не найден ни data/mobile_price_train.csv, ни "
            "data/sample_mobile_prices.csv. Запусти gen_sample.py "
            "или положи датасет с Kaggle."
        )

    missing = [c for c in FEATURE_ORDER + ["price_range"] if c not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing}")

    return df


def train():
    df = load_dataset()
    X = df[FEATURE_ORDER]
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"[cv] Accuracy на кросс-валидации: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n[test] Отчёт по классификации:")
    print(classification_report(
        y_test, y_pred,
        target_names=[PRICE_CLASS_NAMES[i] for i in sorted(PRICE_CLASS_NAMES)],
    ))
    print("[test] Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

    importances = sorted(
        zip(FEATURE_ORDER, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n[model] Важность признаков (топ-8):")
    for name, imp in importances[:8]:
        print(f"  {name:<15} {imp:.3f}")

    joblib.dump({"model": model, "scaler": scaler}, MODEL_OUT)
    print(f"\n[save] Модель сохранена: {MODEL_OUT}")


if __name__ == "__main__":
    train()
