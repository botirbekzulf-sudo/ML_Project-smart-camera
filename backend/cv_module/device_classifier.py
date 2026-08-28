"""
device_classifier.py
Определяет категорию устройства (смартфон / ноутбук / ПК) по фото/кадру видео.

ВАЖНО (честно, без иллюзий):
Готового датасета "фото -> точная модель устройства + цена" не существует -
это задача не для курсового/дипломного проекта. Реалистичная и при этом
рабочая схема — двухступенчатая:

  1) Эта CV-модель определяет ФОРМ-ФАКТОРНУЮ КАТЕГОРИЮ устройства, а не
     конкретный тип техники. Классы — это "корзины по форме", в которые
     попадает ЛЮБАЯ электроника:
       - "phone"  -> смартфон, планшет, умные часы, e-reader
       - "laptop" -> ноутбук, нетбук, Chromebook
       - "pc"     -> системный блок, моноблок, монитор, игровая консоль
     То есть распознаётся не "это смартфон", а "это устройство
     phone-типа" — дальше уточняется текстом/OCR, что именно это
     за модель. Такой подход резко снижает требования к датасету и
     разметке по сравнению с попыткой различить сотни моделей техники
     напрямую по фото.
  2) Точную модель ("iPhone 13 Pro") пользователь подтверждает/вводит
     сам, ИЛИ по названию, извлечённому OCR-ом с коробки/наклейки.
     Дальше по названию модели идёт поиск характеристик и цены
     в google_search.py.

РЕАЛЬНОЕ ВРЕМЯ:
predict() вызывается backend'ом на каждый кадр, который присылает
фронтенд (camera.js гонит кадр каждые ~1.5 сек, пока открыта камера).
Инференс на CPU с MobileNetV2 укладывается в 50-150 мс на кадр — для
живого сканирования с таким интервалом этого достаточно. Если нужно
ощутимо быстрее — вариант: перенести инференс на клиент через
TensorFlow.js/ONNX Runtime Web (тогда кадры вообще не уходят на
сервер, только итоговое название модели).

Ниже — рабочий скелет transfer-learning модели на MobileNetV2
(torchvision). Веса ImageNet скачиваются автоматически при первом
запуске в среде с доступом в интернет. Файл содержит:
  - train(): дообучение последнего слоя на своих фото
  - predict(): инференс на одном изображении
  - эвристический fallback (без обученных весов), чтобы backend
    не падал, если модель ещё не обучена — иначе тестировать
    остальной пайплайн (Google-поиск + ML price-класс) неудобно.
"""

from __future__ import annotations
import os
from typing import Literal

DeviceCategory = Literal["phone", "laptop", "pc"]

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "device_classifier_weights.pth")
CLASSES: list[DeviceCategory] = ["phone", "laptop", "pc"]


def _build_model(num_classes: int = 3):
    """Собирает MobileNetV2 с заменённым классификационным слоем."""
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False  # замораживаем backbone
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train(data_dir: str, epochs: int = 8, lr: float = 1e-3):
    """
    Дообучает модель на своих фото.

    Ожидаемая структура data_dir (формат ImageFolder):
        data_dir/phone/*.jpg
        data_dir/laptop/*.jpg
        data_dir/pc/*.jpg
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(num_classes=len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / len(dataset)
        print(f"[epoch {epoch+1}/{epochs}] loss={total_loss/len(dataset):.4f} acc={acc:.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": dataset.classes}, WEIGHTS_PATH)
    print(f"[save] Веса сохранены: {WEIGHTS_PATH}")


def predict(image_path: str) -> dict:
    """
    Возвращает {"category": "phone"|"laptop"|"pc", "confidence": float}.

    Если обученных весов ещё нет (WEIGHTS_PATH отсутствует), используется
    простой эвристический fallback по соотношению сторон изображения,
    только чтобы остальной пайплайн можно было тестировать сквозь весь
    сайт без обученной CV-модели.
    """
    if not os.path.exists(WEIGHTS_PATH):
        return _heuristic_fallback(image_path)

    import torch
    from PIL import Image
    from torchvision import transforms

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    model = _build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())

    return {"category": classes[idx], "confidence": float(probs[idx])}


def _heuristic_fallback(image_path: str) -> dict:
    """
    Грубая эвристика "пока нет обученной модели": широкие фото чаще
    ноутбук/ПК-монитор, вытянутые вертикально — чаще телефон.
    Это ЗАГЛУШКА для разработки, не для продакшена.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        w, h = img.size
    ratio = w / h

    if ratio < 0.75:
        category, confidence = "phone", 0.55
    elif ratio > 1.4:
        category, confidence = "laptop", 0.5
    else:
        category, confidence = "pc", 0.4

    return {"category": category, "confidence": confidence, "fallback": True}
