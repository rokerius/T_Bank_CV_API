import io
import torch
import numpy as np
import cv2
from ultralytics import YOLO


# Загружаем модель один раз при старте приложения
# (best.pt должен быть в корне проекта или загружен по ссылке в README)
MODEL_PATH = "weights/morty_4000_model.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[WARNING] Не удалось загрузить модель: {e}")
    model = None


def load_image(image_bytes: bytes) -> np.ndarray:
    """Преобразуем байты в OpenCV-изображение"""
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def detect_logos(image_bytes: bytes):
    if model is None:
        return []

    image = load_image(image_bytes)
    results = model.predict(image, imgsz=640, conf=0.4, device=0 if torch.cuda.is_available() else "cpu")

    detections = []
    for r in results:
        for box in r.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            detections.append({
                "bbox": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                }
            })

    return detections
