from ultralytics import YOLO

# Загружаем предобученную модель (например, yolov8n — самая лёгкая)
model = YOLO("yolov8n.pt")
t_bank_path = "model_creations/tbank.yaml"

# Запуск обучения
model.train(
    data=t_bank_path,  # путь к датасету
    epochs=50,          # количество эпох (подбирается)
    imgsz=640,          # размер входного изображения
    batch=16            # размер батча
)

metrics = model.val(data=t_bank_path, iou=0.5)
print(metrics)
