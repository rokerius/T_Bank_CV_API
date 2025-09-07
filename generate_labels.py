import cv2
import numpy as np
import os

# --------------------------
# Настройки
# --------------------------
IMAGES_DIR = "images"
TEMPLATE_PATH = "template.png"
OUTPUT_DIR = "labels"  # сюда будут сохраняться .txt файлы

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Функция поиска логотипа
# --------------------------
def find_logo(image_path, template_path, threshold=0.8):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    boxes = []
    for pt in zip(*loc[::-1]):
        x1, y1 = pt
        x2, y2 = x1 + w, y1 + h
        boxes.append([x1, y1, x2, y2])
    return boxes, img.shape[1], img.shape[0]  # width, height

# --------------------------
# Преобразование координат в YOLOv8
# --------------------------
def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

# --------------------------
# Основной цикл по изображениям
# --------------------------
for filename in os.listdir(IMAGES_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, filename)
    boxes, img_w, img_h = find_logo(image_path, TEMPLATE_PATH)

    if not boxes:
        continue  # если логотип не найден, пропускаем

    # создаем .txt файл с тем же именем
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)

    with open(txt_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center, y_center, width, height = convert_to_yolo(x1, y1, x2, y2, img_w, img_h)
            class_id = 0  # T-Bank
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"Разметка готова и сохранена в папке '{OUTPUT_DIR}'")
