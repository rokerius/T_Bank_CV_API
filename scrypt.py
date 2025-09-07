import os

# пути
img_dir = "data/splits/images/train"
label_dir = "data/splits/labels/train"

# создаём папку labels/train, если её ещё нет
os.makedirs(label_dir, exist_ok=True)

# перебираем картинки
for img_file in os.listdir(img_dir):
    name, ext = os.path.splitext(img_file)
    label_path = os.path.join(label_dir, f"{name}.txt")
    if not os.path.exists(label_path):
        open(label_path, "w").close()  # создаём пустой файл
        print(f"Создан пустой .txt для {img_file}")
