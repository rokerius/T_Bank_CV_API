import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def annotate_images_with_yolo(model_path, images_dir, output_dir, conf_threshold=0.25):
    """
    Размечает изображения с помощью YOLO модели, сохраняя только новые аннотации
    
    Args:
        model_path (str): Путь к файлу модели YOLO (.pt)
        images_dir (str): Директория с изображениями
        output_dir (str): Директория для сохранения аннотаций
        conf_threshold (float): Порог уверенности для детекции
    """
    
    print(f"Загрузка модели из {model_path}")
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for file in os.listdir(images_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    processed_count = 0
    skipped_count = 0
    
    for image_file in image_files:
        image_name = Path(image_file).stem
        image_path = os.path.join(images_dir, image_file)
        output_path = os.path.join(output_dir, f"{image_name}.txt")
        
        if os.path.exists(output_path):
            print(f"Аннотация для {image_file} уже существует, пропускаем")
            skipped_count += 1
            continue
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {image_file}")
                continue
            
            results = model(img, conf=conf_threshold, verbose=False)
            
            boxes = results[0].boxes
            if boxes is not None:
                yolo_annotations = []
                for box in boxes:
                    x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    yolo_annotations.append(annotation_line)
                
                if yolo_annotations:
                    with open(output_path, 'w') as f:
                        f.writelines(yolo_annotations)
                    print(f"Сохранена аннотация для {image_file} ({len(yolo_annotations)} объектов)")
                else:
                    print(f"Объекты не обнаружены на {image_file}")
                    with open(output_path, 'w') as f:
                        pass
            
            processed_count += 1
            
        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")
    
    print(f"\nОбработка завершена!")
    print(f"Обработано: {processed_count}")
    print(f"Пропущено (уже существуют): {skipped_count}")


if __name__ == "__main__":
    MODEL_PATH = "C:/Users/roker/VSCode_projects/T_Bank_CV_API/T_Bank_CV_API/runs/detect/train4/weights/best.pt"
    IMAGES_DIR = "C:/Users/roker/VSCode_projects/T_Bank_CV_API/T_Bank_CV_API/data/raw/images"
    OUTPUT_DIR = "C:/Users/roker/VSCode_projects/T_Bank_CV_API/T_Bank_CV_API/data/raw/labels"
    
    annotate_images_with_yolo(
        model_path=MODEL_PATH,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25
    )