import cv2
import numpy as np
import os
import glob

# Путь к тренировочной папке
base_path = r"D:\python_work\pythonProject\Bank\data_money\train"
target_folders = ["empty","BYN","USD"]  # обрабатываем только эти папки

# Допустимые расширения изображений
extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

def adjust_brightness(image, delta):
    """Изменяет яркость изображения на delta (может быть отрицательным)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + delta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def add_gaussian_noise(image, sigma=30):
    """Добавляет гауссов шум с заданным стандартным отклонением."""
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

for folder in target_folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена, пропускаем.")
        continue

    # Ищем все изображения в папке
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder_path, ext)))

    if not images:
        print(f"В папке {folder_path} нет изображений.")
        continue

    for img_path in images:
        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось прочитать {img_path}")
            continue

        # Получаем имя файла без расширения
        dir_name = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        # 1. Яркость +30
        bright_plus = adjust_brightness(img, 30)
        out_plus = os.path.join(dir_name, f"{base_name}_bright+30{ext}")
        cv2.imwrite(out_plus, bright_plus, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Сохранено: {out_plus}")

        # 2. Яркость -30
        bright_minus = adjust_brightness(img, -30)
        out_minus = os.path.join(dir_name, f"{base_name}_bright-30{ext}")
        cv2.imwrite(out_minus, bright_minus, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Сохранено: {out_minus}")

        # 3. Шум +30 (sigma=30)
        noisy = add_gaussian_noise(img, sigma=30)
        out_noise = os.path.join(dir_name, f"{base_name}_noise30{ext}")
        cv2.imwrite(out_noise, noisy, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Сохранено: {out_noise}")

print("Аугментация завершена.")