import cv2
import mediapipe as mp
import os
import numpy as np
from pathlib import Path

# ------------------------------
# НАСТРОЙКИ
# ------------------------------
INPUT_DIR = r"D:\python_work\pythonProject\Bank\extracted_frames_enhanced"   # папка с кадрами
OUTPUT_DIR = r"D:\python_work\pythonProject\Bank\extracted_frames_hands" # папка для сохранения
PADDING_FACTOR = 0.3          # отступ вокруг кисти (как при обучении)
TARGET_SIZE = (224, 224)      # размер выходного изображения

# ------------------------------
# ПОДГОТОВКА
# ------------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def get_hand_bbox(hand_landmarks, img_shape, padding_factor=0.3):
    """Возвращает bounding box всей кисти с отступом."""
    h, w, _ = img_shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * padding_factor
    pad_y = box_h * padding_factor
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))
    return (x1, y1, x2, y2)

# ------------------------------
# ОБРАБОТКА
# ------------------------------
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
print(f"Найдено изображений: {len(image_files)}")

total_hands = 0
for img_file in image_files:
    img_path = os.path.join(INPUT_DIR, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Не удалось прочитать {img_path}, пропускаем")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            bbox = get_hand_bbox(hand_landmarks, image.shape, padding_factor=PADDING_FACTOR)
            x1, y1, x2, y2 = bbox
            hand_roi = image[y1:y2, x1:x2]
            if hand_roi.size == 0:
                continue

            # Ресайз до целевого размера
            resized = cv2.resize(hand_roi, TARGET_SIZE)

            # Формируем имя: исходный_файл_рукаN.jpg
            base, ext = os.path.splitext(img_file)
            out_filename = f"{base}_hand{idx}{ext}"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_hands += 1
            print(f"Сохранено: {out_path}")
    else:
        print(f"Руки не найдены: {img_file}")

print(f"\nГотово! Всего сохранено обрезков рук: {total_hands}")