import cv2
import os
from pathlib import Path
import numpy as np


def enhance_image(image):
    """Повышение резкости и контраста изображения."""
    # Увеличение резкости через фильтр unsharp mask
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    # Улучшение контраста (CLAHE) применяем к каналу яркости в LAB
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


# Путь к видео
VIDEO_PATH = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"
OUTPUT_DIR = "extracted_frames_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
saved_count = 0
frame_step = 5  # каждый 20-й кадр

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_step == 0:
        # Улучшаем кадр
        enhanced = enhance_image(frame)

        # Сохраняем с высоким качеством JPEG
        out_path = os.path.join(OUTPUT_DIR, f"frame_{saved_count:06d}.jpg")
        cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"Saved enhanced frame {saved_count} at {out_path}")

    frame_id += 1

cap.release()
print(f"Всего сохранено улучшенных кадров: {saved_count}")