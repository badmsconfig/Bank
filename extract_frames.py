import cv2
import os
from pathlib import Path

video_path = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"  # укажите путь к вашему видео
output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Сохраняем каждый 10-й кадр
    if frame_id % 20 == 0:
        out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_count += 1
        print(f"Saved {out_path}")
    frame_id += 1

cap.release()
print(f"Всего сохранено {saved_count} кадров")