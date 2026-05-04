# utils.py
# Вспомогательные функции: отрисовка текста, зоны, препроцессинг ROI, извлечение признаков
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
import config

# -----------------------------------------------------------------------------
def put_russian_text(img, text, position, font_size, color):
    """
    Наложение текста на изображение с поддержкой кириллицы.
    Использует PIL для рендеринга шрифта Windows.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font_paths = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/calibri.ttf"]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color[::-1])  # BGR->RGB
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

# -----------------------------------------------------------------------------
def load_zones(zones_path):
    """Загружает зоны из JSON файла и возвращает словарь."""
    with open(zones_path, "r") as f:
        zones = json.load(f)
    print("Zones loaded:", list(zones.keys()))
    return zones

# -----------------------------------------------------------------------------
def detect_zone(x, y, zones, tolerance=40):
    """
    Определяет, в какой зоне (полигон или прямоугольник) находится точка (x,y).
    tolerance - допустимое расстояние до полигона (для неточных попаданий).
    """
    point = Point(x, y)
    for name, z in zones.items():
        if "points" in z:
            poly = Polygon(z["points"])
            if poly.contains(point) or poly.distance(point) <= tolerance:
                return name
        else:
            if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
                return name
    return None

# -----------------------------------------------------------------------------
def get_hand_bbox(hand_landmarks, img_shape, padding_factor=config.HAND_BBOX_PADDING_FACTOR,
                  use_palm_only=config.USE_ONLY_HAND_PALM):
    """
    Возвращает ограничивающий прямоугольник кисти (x1,y1,x2,y2) в координатах кадра.
    Если use_palm_only=True, используем только точки ладони (индексы 5-20).
    """
    h, w, _ = img_shape
    if use_palm_only:
        indices = range(5, 21)  # точки ладони
        xs = [hand_landmarks.landmark[i].x * w for i in indices]
        ys = [hand_landmarks.landmark[i].y * h for i in indices]
    else:
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

# -----------------------------------------------------------------------------
def preprocess_hand_roi(roi, target_size=(224, 224)):
    """Подготавливает вырезанную область для подачи в CNN."""
    roi_resized = cv2.resize(roi, target_size)
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    return np.expand_dims(roi_normalized, axis=0)

# -----------------------------------------------------------------------------
def extract_frame_features(results, frame_shape, zones, zone_names):
    """
    Извлекает вектор признаков из 100 чисел для одного кадра:
    - 84 координаты (x,y) 21 точки правой и левой руки (42+42)
    - 8 one-hot зона правой руки
    - 8 one-hot зона левой руки
    """
    h, w, _ = frame_shape
    features = []
    # Правая рука (42 числа)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
        if len(features) < 42:
            features.extend([0.0] * (42 - len(features)))
    else:
        features.extend([0.0] * 42)
    # Левая рука (42 числа)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
        if len(features) < 84:
            features.extend([0.0] * (84 - len(features)))
    else:
        features.extend([0.0] * 42)
    # Зона правой руки (one-hot, 8 чисел)
    right_zone = None
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rx = int(wrist.x * w)
        ry = int(wrist.y * h)
        right_zone = detect_zone(rx, ry, zones)
    for zone in zone_names:
        features.append(1.0 if right_zone == zone else 0.0)
    # Зона левой руки (one-hot, 8 чисел)
    left_zone = None
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lx = int(wrist.x * w)
        ly = int(wrist.y * h)
        left_zone = detect_zone(lx, ly, zones)
    for zone in zone_names:
        features.append(1.0 if left_zone == zone else 0.0)
    return np.array(features, dtype=np.float32)

# -----------------------------------------------------------------------------
def load_timeline_for_monitor(csv_path, fps):
    """
    Загружает CSV/XLSX с таймлайном действий.
    Возвращает DataFrame с колонками: start_sec, end_sec, frame_start, frame_end, duration_sec, action_name.
    """
    def time_str_to_seconds(s):
        if not isinstance(s, str):
            s = str(s)
        parts = s.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            try:
                return float(s)
            except:
                return 0.0
    if csv_path.lower().endswith('.xlsx'):
        df = pd.read_excel(csv_path, engine='openpyxl', dtype=str)
    else:
        df = pd.read_csv(csv_path, encoding='utf-8')
    df['start'] = df['start'].astype(str)
    df['end'] = df['end'].astype(str)
    df['start_sec'] = df['start'].apply(time_str_to_seconds)
    df['end_sec'] = df['end'].apply(time_str_to_seconds)
    df['frame_start'] = (df['start_sec'] * fps).round().astype(int)
    df['frame_end'] = (df['end_sec'] * fps).round().astype(int)
    df['duration_sec'] = df['duration_sec'].astype(float)
    return df