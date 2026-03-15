import cv2
import mediapipe as mp
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------
# PATHS
# ------------------------------------------------

VIDEO_PATH = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"
ZONES_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\zones.json"
CSV_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\video_01_timeline.xlsx"


# ------------------------------------------------
# Функция для отображения русского текста
# ------------------------------------------------

def put_russian_text(img, text, position, font_size, color):
    """Отображение русского текста с использованием PIL"""
    # Конвертируем OpenCV image (BGR) в PIL image (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Создаем объект для рисования
    draw = ImageDraw.Draw(pil_img)

    # Пытаемся загрузить шрифт, поддерживающий кириллицу
    try:
        # Пробуем разные пути к шрифтам
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",  # Arial
            "C:/Windows/Fonts/times.ttf",  # Times New Roman
            "C:/Windows/Fonts/calibri.ttf",  # Calibri
            "C:/Windows/Fonts/DejaVuSans.ttf"  # DejaVu (если есть)
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            # Если шрифт не найден, используем стандартный
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Рисуем текст
    draw.text(position, text, font=font, fill=color[::-1])  # BGR -> RGB

    # Конвертируем обратно в OpenCV image
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


# ------------------------------------------------
# LOAD ZONES
# ------------------------------------------------

with open(ZONES_PATH, "r") as f:
    ZONES = json.load(f)

print("Zones loaded:", list(ZONES.keys()))

# ------------------------------------------------
# LOAD TIMELINE
# ------------------------------------------------

timeline = []

try:
    # Пробуем прочитать как Excel
    df = pd.read_excel(CSV_PATH)
    print(f"Loaded as Excel: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Конвертируем в список словарей
    for _, row in df.iterrows():
        event = {
            "frame_start": int(row["frame_start"]),
            "frame_end": int(row["frame_end"]),
            "operation": str(row.get("operation", "")),
            "suboperation": str(row.get("suboperation", "")),
            "action": str(row.get("action", "")),
            "object": str(row.get("object", "")),
            "hands": str(row.get("hand", ""))
        }
        timeline.append(event)

    print("Timeline events loaded:", len(timeline))

except Exception as e:
    print("Error loading file:", e)


# ------------------------------------------------
# FUNCTION: GET CURRENT ACTIONS
# ------------------------------------------------

def get_current_actions(frame_id):
    active = []
    for event in timeline:
        if event["frame_start"] <= frame_id <= event["frame_end"]:
            parts = []
            if event["operation"] and event["operation"] != 'nan':
                parts.append(event["operation"])
            if event["suboperation"] and event["suboperation"] != 'nan':
                parts.append(event["suboperation"])
            if event["action"] and event["action"] != 'nan':
                action_text = event["action"]
                if event["object"] and event["object"] != 'nan':
                    action_text += f" {event['object']}"
                parts.append(action_text)
            if event["hands"] and event["hands"] != 'nan':
                parts.append(f"({event['hands']})")

            if parts:
                txt = " | ".join(parts)
                active.append(txt)
    return active


# ------------------------------------------------
# ACTION RULES
# ------------------------------------------------

ACTION_RULES = {
    ("tray", "counter"): "take_money",
    ("counter", "printer"): "print_receipt",
    ("counter", "safe"): "store_cash",
    ("safe", "counter"): "take_from_safe",
    ("printer", "counter"): "take_receipt"
}

# ------------------------------------------------
# MEDIAPIPE INIT
# ------------------------------------------------

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Улучшенные настройки для лучшего детектирования
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,      # Повысили до 0.7
    min_tracking_confidence=0.7,        # Повысили до 0.7
    model_complexity=1,                  # 0, 1 или 2 (2 - самая точная, но медленная)
    smooth_landmarks=True,                # Сглаживание landmarks
    enable_segmentation=False,            # Нам не нужна сегментация
    smooth_segmentation=False,
    refine_face_landmarks=False           # Нам не нужны лицевые landmarks
)

# Более толстые и яркие линии для скелета
pose_drawing_spec = mp_draw.DrawingSpec(
    color=(0, 255, 0),
    thickness=3,                         # Увеличили толщину
    circle_radius=4                       # Увеличили радиус точек
)

hand_drawing_spec = mp_draw.DrawingSpec(
    color=(0, 255, 255),
    thickness=3,
    circle_radius=3
)

# Добавим счетчики для диагностики
detection_stats = {
    'pose_detected': 0,
    'right_hand_detected': 0,
    'left_hand_detected': 0,
    'frames_processed': 0
}

# ------------------------------------------------
# VIDEO
# ------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video not found")
    exit()

# ------------------------------------------------
# UI FLAGS
# ------------------------------------------------

paused = False
show_pose = True
show_hands = True
show_info = True

# ------------------------------------------------
# TRACKING VARIABLES
# ------------------------------------------------

right_history = []
left_history = []

MAX_HISTORY = 30

prev_right_zone = None
prev_left_zone = None

frame_id = 0


# ------------------------------------------------
# ZONE DETECTION
# ------------------------------------------------

def detect_zone(x, y, zones):
    for name, z in zones.items():
        if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
            return name
    return None


# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not paused:
        # Улучшение качества изображения перед обработкой
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Опционально: небольшое увеличение резкости
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # rgb = cv2.filter2D(rgb, -1, kernel)

        try:
            results = holistic.process(rgb)
            h, w, _ = frame.shape

            # Статистика детекции
            detection_stats['frames_processed'] += 1

            # ---------------- POSE ----------------
            if results.pose_landmarks:
                detection_stats['pose_detected'] += 1
                if show_pose:
                    # Рисуем скелет с улучшенными настройками
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        pose_drawing_spec,
                        pose_drawing_spec
                    )

                    # Опционально: добавляем ключевые точки для диагностики
                    # Рисуем основные суставы крупными точками
                    important_landmarks = [11, 12, 13, 14, 15, 16, 23, 24]  # плечи, локти, запястья, бедра
                    for idx in important_landmarks:
                        landmark = results.pose_landmarks.landmark[idx]
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

            # ---------------- RIGHT HAND ----------------
            if results.right_hand_landmarks:
                detection_stats['right_hand_detected'] += 1
                if show_hands:
                    # Рисуем скелет руки
                    mp_draw.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        hand_drawing_spec,
                        hand_drawing_spec
                    )

                    wrist = results.right_hand_landmarks.landmark[0]
                    rx = int(wrist.x * w)
                    ry = int(wrist.y * h)

                    # Проверяем, что координаты в пределах кадра
                    if 0 <= rx < w and 0 <= ry < h:
                        right_zone = detect_zone(rx, ry, ZONES)

                        if prev_right_zone and right_zone and prev_right_zone != right_zone:
                            key = (prev_right_zone, right_zone)
                            if key in ACTION_RULES:
                                action = ACTION_RULES[key]
                                print(f"RIGHT ACTION: {action} at frame {frame_id}")

                        prev_right_zone = right_zone
                        right_history.append((rx, ry))
                        if len(right_history) > MAX_HISTORY:
                            right_history.pop(0)

                        for i in range(1, len(right_history)):
                            cv2.line(frame, right_history[i - 1], right_history[i], (255, 0, 0), 2)

                        cv2.putText(frame, f"RIGHT {right_zone}", (rx + 10, ry),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ---------------- LEFT HAND ----------------
            if results.left_hand_landmarks:
                detection_stats['left_hand_detected'] += 1
                if show_hands:
                    mp_draw.draw_landmarks(
                        frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        hand_drawing_spec,
                        hand_drawing_spec
                    )

                    wrist = results.left_hand_landmarks.landmark[0]
                    lx = int(wrist.x * w)
                    ly = int(wrist.y * h)

                    if 0 <= lx < w and 0 <= ly < h:
                        left_zone = detect_zone(lx, ly, ZONES)

                        if prev_left_zone and left_zone and prev_left_zone != left_zone:
                            key = (prev_left_zone, left_zone)
                            if key in ACTION_RULES:
                                action = ACTION_RULES[key]
                                print(f"LEFT ACTION: {action} at frame {frame_id}")

                        prev_left_zone = left_zone
                        left_history.append((lx, ly))
                        if len(left_history) > MAX_HISTORY:
                            left_history.pop(0)

                        for i in range(1, len(left_history)):
                            cv2.line(frame, left_history[i - 1], left_history[i], (0, 0, 255), 2)

                        cv2.putText(frame, f"LEFT {left_zone}", (lx + 10, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Выводим статистику детекции каждые 100 кадров
            if frame_id % 100 == 0 and frame_id > 0:
                pose_rate = (detection_stats['pose_detected'] / detection_stats['frames_processed']) * 100
                right_rate = (detection_stats['right_hand_detected'] / detection_stats['frames_processed']) * 100
                left_rate = (detection_stats['left_hand_detected'] / detection_stats['frames_processed']) * 100
                print(f"Stats - Pose: {pose_rate:.1f}%, Right hand: {right_rate:.1f}%, Left hand: {left_rate:.1f}%")

        except Exception as e:
            print(f"Holistic error at frame {frame_id}: {e}")
            import traceback

            traceback.print_exc()

    # ---------------- DRAW ZONES ----------------
    for name, z in ZONES.items():
        cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0, 255, 255), 2)
        cv2.putText(frame, name, (z["x1"], z["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---------------- DRAW FRAME BOX (TOP-LEFT) ----------------
    # Увеличим размер рамки для длинного текста
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (700, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Рамка вокруг области текста
    cv2.rectangle(frame, (10, 10), (700, 250), (255, 255, 255), 2)

    # ---------------- INFO INSIDE BOX ----------------
    if show_info:
        # Информация о кадре (английский текст через OpenCV)
        cv2.putText(frame, f"Frame: {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Действия из timeline (русский текст через PIL)
        actions = get_current_actions(frame_id)

        if not actions:
            frame = put_russian_text(frame, "Нет действий", (20, 70), 20, (100, 100, 100))

        y = 70
        for a in actions:
            # Обрезаем слишком длинный текст
            if len(a) > 60:
                a = a[:57] + "..."
            frame = put_russian_text(frame, a, (20, y), 18, (0, 255, 0))
            y += 25

    cv2.imshow("Timeline Visualization", frame)

    # ---------------- CONTROLS ----------------
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        paused = not paused
        print("PAUSE:", paused)
    elif key == ord('p'):
        show_pose = not show_pose
    elif key == ord('h'):
        show_hands = not show_hands
    elif key == ord('i'):
        show_info = not show_info

    frame_id += 1

cap.release()
cv2.destroyAllWindows()