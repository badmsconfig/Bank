import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
from shapely.geometry import Point, Polygon
import os

# Правильные импорты mediapipe для версии 0.10.x
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_draw

# ------------------------------------------------
# PATHS
# ------------------------------------------------
VIDEO_PATH = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"
ZONES_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\zones.json"
CSV_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\video_01_timeline.xlsx"
MODEL_PATH = r"D:\python_work\pythonProject\Bank\src\trained_models\custom_classifier.h5"
CLASSES_TRAIN_PATH = r"D:\python_work\pythonProject\Bank\data_money\train"

# ------------------------------------------------
# Загрузка модели распознавания валют
# ------------------------------------------------
print("Загрузка модели...")
currency_model = load_model(MODEL_PATH)
print("Модель загружена.")

# Определяем классы (названия валют)
if os.path.exists(CLASSES_TRAIN_PATH):
    class_names = sorted([d for d in os.listdir(CLASSES_TRAIN_PATH)
                          if os.path.isdir(os.path.join(CLASSES_TRAIN_PATH, d))])
    print("Классы валют из папки train:", class_names)
else:
    class_names = ['BYN', 'EUR', 'RUB', 'USD']
    print("Внимание: используем классы по умолчанию:", class_names)

# ------------------------------------------------
# Функция для отображения русского текста
# ------------------------------------------------
def put_russian_text(img, text, position, font_size, color):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
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
    draw.text(position, text, font=font, fill=color[::-1])
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

# ------------------------------------------------
# LOAD ZONES
# ------------------------------------------------
with open(ZONES_PATH, "r") as f:
    ZONES = json.load(f)
print("Zones loaded:", list(ZONES.keys()))

# ------------------------------------------------
# LOAD TIMELINE (исправленная версия)
# ------------------------------------------------
timeline = []
try:
    df = pd.read_excel(CSV_PATH)
    print(f"Loaded as Excel: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Приводим frame_start и frame_end к числам
    df['frame_start'] = pd.to_numeric(df['frame_start'], errors='coerce')
    df['frame_end'] = pd.to_numeric(df['frame_end'], errors='coerce')
    df = df.dropna(subset=['frame_start', 'frame_end'])
    df['frame_start'] = df['frame_start'].astype(int)
    df['frame_end'] = df['frame_end'].astype(int)

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
# ACTION RULES (опционально)
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
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
)

hand_drawing_spec = mp_draw.DrawingSpec(
    color=(0, 255, 255),
    thickness=3,
    circle_radius=3
)

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
show_pose = False          # скелет выключен по умолчанию
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
def point_in_polygon(x, y, poly):
    """Проверка, находится ли точка (x, y) внутри полигона poly (список вершин [[x1,y1], [x2,y2], ...])."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

from shapely.geometry import Point, Polygon

def detect_zone(x, y, zones, tolerance=40):
    """
    Определяет, находится ли точка (x,y) внутри зоны (с допуском tolerance пикселей).
    tolerance – на сколько пикселей можно выйти за границу, чтобы всё равно считаться внутри.
    """
    point = Point(x, y)
    for name, z in zones.items():
        if "points" in z:
            poly = Polygon(z["points"])
            # Проверяем: внутри или на расстоянии <= tolerance
            if poly.contains(point) or poly.distance(point) <= tolerance:
                return name
        else:
            # старый прямоугольный формат (если есть)
            if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
                return name
    return None

def detect_zone(x, y, zones):
    for name, z in zones.items():
        if "points" in z:
            if point_in_polygon(x, y, z["points"]):
                return name
        else:
            # поддержка старого формата
            if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
                return name
    return None

# ------------------------------------------------
# FUNCTIONS FOR CURRENCY RECOGNITION
# ------------------------------------------------
def get_hand_bbox(hand_landmarks, img_shape, padding_factor=0.3):
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

def preprocess_hand_roi(roi, target_size=(224,224)):
    roi_resized = cv2.resize(roi, target_size)
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    return np.expand_dims(roi_normalized, axis=0)

CONFIDENCE_THRESHOLD = 0.5

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not paused:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = holistic.process(rgb)
            h, w, _ = frame.shape

            # ---------------- POSE (опционально) ----------------
            if show_pose and results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

            # ---------------- RIGHT HAND ----------------
            if results.right_hand_landmarks and show_hands:
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
                        cv2.line(frame, right_history[i-1], right_history[i], (255,0,0), 2)

                    cv2.putText(frame, f"RIGHT {right_zone}", (rx+10, ry),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                    # ---- Currency recognition ----
                    bbox = get_hand_bbox(results.right_hand_landmarks, frame.shape, padding_factor=0.3)
                    x1, y1, x2, y2 = bbox
                    hand_roi = frame[y1:y2, x1:x2]
                    if hand_roi.size > 0:
                        input_tensor = preprocess_hand_roi(hand_roi)
                        preds = currency_model.predict(input_tensor, verbose=0)[0]
                        class_idx = np.argmax(preds)
                        confidence = preds[class_idx]
                        # Отладка
                        print(
                            f"Right hand: class={class_idx} ({class_names[class_idx] if class_idx < len(class_names) else 'unknown'}), confidence={confidence:.4f}")
                        if confidence > CONFIDENCE_THRESHOLD:
                            currency = class_names[class_idx]
                            cv2.putText(frame, f"R: {currency} ({confidence:.2f})", (rx + 10, ry + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        print("Right hand ROI empty")

            # ---------------- LEFT HAND ----------------
            if results.left_hand_landmarks and show_hands:
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
                        cv2.line(frame, left_history[i-1], left_history[i], (0,0,255), 2)

                    cv2.putText(frame, f"LEFT {left_zone}", (lx+10, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    # ---- Currency recognition ----
                    bbox = get_hand_bbox(results.left_hand_landmarks, frame.shape, padding_factor=0.3)
                    x1, y1, x2, y2 = bbox
                    hand_roi = frame[y1:y2, x1:x2]
                    if hand_roi.size > 0:
                        input_tensor = preprocess_hand_roi(hand_roi)
                        preds = currency_model.predict(input_tensor, verbose=0)[0]
                        class_idx = np.argmax(preds)
                        confidence = preds[class_idx]
                        # Отладка
                        print(
                            f"Left hand: class={class_idx} ({class_names[class_idx] if class_idx < len(class_names) else 'unknown'}), confidence={confidence:.4f}")
                        if confidence > CONFIDENCE_THRESHOLD:
                            currency = class_names[class_idx]
                            cv2.putText(frame, f"L: {currency} ({confidence:.2f})", (lx + 10, ly + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        print("Left hand ROI empty")

        except Exception as e:
            print(f"Holistic error at frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()

    # ---------------- DRAW ZONES ----------------
    for name, z in ZONES.items():
        if "points" in z:  # рисуем полигон
            pts = np.array(z["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            # подпись в центре полигона
            cx = int(sum(p[0] for p in z["points"]) / len(z["points"]))
            cy = int(sum(p[1] for p in z["points"]) / len(z["points"]))
            cv2.putText(frame, name, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:  # старый прямоугольный формат
            cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0, 255, 255), 2)
            cv2.putText(frame, name, (z["x1"], z["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---------------- DRAW FRAME BOX (TOP-LEFT) ----------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (700,250), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10,10), (700,250), (255,255,255), 2)

    # ---------------- INFO INSIDE BOX ----------------
    if show_info:
        cv2.putText(frame, f"Frame: {frame_id}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        actions = get_current_actions(frame_id)
        if not actions:
            frame = put_russian_text(frame, "Нет действий", (20,70), 20, (100,100,100))
        y = 70
        for a in actions:
            if len(a) > 60:
                a = a[:57] + "..."
            frame = put_russian_text(frame, a, (20,y), 18, (0,255,0))
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
        print("POSE:", show_pose)
    elif key == ord('h'):
        show_hands = not show_hands
        print("HANDS:", show_hands)
    elif key == ord('i'):
        show_info = not show_info
        print("INFO:", show_info)

    frame_id += 1

cap.release()
cv2.destroyAllWindows()