# main.py
# Точка входа в программу. Инициализирует всё и запускает цикл обработки видео.
import cv2
import numpy as np
from collections import deque
import config
from utils import load_zones, extract_frame_features, load_timeline_for_monitor, put_russian_text, get_hand_bbox, detect_zone
from models import load_currency_model, load_lstm_model, AsyncRecognizer
from action_monitor import ActionMonitor
from workflow_checker import WorkflowMonitor
from gui import draw_buttons, mouse_callback, handle_keyboard, set_button_positions
import gui  # для доступа к глобальным флагам GUI

# -------- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ --------
currency_model, class_names, empty_class = load_currency_model()
lstm_model, label_encoder = load_lstm_model()

# -------- ЗАГРУЗКА ЗОН --------
zones = load_zones(config.ZONES_PATH)
zone_names = list(zones.keys())

# -------- СОЗДАНИЕ РАСПОЗНОВАТЕЛЯ ВАЛЮТ --------
recognizer = AsyncRecognizer(
    model=currency_model,
    class_names=class_names,
    empty_class=empty_class,
    confidence_thresh=config.CONFIDENCE_THRESHOLD,
    history_size=config.HAND_HISTORY_SIZE
)

# =============================================================================
# МАППИНГ ДЕЙСТВИЙ ИЗ LSTM В НАЗВАНИЯ WORKFLOW
# =============================================================================
# Если ваши LSTM предсказывают, например, "Waiting / No action", а в workflow используется "IDLE",
# добавьте соответствия. При необходимости расширьте.
ACTION_MAPPING = {
    "Waiting / No action": "IDLE",
    "Greeting": "GREET",
    "Receiving money": "RECEIVE_CASH",
    "Counting manual": "COUNT_CASH_MANUAL",
    "Counting machine": "COUNT_CASH_MACHINE",
    "Checking banknote": "CHECK_BANKNOTE",
    "Processing": "PROCESSING",
    "Dispensing": "DISPENSING",
    "Complete": "COMPLETE",
    "Farewell": "FAREWELL"
    # Добавьте другие по необходимости
}

def map_action_to_workflow(lstm_action):
    """Преобразует действие из LSTM в идентификатор, понятный workflow."""
    return ACTION_MAPPING.get(lstm_action, lstm_action)

def get_current_currency_from_hands(recognizer):
    """Возвращает валюту из правой или левой руки (не empty)."""
    for hand in ['right', 'left']:
        state = recognizer.get_state(hand)
        if state['last_currency'] and state['last_currency'].lower() != 'empty':
            return state['last_currency']
    return None

# -------- ИНИЦИАЛИЗАЦИЯ WORKFLOW MONITOR --------
workflow_monitor = WorkflowMonitor(config.WORKFLOW_CONFIG_PATH)

# -------- ПОДГОТОВКА ВИДЕО --------
cap = cv2.VideoCapture(config.VIDEO_PATH)
if not cap.isOpened():
    print("Video not found")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, total frames: {total_frames}")

# -------- ЗАГРУЗКА ТАЙМЛАЙНА И СОЗДАНИЕ МОНИТОРА АНОМАЛИЙ --------
timeline_df = load_timeline_for_monitor(config.CSV_PATH, fps)
monitor = ActionMonitor(timeline_df, fps)

# -------- MEDIAPIPE HOLISTIC --------
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_draw
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
)

# -------- БУФЕРЫ ДЛЯ LSTM И ДЕТЕКЦИИ ДВИЖЕНИЯ --------
feature_buffer = deque(maxlen=int(fps * 5))      # храним признаки до 5 секунд
motion_buffer = deque(maxlen=5)                   # для сглаживания motion
last_motion_frames = 0
current_event_frames = 0
event_active = False
no_hands_frames = 0
last_classification_frame = 0
current_predicted_action = None
current_confidence = 0.0

# -------- GUI ПЕРЕМЕННЫЕ --------
gui.paused = False
gui.show_pose = config.SHOW_POSE_DEFAULT
gui.show_hands = config.SHOW_HANDS_DEFAULT
gui.show_info = config.SHOW_INFO_DEFAULT
gui.buttons = {}
gui.current_mouse_pos = (-1, -1)

# -------- ТРЕКИНГ РУК --------
right_history = []
left_history = []
MAX_HISTORY = 30
prev_right_zone = None
prev_left_zone = None

frame_id = 0

# переменная для хранения последнего сообщения о нарушении workflow
workflow_violation_message = ""

# -------- НАСТРОЙКА ОКНА И МЫШИ --------
cv2.namedWindow("Timeline Visualization")
mouse_param = {
    'paused': gui.paused,
    'frame_id': frame_id,
    'cap': cap,
    'total_frames': total_frames,
    'right_history': right_history,
    'left_history': left_history,
    'prev_right_zone': prev_right_zone,
    'prev_left_zone': prev_left_zone,
    'recognizer': recognizer,
    'monitor': monitor,
    'feature_buffer': feature_buffer,
    'current_event_frames': current_event_frames,
    'last_motion_frames': last_motion_frames,
    'event_active': event_active,
    'current_predicted_action': current_predicted_action,
    'current_confidence': current_confidence
}
cv2.setMouseCallback("Timeline Visualization", mouse_callback, mouse_param)

# -------- ОСНОВНОЙ ЦИКЛ --------
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        if not gui.paused:
            gui.paused = True
        if 'frame' in locals() and frame is not None:
            h, w = frame.shape[:2]
        else:
            h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "End of video. Use buttons to rewind.", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    else:
        if not gui.paused:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = holistic.process(rgb)
                h, w, _ = frame.shape

                # 1. Извлечение признаков и добавление в буфер LSTM
                feat = extract_frame_features(results, frame.shape, zones, zone_names)
                feature_buffer.append(feat)
                current_event_frames += 1

                # 2. Детекция движения (сумма изменений координат 84 точек рук)
                motion = 0.0
                if len(feature_buffer) >= 2:
                    prev_feat = feature_buffer[-2]
                    motion = np.sum(np.abs(feat[:84] - prev_feat[:84]))
                    motion_buffer.append(motion)
                    motion_smoothed = np.median(motion_buffer)
                    if motion_smoothed > config.MOTION_THRESHOLD:
                        last_motion_frames = 0
                        if not event_active:
                            event_active = True
                            print(f"ACTIVATE event at frame {frame_id}, motion={motion:.4f}")
                    else:
                        last_motion_frames += 1
                else:
                    last_motion_frames = 0

                # 3. Классификация LSTM при завершении движения
                if event_active and last_motion_frames > config.INFERENCE_DELAY_SEC * fps and current_event_frames >= config.MIN_SEQ_LEN:
                    # Подготовка последовательности: максимум 551 кадр, паддинг нулями слева
                    seq = np.array(list(feature_buffer))
                    max_len = 551
                    if seq.shape[0] > max_len:
                        seq = seq[-max_len:]
                    else:
                        pad = np.zeros((max_len - seq.shape[0], 100))
                        seq = np.vstack([pad, seq])
                    seq = seq.reshape(1, max_len, 100)
                    pred_probs = lstm_model.predict(seq, verbose=0)[0]
                    pred_class_idx = np.argmax(pred_probs)
                    pred_action = label_encoder.classes_[pred_class_idx]
                    confidence = pred_probs[pred_class_idx]
                    print(f"Frame {frame_id}: Predicted {pred_action} ({confidence:.3f})")
                    anomaly = monitor.update(pred_action, confidence, frame_id)
                    current_predicted_action = pred_action
                    current_confidence = confidence

                    # ========== ПЕРЕДАЧА ДЕЙСТВИЯ В WORKFLOW ==========
                    wf_action = map_action_to_workflow(pred_action)
                    currency = get_current_currency_from_hands(recognizer)
                    violation = workflow_monitor.on_action(wf_action, currency=currency)
                    if violation:
                        workflow_violation_message = violation
                    else:
                        workflow_violation_message = ""
                    # ========================================================

                    # Сброс состояния события
                    feature_buffer.clear()
                    current_event_frames = 0
                    last_motion_frames = 0
                    event_active = False
                    last_classification_frame = frame_id

                # Принудительная классификация каждые 5 секунд (для отладки)
                if frame_id - last_classification_frame > fps * 5 and current_event_frames >= config.MIN_SEQ_LEN:
                    print(f"FORCED classification at frame {frame_id}")
                    seq = np.array(list(feature_buffer))
                    max_len = 551
                    if seq.shape[0] > max_len:
                        seq = seq[-max_len:]
                    else:
                        pad = np.zeros((max_len - seq.shape[0], 100))
                        seq = np.vstack([pad, seq])
                    seq = seq.reshape(1, max_len, 100)
                    pred_probs = lstm_model.predict(seq, verbose=0)[0]
                    pred_class_idx = np.argmax(pred_probs)
                    pred_action = label_encoder.classes_[pred_class_idx]
                    confidence = pred_probs[pred_class_idx]
                    print(f"FORCED Frame {frame_id}: Predicted {pred_action} ({confidence:.3f})")
                    anomaly = monitor.update(pred_action, confidence, frame_id)
                    current_predicted_action = pred_action
                    current_confidence = confidence

                    # ========== ПЕРЕДАЧА ДЕЙСТВИЯ В WORKFLOW (принудительная) ==========
                    wf_action = map_action_to_workflow(pred_action)
                    currency = get_current_currency_from_hands(recognizer)
                    violation = workflow_monitor.on_action(wf_action, currency=currency)
                    if violation:
                        workflow_violation_message = violation
                    else:
                        workflow_violation_message = ""

                    last_classification_frame = frame_id
                    # НЕ сбрасываем event_active и буфер

                # Обработка отсутствия рук (для отладки)
                if not event_active and (results.right_hand_landmarks is None and results.left_hand_landmarks is None):
                    no_hands_frames += 1
                    if no_hands_frames > fps * 2:
                        if current_predicted_action != "Waiting / No action":
                            print(f"NO HANDS for 2 sec at frame {frame_id}, setting Waiting")
                            current_predicted_action = "Waiting / No action"
                            current_confidence = 1.0
                else:
                    no_hands_frames = 0

                # ----- ОТРИСОВКА СКЕЛЕТА (опционально) -----
                if gui.show_pose and results.pose_landmarks:
                    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                           mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                           mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

                # ----- ПРАВАЯ РУКА -----
                if results.right_hand_landmarks:
                    if gui.show_hands:
                        bbox = get_hand_bbox(results.right_hand_landmarks, frame.shape)
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    wrist = results.right_hand_landmarks.landmark[0]
                    rx = int(wrist.x * w)
                    ry = int(wrist.y * h)
                    if 0 <= rx < w and 0 <= ry < h:
                        right_zone = detect_zone(rx, ry, zones)
                        # Здесь можно добавить логику ACTION_RULES (не используется в текущей версии)
                        prev_right_zone = right_zone
                        if gui.show_hands:
                            right_history.append((rx, ry))
                            if len(right_history) > MAX_HISTORY:
                                right_history.pop(0)
                            for i in range(1, len(right_history)):
                                cv2.line(frame, right_history[i-1], right_history[i], (255,0,0), 2)
                            cv2.putText(frame, f"RIGHT {right_zone}", (rx+10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    # Распознавание валют
                    if recognizer.should_recognize('right', frame_id):
                        bbox = get_hand_bbox(results.right_hand_landmarks, frame.shape)
                        x1, y1, x2, y2 = bbox
                        hand_roi = frame[y1:y2, x1:x2]
                        if hand_roi.size > 0:
                            recognizer.request_recognition('right', hand_roi, frame_id)
                    else:
                        recognizer.update_frame_counter('right')
                    state = recognizer.get_state('right')
                    if state['last_currency'] is not None:
                        if state['last_currency'].lower() == 'empty':
                            cv2.putText(frame, "R: empty", (rx+10, ry+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        else:
                            cv2.putText(frame, f"R: {state['last_currency']} ({state['last_confidence']:.2f})",
                                        (rx+10, ry+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # ----- ЛЕВАЯ РУКА (аналогично) -----
                if results.left_hand_landmarks:
                    if gui.show_hands:
                        bbox = get_hand_bbox(results.left_hand_landmarks, frame.shape)
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    wrist = results.left_hand_landmarks.landmark[0]
                    lx = int(wrist.x * w)
                    ly = int(wrist.y * h)
                    if 0 <= lx < w and 0 <= ly < h:
                        left_zone = detect_zone(lx, ly, zones)
                        prev_left_zone = left_zone
                        if gui.show_hands:
                            left_history.append((lx, ly))
                            if len(left_history) > MAX_HISTORY:
                                left_history.pop(0)
                            for i in range(1, len(left_history)):
                                cv2.line(frame, left_history[i-1], left_history[i], (0,0,255), 2)
                            cv2.putText(frame, f"LEFT {left_zone}", (lx+10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    if recognizer.should_recognize('left', frame_id):
                        bbox = get_hand_bbox(results.left_hand_landmarks, frame.shape)
                        x1, y1, x2, y2 = bbox
                        hand_roi = frame[y1:y2, x1:x2]
                        if hand_roi.size > 0:
                            recognizer.request_recognition('left', hand_roi, frame_id)
                    else:
                        recognizer.update_frame_counter('left')
                    state = recognizer.get_state('left')
                    if state['last_currency'] is not None:
                        if state['last_currency'].lower() == 'empty':
                            cv2.putText(frame, "L: empty", (lx+10, ly+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        else:
                            cv2.putText(frame, f"L: {state['last_currency']} ({state['last_confidence']:.2f})",
                                        (lx+10, ly+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            except Exception as e:
                print(f"Holistic error at frame {frame_id}: {e}")

    # ----- ОТРИСОВКА ЗОН -----
    for name, z in zones.items():
        if "points" in z:
            pts = np.array(z["points"], np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0,255,255), thickness=2)
            cx = int(sum(p[0] for p in z["points"]) / len(z["points"]))
            cy = int(sum(p[1] for p in z["points"]) / len(z["points"]))
            cv2.putText(frame, name, (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0,255,255), 2)
            cv2.putText(frame, name, (z["x1"], z["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # ----- ИНФОРМАЦИОННАЯ ПАНЕЛЬ (полупрозрачный фон) -----
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (700,500), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10,10), (700,500), (255,255,255), 2)
    if gui.show_info:
        cv2.putText(frame, f"Frame: {frame_id} / {total_frames}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Controls: SPACE=Pause/Play, LEFT/RIGHT=Seek +/-30, R=Reset tracks", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if current_predicted_action:
            text = f"Predicted: {current_predicted_action} ({current_confidence:.2f})"
            frame = put_russian_text(frame, text, (20,100), 20, (0,255,0))
            if monitor.anomaly_message:
                frame = put_russian_text(frame, f"⚠ {monitor.anomaly_message}", (20,130), 18, (0,0,255))

                # Отображение сообщения от workflow
                if workflow_violation_message:
                    frame = put_russian_text(frame, workflow_violation_message, (20, 160), 20, (0, 0, 255))
            else:
                frame = put_russian_text(frame, "Нет предсказания", (20, 100), 20, (150, 150, 150))

    # ----- РИСУЕМ КНОПКИ -----
    draw_buttons(frame, gui.current_mouse_pos[0], gui.current_mouse_pos[1])
    cv2.imshow("Timeline Visualization", frame)

    # ----- ОБРАБОТКА КЛАВИАТУРЫ -----
    key = cv2.waitKey(1) & 0xFF
    state_dict = {
        'running': running,
        'paused': gui.paused,
        'show_pose': gui.show_pose,
        'show_hands': gui.show_hands,
        'show_info': gui.show_info,
        'right_history': right_history,
        'left_history': left_history,
        'prev_right_zone': prev_right_zone,
        'prev_left_zone': prev_left_zone,
        'recognizer': recognizer,
        'monitor': monitor,
        'feature_buffer': feature_buffer,
        'current_event_frames': current_event_frames,
        'last_motion_frames': last_motion_frames,
        'event_active': event_active,
        'current_predicted_action': current_predicted_action,
        'current_confidence': current_confidence,
        'frame_id': frame_id,
        'cap': cap,
        'total_frames': total_frames
    }
    state_dict = handle_keyboard(key, state_dict)
    running = state_dict['running']
    gui.paused = state_dict['paused']
    gui.show_pose = state_dict['show_pose']
    gui.show_hands = state_dict['show_hands']
    gui.show_info = state_dict['show_info']
    right_history = state_dict['right_history']
    left_history = state_dict['left_history']
    prev_right_zone = state_dict['prev_right_zone']
    prev_left_zone = state_dict['prev_left_zone']
    recognizer = state_dict['recognizer']
    monitor = state_dict['monitor']
    feature_buffer = state_dict['feature_buffer']
    current_event_frames = state_dict['current_event_frames']
    last_motion_frames = state_dict['last_motion_frames']
    event_active = state_dict['event_active']
    current_predicted_action = state_dict['current_predicted_action']
    current_confidence = state_dict['current_confidence']
    frame_id = state_dict['frame_id']

    # ----- УВЕЛИЧЕНИЕ СЧЁТЧИКА КАДРОВ, ЕСЛИ НЕ ПАУЗА И КАДР БЫЛ -----
    if not gui.paused and ret:
        frame_id += 1

# ----- ЗАВЕРШЕНИЕ -----
cap.release()
cv2.destroyAllWindows()