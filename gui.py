# gui.py
# GUI элементы: кнопки, обработка кликов, горячие клавиши
import cv2
import config

# глобальные переменные для состояния GUI (будут установлены в main)
buttons = {}
current_mouse_pos = (-1, -1)
paused = False
show_pose = config.SHOW_POSE_DEFAULT
show_hands = config.SHOW_HANDS_DEFAULT
show_info = config.SHOW_INFO_DEFAULT

# -----------------------------------------------------------------------------
def set_button_positions(frame_width):
    """Вычисляет координаты кнопок в правом верхнем углу."""
    global buttons
    start_x = frame_width - (config.BUTTON_WIDTH * 4 + config.BUTTON_SPACING * 3) - 20
    buttons = {
        "play_pause": (start_x, config.BUTTON_Y, start_x + config.BUTTON_WIDTH, config.BUTTON_Y + config.BUTTON_HEIGHT),
        "rewind": (start_x + config.BUTTON_WIDTH + config.BUTTON_SPACING, config.BUTTON_Y,
                   start_x + config.BUTTON_WIDTH * 2 + config.BUTTON_SPACING, config.BUTTON_Y + config.BUTTON_HEIGHT),
        "forward": (start_x + (config.BUTTON_WIDTH + config.BUTTON_SPACING) * 2, config.BUTTON_Y,
                    start_x + config.BUTTON_WIDTH * 3 + config.BUTTON_SPACING * 2, config.BUTTON_Y + config.BUTTON_HEIGHT),
        "reset": (start_x + (config.BUTTON_WIDTH + config.BUTTON_SPACING) * 3, config.BUTTON_Y,
                  start_x + config.BUTTON_WIDTH * 4 + config.BUTTON_SPACING * 3, config.BUTTON_Y + config.BUTTON_HEIGHT)
    }

# -----------------------------------------------------------------------------
def draw_buttons(frame, mouse_x, mouse_y):
    """Рисует кнопки на кадре, подсвечивая при наведении."""
    h, w = frame.shape[:2]
    if not buttons:
        set_button_positions(w)
    if w != (buttons["play_pause"][2] - buttons["play_pause"][0] + 20 + config.BUTTON_WIDTH * 4 + config.BUTTON_SPACING * 3):
        set_button_positions(w)
    button_texts = {
        "play_pause": "Play/Pause" if not paused else "Pause",
        "rewind": f"-{config.SEEK_STEP}",
        "forward": f"+{config.SEEK_STEP}",
        "reset": "Reset"
    }
    for name, (x1, y1, x2, y2) in buttons.items():
        if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
            color = (100, 100, 200)   # подсветка
        else:
            color = (50, 50, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        text = button_texts[name]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        text_x = x1 + (config.BUTTON_WIDTH - text_size[0]) // 2
        text_y = y1 + (config.BUTTON_HEIGHT + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)

# -----------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    """
    Callback для обработки кликов по кнопкам и обновления позиции мыши.
    param содержит словарь с изменяемыми переменными:
        - paused
        - frame_id
        - cap
        - total_frames
        - right_history, left_history
        - prev_right_zone, prev_left_zone
        - recognizer
        - monitor
        - feature_buffer, current_event_frames, last_motion_frames, event_active
        - current_predicted_action, current_confidence
    """
    global current_mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        for name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "play_pause":
                    param['paused'] = not param['paused']
                elif name == "rewind":
                    new_frame = max(0, param['frame_id'] - config.SEEK_STEP)
                    if new_frame != param['frame_id']:
                        param['frame_id'] = new_frame
                        param['cap'].set(cv2.CAP_PROP_POS_FRAMES, param['frame_id'])
                        param['right_history'].clear()
                        param['left_history'].clear()
                        param['prev_right_zone'] = None
                        param['prev_left_zone'] = None
                        param['recognizer'].reset()
                        param['monitor'].reset()
                        param['feature_buffer'].clear()
                        param['current_event_frames'] = 0
                        param['last_motion_frames'] = 0
                        param['event_active'] = False
                        param['current_predicted_action'] = None
                        param['current_confidence'] = 0.0
                elif name == "forward":
                    new_frame = min(param['total_frames'] - 1, param['frame_id'] + config.SEEK_STEP)
                    if new_frame != param['frame_id']:
                        param['frame_id'] = new_frame
                        param['cap'].set(cv2.CAP_PROP_POS_FRAMES, param['frame_id'])
                        param['right_history'].clear()
                        param['left_history'].clear()
                        param['prev_right_zone'] = None
                        param['prev_left_zone'] = None
                        param['recognizer'].reset()
                        param['monitor'].reset()
                        param['feature_buffer'].clear()
                        param['current_event_frames'] = 0
                        param['last_motion_frames'] = 0
                        param['event_active'] = False
                        param['current_predicted_action'] = None
                        param['current_confidence'] = 0.0
                elif name == "reset":
                    param['right_history'].clear()
                    param['left_history'].clear()
                    param['prev_right_zone'] = None
                    param['prev_left_zone'] = None
                    param['recognizer'].reset()
                    param['monitor'].reset()
                    param['feature_buffer'].clear()
                    param['current_event_frames'] = 0
                    param['last_motion_frames'] = 0
                    param['event_active'] = False
                    param['current_predicted_action'] = None
                    param['current_confidence'] = 0.0
                break

# -----------------------------------------------------------------------------
def handle_keyboard(key, state):
    """
    Обработка горячих клавиш. state - словарь с переменными состояния.
    Возвращает изменённый state.
    """
    if key == 27:  # ESC
        state['running'] = False
    elif key == 32:  # SPACE
        state['paused'] = not state['paused']
    elif key == ord('p'):
        state['show_pose'] = not state['show_pose']
    elif key == ord('h'):
        state['show_hands'] = not state['show_hands']
    elif key == ord('i'):
        state['show_info'] = not state['show_info']
    elif key == ord('r'):
        state['right_history'].clear()
        state['left_history'].clear()
        state['prev_right_zone'] = None
        state['prev_left_zone'] = None
        state['recognizer'].reset()
        state['monitor'].reset()
        state['feature_buffer'].clear()
        state['current_event_frames'] = 0
        state['last_motion_frames'] = 0
        state['event_active'] = False
        state['current_predicted_action'] = None
        state['current_confidence'] = 0.0
    elif key == 81 or key == ord('a'):  # left arrow or 'a'
        new_frame = max(0, state['frame_id'] - config.SEEK_STEP)
        if new_frame != state['frame_id']:
            state['frame_id'] = new_frame
            state['cap'].set(cv2.CAP_PROP_POS_FRAMES, state['frame_id'])
            state['right_history'].clear()
            state['left_history'].clear()
            state['prev_right_zone'] = None
            state['prev_left_zone'] = None
            state['recognizer'].reset()
            state['monitor'].reset()
            state['feature_buffer'].clear()
            state['current_event_frames'] = 0
            state['last_motion_frames'] = 0
            state['event_active'] = False
            state['current_predicted_action'] = None
            state['current_confidence'] = 0.0
    elif key == 83 or key == ord('d'):  # right arrow or 'd'
        new_frame = min(state['total_frames'] - 1, state['frame_id'] + config.SEEK_STEP)
        if new_frame != state['frame_id']:
            state['frame_id'] = new_frame
            state['cap'].set(cv2.CAP_PROP_POS_FRAMES, state['frame_id'])
            state['right_history'].clear()
            state['left_history'].clear()
            state['prev_right_zone'] = None
            state['prev_left_zone'] = None
            state['recognizer'].reset()
            state['monitor'].reset()
            state['feature_buffer'].clear()
            state['current_event_frames'] = 0
            state['last_motion_frames'] = 0
            state['event_active'] = False
            state['current_predicted_action'] = None
            state['current_confidence'] = 0.0
    return state