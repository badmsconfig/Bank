# config.py
# =============================================================================
# Файл конфигурации для анализа видео с MediaPipe и нейросетями
# =============================================================================

# ----------------------------- ПУТИ К ФАЙЛАМ ---------------------------------
VIDEO_PATH = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"
ZONES_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\zones.json"
CSV_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\FINAL.csv"
MODEL_PATH = r"D:\python_work\pythonProject\Bank\src\trained_models\best_model_finetuned_2_GR.keras"
CLASSES_TRAIN_PATH = r"D:\python_work\pythonProject\Bank\data_money\train"
LSTM_MODEL_PATH = r"D:\python_work\pythonProject\Bank\src\action_classifier_lstm.h5"
LABEL_ENCODER_PATH = r"D:\python_work\pythonProject\Bank\dataset_lstm_balanced\label_encoder.pkl"
WORKFLOW_CONFIG_PATH = r"D:\python_work\pythonProject\Bank\data\annotations\workflow_config.json"


# ------------------------ ПАРАМЕТРЫ РАСПОЗНАВАНИЯ ВАЛЮТ -----------------------
CONFIDENCE_THRESHOLD = 0.7       # Минимальная уверенность для смены предсказания
RECOGNITION_INTERVAL = 5         # Распознавать каждые N кадров (для производительности)
HAND_HISTORY_SIZE = 4            # Размер истории для сглаживания результатов
CLASS_EMPTY = 'empty'            # Название класса "пустая рука"
HAND_BBOX_PADDING_FACTOR = 0.3   # Отступ вокруг кисти при вырезании области
USE_ONLY_HAND_PALM = False       # Использовать только ладонь (пальцы исключить)

# ------------------------ ПАРАМЕТРЫ LSTM КЛАССИФИКАТОРА -----------------------
INFERENCE_DELAY_SEC = 1.0        # Сколько секунд без движения для запуска классификации
MIN_SEQ_LEN = 10                 # Минимальная длина последовательности кадров
ANOMALY_DURATION_MULT = 2.0      # Множитель допустимой длительности действия (среднее+стд*коэфф)

# ------------------------ УПРАВЛЕНИЕ ВИДЕО -----------------------------------
SEEK_STEP = 30                   # Кол-во кадров для перемотки кнопками

# ------------------------ ПАРАМЕТРЫ ДЕТЕКЦИИ ДВИЖЕНИЯ ------------------------
MOTION_THRESHOLD = 0.15          # Порог суммы изменений координат рук (чем выше, тем менее чувствительно)

# ------------------------ GUI ФЛАГИ (по умолчанию) --------------------------
SHOW_POSE_DEFAULT = False        # Показывать скелет всего тела
SHOW_HANDS_DEFAULT = True        # Показывать bounding box и траекторию рук
SHOW_INFO_DEFAULT = True         # Показывать информационную панель

# ------------------------ РАЗМЕРЫ ИНТЕРФЕЙСА --------------------------------
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_Y = 30
BUTTON_SPACING = 10