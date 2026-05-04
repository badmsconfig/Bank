# models.py
# Загрузка моделей Keras и класс асинхронного распознавания валют
import os
import threading
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
import pickle
import config
from utils import preprocess_hand_roi


# -----------------------------------------------------------------------------
def load_currency_model():
    """Загружает CNN модель для распознавания валют."""
    print("Загрузка модели распознавания валют...")
    model = load_model(config.MODEL_PATH)
    print("Модель валют загружена.")

    # Определяем список классов валют из папки train (если существует)
    if os.path.exists(config.CLASSES_TRAIN_PATH):
        class_names = sorted([d for d in os.listdir(config.CLASSES_TRAIN_PATH)
                              if os.path.isdir(os.path.join(config.CLASSES_TRAIN_PATH, d))])
        print("Классы валют из папки train:", class_names)
    else:
        class_names = ['empty', 'money', 'other']
        print("Внимание: используем классы по умолчанию:", class_names)

    # Приводим CLASS_EMPTY к реальному имени класса
    empty_class = config.CLASS_EMPTY
    if empty_class.lower() not in [c.lower() for c in class_names]:
        print(f"Предупреждение: класс '{empty_class}' не найден в class_names.")
    else:
        for c in class_names:
            if c.lower() == empty_class.lower():
                empty_class = c
                break
    return model, class_names, empty_class


# -----------------------------------------------------------------------------
def load_lstm_model():
    """Загружает LSTM модель классификатора действий и label encoder."""
    print("Загрузка LSTM классификатора действий...")
    lstm_model = load_model(config.LSTM_MODEL_PATH)
    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"LSTM модель загружена. Классы: {label_encoder.classes_}")
    return lstm_model, label_encoder


# -----------------------------------------------------------------------------
class AsyncRecognizer:
    """
    Асинхронный распознаватель валют для двух рук.
    Позволяет не блокировать основной поток при предсказании CNN.
    """

    def __init__(self, model, class_names, empty_class, confidence_thresh, history_size):
        self.model = model
        self.class_names = class_names
        self.empty_class = empty_class
        self.confidence_thresh = confidence_thresh
        self.history_size = history_size
        self.lock = threading.Lock()
        self.hand_states = {
            'right': {
                'last_currency': None,
                'last_confidence': 0.0,
                'history': [],
                'recognizing': False,
                'last_recognition_frame': -1,
                'frames_since_last': 0
            },
            'left': {
                'last_currency': None,
                'last_confidence': 0.0,
                'history': [],
                'recognizing': False,
                'last_recognition_frame': -1,
                'frames_since_last': 0
            }
        }

    def reset(self):
        """Сброс состояния обеих рук."""
        with self.lock:
            for hand in ['right', 'left']:
                self.hand_states[hand]['last_currency'] = None
                self.hand_states[hand]['last_confidence'] = 0.0
                self.hand_states[hand]['history'].clear()
                self.hand_states[hand]['recognizing'] = False
                self.hand_states[hand]['last_recognition_frame'] = -1
                self.hand_states[hand]['frames_since_last'] = 0

    def request_recognition(self, hand_name, hand_roi, frame_id):
        """Запускает распознавание в отдельном потоке, если не запущено."""
        with self.lock:
            if self.hand_states[hand_name]['recognizing']:
                return
            self.hand_states[hand_name]['recognizing'] = True
            self.hand_states[hand_name]['last_recognition_frame'] = frame_id
        thread = threading.Thread(target=self._recognize_thread, args=(hand_name, hand_roi))
        thread.daemon = True
        thread.start()

    def _recognize_thread(self, hand_name, hand_roi):
        """Потоковая функция: делает предсказание и обновляет состояние."""
        try:
            input_tensor = preprocess_hand_roi(hand_roi)
            preds = self.model.predict(input_tensor, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]
            pred_class = self.class_names[class_idx]
            if confidence >= self.confidence_thresh:
                with self.lock:
                    self.hand_states[hand_name]['history'].append(pred_class)
                    if len(self.hand_states[hand_name]['history']) > self.history_size:
                        self.hand_states[hand_name]['history'].pop(0)
                    history = self.hand_states[hand_name]['history']
                    if not history:
                        return
                    non_empty_classes = [c for c in history if c.lower() != self.empty_class.lower()]
                    if non_empty_classes:
                        best = Counter(non_empty_classes).most_common(1)[0][0]
                        self.hand_states[hand_name]['last_currency'] = best
                        self.hand_states[hand_name]['last_confidence'] = confidence
                    else:
                        best = Counter(history).most_common(1)[0][0]
                        self.hand_states[hand_name]['last_currency'] = best
                        self.hand_states[hand_name]['last_confidence'] = confidence
        except Exception as e:
            print(f"Error in recognition thread for {hand_name}: {e}")
        finally:
            with self.lock:
                self.hand_states[hand_name]['recognizing'] = False

    def update_frame_counter(self, hand_name):
        """Увеличивает счётчик кадров с прошлого распознавания."""
        with self.lock:
            self.hand_states[hand_name]['frames_since_last'] += 1

    def should_recognize(self, hand_name, frame_id):
        """
        Проверяет, нужно ли запустить распознавание для указанной руки.
        Условие: прошло RECOGNITION_INTERVAL кадров или последняя валюта == None.
        """
        with self.lock:
            if self.hand_states[hand_name]['recognizing']:
                return False
            if (self.hand_states[hand_name]['frames_since_last'] >= config.RECOGNITION_INTERVAL or
                    self.hand_states[hand_name]['last_currency'] is None):
                self.hand_states[hand_name]['frames_since_last'] = 0
                return True
            return False

    def get_state(self, hand_name):
        """Возвращает текущую распознанную валюту и уверенность для руки."""
        with self.lock:
            return {
                'last_currency': self.hand_states[hand_name]['last_currency'],
                'last_confidence': self.hand_states[hand_name]['last_confidence']
            }