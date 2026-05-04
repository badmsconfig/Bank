# action_monitor.py
# Класс для отслеживания последовательности действий и длительности
import numpy as np
import config

class ActionMonitor:
    """
    Использует загруженный таймлайн (CSV) для проверки:
    - допустимых переходов между действиями
    - превышения максимальной длительности действия
    """
    def __init__(self, timeline_df, fps, anomaly_duration_mult=config.ANOMALY_DURATION_MULT):
        self.fps = fps
        self.anomaly_duration_mult = anomaly_duration_mult
        # Набор допустимых переходов (prev_action, curr_action)
        self.transitions = set()
        prev = None
        for _, row in timeline_df.iterrows():
            curr = row['action_name']
            if prev is not None and prev != curr:
                self.transitions.add((prev, curr))
            prev = curr
        # Статистика длительности каждого действия (среднее, стд)
        self.duration_stats = {}
        for action in timeline_df['action_name'].unique():
            durations = timeline_df[timeline_df['action_name'] == action]['duration_sec'].astype(float).values
            if len(durations) > 0:
                self.duration_stats[action] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations)
                }
        self.current_action = None
        self.action_start_frame = None
        self.anomaly_message = ""

    def reset(self):
        """Сброс текущего состояния монитора."""
        self.current_action = None
        self.action_start_frame = None
        self.anomaly_message = ""

    def update(self, predicted_action, confidence, frame_id):
        """
        Обновляет монитор новым предсказанным действием.
        Возвращает сообщение об аномалии (если есть), иначе пустую строку.
        """
        self.anomaly_message = ""
        # Проверка недопустимого перехода
        if self.current_action is not None and self.current_action != predicted_action:
            if (self.current_action, predicted_action) not in self.transitions:
                self.anomaly_message = f"Недопустимый переход: {self.current_action} → {predicted_action}"
        # Проверка длительности текущего действия
        if self.current_action == predicted_action and self.action_start_frame is not None:
            duration_frames = frame_id - self.action_start_frame
            duration_sec = duration_frames / self.fps
            stats = self.duration_stats.get(predicted_action)
            if stats is not None:
                max_duration = stats['mean'] + self.anomaly_duration_mult * stats['std']
                if duration_sec > max_duration:
                    self.anomaly_message = f"Слишком долгое действие: {predicted_action} ({duration_sec:.1f}с > {max_duration:.1f}с)"
        # Смена действия
        if self.current_action != predicted_action:
            self.current_action = predicted_action
            self.action_start_frame = frame_id
        return self.anomaly_message