import cv2
import json
import numpy as np

VIDEO_PATH = r"D:\python_work\pythonProject\Bank\data\raw\video_01_10min.mp4"
OUTPUT_FILE = "zones.json"

zones = {}
current_polygon = []
point_radius = 3

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Не удалось загрузить видео")
    exit()

base_frame = frame.copy()

cv2.namedWindow("Zone Editor (Polygons)")
cv2.setMouseCallback("Zone Editor (Polygons)", lambda event, x, y, flags, param: mouse_callback(event, x, y))

def mouse_callback(event, x, y):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        print(f"Точка {len(current_polygon)}: ({x}, {y})")

print("=== Редактор полигональных зон (автоименование) ===")
print("Левый клик - добавить вершину")
print("Enter - завершить полигон (имя zone_1, zone_2, ...)")
print("Backspace - удалить последнюю вершину")
print("R - сбросить все зоны")
print("S - сохранить в файл")
print("ESC - выход")

zone_counter = 1

while True:
    display = base_frame.copy()

    # Рисуем сохранённые зоны
    for name, points in zones.items():
        if len(points) >= 3:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], isClosed=True, color=(0,255,0), thickness=2)
            # Подпись в центре полигона
            cx = int(sum(p[0] for p in points) / len(points))
            cy = int(sum(p[1] for p in points) / len(points))
            cv2.putText(display, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Рисуем текущий полигон
    if len(current_polygon) > 0:
        for (x, y) in current_polygon:
            cv2.circle(display, (x, y), point_radius, (0,255,255), -1)
        for i in range(len(current_polygon)-1):
            cv2.line(display, current_polygon[i], current_polygon[i+1], (0,255,255), 2)
        if len(current_polygon) >= 3:
            cv2.line(display, current_polygon[-1], current_polygon[0], (0,255,255), 1, lineType=cv2.LINE_AA)

    # Подсказка
    info = f"Точек: {len(current_polygon)}. Enter - сохранить как zone_{zone_counter}"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Zone Editor (Polygons)", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 13:  # Enter
        if len(current_polygon) >= 3:
            name = f"zone_{zone_counter}"
            zones[name] = current_polygon.copy()
            print(f"Зона '{name}' сохранена")
            zone_counter += 1
            current_polygon = []
        else:
            print("Для сохранения нужно минимум 3 точки")

    elif key == ord('r') or key == ord('R'):
        zones.clear()
        current_polygon.clear()
        zone_counter = 1
        print("Все зоны сброшены")

    elif key == 8:  # Backspace
        if current_polygon:
            current_polygon.pop()
            print("Последняя точка удалена")
        else:
            print("Нет точек для удаления")

    elif key == ord('s') or key == ord('S'):
        # Конвертируем в формат с ключом "points"
        output_data = {name: {"points": points} for name, points in zones.items()}
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Зоны сохранены в {OUTPUT_FILE}")

cap.release()
cv2.destroyAllWindows()