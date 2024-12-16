import cv2 #библеотека камеры
import numpy as np #библиотека расчетов
from notifypy import Notify#библиотека уведомлений
# Загрузка YOLO( мат-модель для определения объектов)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Проверяем индексы и корректируем их
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Загружаем классы объектов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Инициализируем видеопоток
cap = cv2.VideoCapture(0)  # 0 для использования веб-камеры, или замените на свой файл видео

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Подготовка изображения
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)#Отдаем нейронке

    # Получение вывода от сети
    outs = net.forward(output_layers)

    # Обработка результатов
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Прямоугольник вокруг объектов
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Непосредственно детекция личностей
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_detected = False  # Флаг для отслеживания обнаружения человека

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                person_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Рисуем прямоугольник
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    if person_detected:
        print("Объект 'person' обнаружен!")

    elif person_detected == False:
        print("Объект 'person' пропал!")
        notification = Notify()
        notification.title = "Человек убежал"
        notification.message = "Догоняй пока не съели"
        notification.send()
    cv2.imshow("Image", frame)

    # Выйти из цикла при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
