import cv2
from ultralytics import YOLO

# 1. Cargar modelo YOLO preentrenado
# Puedes cambiar yolov8n.pt por yolov8s.pt, yolov8m.pt, etc.
model = YOLO("yolov8n.pt")

# 2. Inicialización de la cámara USB
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error crítico: No se pudo abrir la interfaz de la cámara USB.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error de lectura de frame. Abortando flujo estándar.")
        break

    frame_h, frame_w = frame.shape[:2]

    # 3. Cálculo dinámico de la ROI
    w = int(frame_w * 0.6)
    h = int(frame_h * 0.6)
    x = (frame_w - w) // 2
    y = (frame_h - h) // 2

    roi = frame[y:y+h, x:x+w]

    # 4. Inferencia sobre la ROI
    results = model(roi, verbose=False)

    object_detected = False

    # 5. Procesamiento de resultados
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Coordenadas relativas a la ROI
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

            # Confianza y clase
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Traslación de coordenadas al frame original
            fx1, fy1 = bx1 + x, by1 + y
            fx2, fy2 = bx2 + x, by2 + y

            # Renderizado de detecciones
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (fx1, max(fy1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            object_detected = True

    # 6. Actualización de estado del sistema
    if object_detected:
        status = "OBJETO(S) DETECTADO(S)"
        color_status = (0, 255, 0)
    else:
        status = "ZONA DESPEJADA"
        color_status = (0, 0, 255)

    # 7. Renderizado de UI estática
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(
        frame,
        "Zona de inspeccion",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        status,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_status,
        2
    )

    # 8. Salida por pantalla
    cv2.imshow("Sistema de Inspeccion YOLO - Windows", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberación de recursos
cap.release()
cv2.destroyAllWindows()