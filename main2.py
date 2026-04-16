import cv2
import config
from ultralytics import YOLO

# Cargar el modelo YOLO ligero. 
# La primera vez que se ejecute, descargará el archivo 'yolov8n.pt' automáticamente.
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara USB.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Fallo al capturar frame de la cámara.")
        break

    frame_h, frame_w = frame.shape[:2]

    # Generación de ROI centrada (se mantiene la lógica dinámica original)
    w = int(frame_w * 0.6)
    h = int(frame_h * 0.6)
    x = (frame_w - w) // 2
    y = (frame_h - h) // 2

    # Extraer la región de interés
    roi = frame[y:y+h, x:x+w]

    # Inferencia de YOLO sobre la ROI para reducir carga computacional
    # stream=True reduce el uso de memoria RAM
    # verbose=False evita saturar la terminal con logs de inferencia
    results = model(roi, stream=True, verbose=False)

    object_detected = False

    # Procesar resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas relativas a la ROI
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Clase y confianza
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            # Trasladar coordenadas de la ROI al Frame original
            fx1, fy1 = bx1 + x, by1 + y
            fx2, fy2 = bx2 + x, by2 + y

            # Dibujar bounding box del objeto
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            
            # Etiqueta de clase y confianza
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            object_detected = True

    # Actualizar estado global
    if object_detected:
        status = "OBJETO(S) DETECTADO(S)"
        color_status = (0, 255, 0)
    else:
        status = "ZONA DESPEJADA"
        color_status = (0, 0, 255)

    # Dibujar la zona de inspección (ROI) en el frame principal
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, "Zona de inspeccion", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Mostrar estado general
    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

    # Mostrar ventanas
    cv2.imshow("Camara Principal", frame)
    cv2.imshow("ROI Procesada", roi)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
