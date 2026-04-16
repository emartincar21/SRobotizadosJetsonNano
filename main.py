import cv2
import numpy as np
import config

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame")
        break

    frame_h, frame_w = frame.shape[:2]

    # ROI centrada 
    w = int(frame_w * 0.6)
    h = int(frame_h * 0.6)
    x = (frame_w - w) // 2
    y = (frame_h - h) // 2

    roi = frame[y:y+h, x:x+w]

    #detección básica de objeto ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    area = cv2.countNonZero(thresh)

    if area > config.MIN_OBJECT_AREA:
        status = "OBJETO DETECTADO"
        color = (0, 255, 0)
    else:
        status = "SIN OBJETO"
        color = (0, 0, 255)

    #pantalla 
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, "Zona de inspeccion", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"Area detectada: {area}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Camara", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Mascara objeto", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()