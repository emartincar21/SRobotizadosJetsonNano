import cv2
import numpy as np
import jetson.inference
import jetson.utils

# 1. Inicialización del modelo SSD-MobileNet-v2 a través de TensorRT
# Nota: La primera ejecución tardará varios minutos mientras compila el motor .engine
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 2. Inicialización de la cámara USB
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error crítico: No se pudo abrir la interfaz de la cámara USB.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error de lectura de frame. Abortando flujo estandar.")
        break

    frame_h, frame_w = frame.shape[:2]

    # Cálculo dinámico de la ROI
    w = int(frame_w * 0.6)
    h = int(frame_h * 0.6)
    x = (frame_w - w) // 2
    y = (frame_h - h) // 2

    roi = frame[y:y+h, x:x+w]

    # 3. Interoperabilidad memoria CPU (OpenCV) -> GPU (CUDA)
    # jetson-inference requiere formato RGBA
    roi_rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
    
    # Conversión de matriz numpy a imagen CUDA en memoria compartida
    cuda_img = jetson.utils.cudaFromNumpy(roi_rgba)

    # 4. Inferencia
    # overlay="none" desactiva el renderizado interno para manejarlo con OpenCV
    detections = net.Detect(cuda_img, overlay="none")

    object_detected = False

    # 5. Procesamiento de tensores de salida
    for d in detections:
        # Coordenadas relativas a la ROI
        bx1, by1, bx2, by2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
        
        # Extracción de metadatos
        class_name = net.GetClassDesc(d.ClassID)
        conf = d.Confidence

        # Traslación de coordenadas al marco original absoluto
        fx1, fy1 = bx1 + x, by1 + y
        fx2, fy2 = bx2 + x, by2 + y

        # Renderizado de detecciones
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (fx1, fy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        object_detected = True

    # 6. Actualización de estado del sistema
    if object_detected:
        status = "OBJETO(S) DETECTADO(S)"
        color_status = (0, 255, 0)
    else:
        status = "ZONA DESPEJADA"
        color_status = (0, 0, 255)

    # Renderizado de UI estática
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, "Zona de inspeccion", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

    # Salida por pantalla
    cv2.imshow("Sistema de Inspeccion L4T", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberación de recursos de hardware
cap.release()
cv2.destroyAllWindows()
