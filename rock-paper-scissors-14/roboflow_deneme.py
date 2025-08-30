# Kamera ile Rock-Paper-Scissors tahmini - Windows ready
# Ön koşul: pip install ultralytics opencv-contrib-python torch numpy<2

import cv2
from ultralytics import YOLO

# Fine-tuned modelini yükle
model = YOLO(r"C:\Users\Batu\Documents\yolo_denemeleri\best.pt")

# Kamera aç (0 = varsayılan webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tahmini
    results = model(frame, device="cpu")  # CPU kullanıyoruz, CUDA sorunları yok

    # Annotated frame
    annotated_frame = results[0].plot()

    # Pencerede göster
    cv2.imshow("Rock-Paper-Scissors YOLOv8", annotated_frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
