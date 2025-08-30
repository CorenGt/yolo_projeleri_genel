# YOLOv8 video tahmini - çıktı videoyu kaydediyor
# Ön koşul: pip install ultralytics

from ultralytics import YOLO

# Önceden eğitilmiş COCO modeli (arabalar, insanlar vs. tanıyabilir)
model = YOLO("yolov8n.pt")  # Daha yüksek doğruluk için yolov8s.pt, yolov8m.pt de kullanılabilir

# Video yolu
video_path = r"C:\Users\Batu\Documents\yolo_denemeleri\test.mp4"

# Tahmin yap, tüm video boyunca çalış ve çıktı videoyu kaydet
results = model(
    video_path,
    device="cpu",    # GPU varsa "cuda" yazabilirsin
    imgsz=640,       # Model input boyutu
    conf=0.1,        # Confidence threshold
    save=True        # Annotated videoyu otomatik kaydeder
)

# Çıktı video runs/detect/exp/ klasöründe
print("Tahmin tamamlandı! Annotated video 'runs/detect/exp/' klasörüne kaydedildi.")
