from ultralytics import YOLO

model = YOLO(r"C:\Users\Batu\Documents\yolo_denemeleri\airplane_detection\runs\detect\train2\weights\best.pt")

test_photo = r"C:\Users\Batu\Documents\yolo_denemeleri\airplane_detection\test_photo.jpg"
result = model(test_photo, device="cpu", show=True, save=True)