from ultralytics import YOLO
import cv2


model = YOLO(r"C:\Users\Batu\Documents\yolo_denemeleri\airplane_detection\runs\detect\train2\weights\best.pt")


test_data_path = r"C:\Users\Batu\Documents\yolo_denemeleri\airplane_detection\test_video.mp4"
cap = cv2.VideoCapture(test_data_path)


if not cap.isOpened():
    print("video acılmadı")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        exit()

    results = model(frame, device="cpu")

    annonted_frame = results[0].plot()

    cv2.imshow("Yolov8 Video", annonted_frame)


    if cv2.waitKey(1) and 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()