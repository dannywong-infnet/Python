from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()