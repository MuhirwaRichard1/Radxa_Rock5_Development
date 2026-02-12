import cv2
from ultralytics import YOLO

# Load the exported RKNN model
model = YOLO("./yolo26s_rknn_model")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the frame
    results = model(frame)

    # Visualize results
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO RKNN Webcam", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
