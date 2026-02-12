import threading
import time

import cv2
from ultralytics import YOLO


class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        thread = threading.Thread(target=self.update)
        thread.daemon = True
        thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()


# Load RKNN model
model = YOLO("./yolo26m_rknn_model")

# Open webcam with threaded capture
cap = Camera(0)

selected_id = None  # Store selected track ID

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        tracks = param  # current detections
        if tracks is not None:
            for box in tracks.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                coords = box.xyxy[0]
                if any(c != c for c in coords):
                    continue
                x1, y1, x2, y2 = map(int, coords)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_id = track_id
                    print(f"Selected ID: {selected_id}")

cv2.namedWindow("YOLO ByteTrack")
cv2.setMouseCallback("YOLO ByteTrack", mouse_callback)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking (ByteTrack)
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    result = results[0]

    # Send result to mouse callback
    cv2.setMouseCallback("YOLO ByteTrack", mouse_callback, result)

    annotated_frame = frame.copy()

    if result.boxes is not None:
        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            coords = box.xyxy[0]
            if any(c != c for c in coords):  # skip NaN values
                continue
            x1, y1, x2, y2 = map(int, coords)

            # Highlight selected object
            if track_id == selected_id:
                color = (0, 0, 255)  # RED for selected
                thickness = 4
            else:
                color = (0, 255, 0)  # GREEN normal
                thickness = 2

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # FPS counter
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
