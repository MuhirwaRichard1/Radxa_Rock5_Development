import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("./yolo26n_rknn_model")

cap = cv2.VideoCapture(0)

TARGET_FILL = 0.45      # object should fill 45% of screen
MIN_ZOOM = 1.0
MAX_ZOOM = 3.0
ZOOM_SMOOTH = 0.08
MOVE_SMOOTH = 0.15


selected_id = None
zoom_factor = 1.0
target_zoom = 1.0

# Smooth parameters
ZOOM_SPEED = 0.08
MOVE_SPEED = 0.15


smooth_center = None


def mouse_callback(event, x, y, flags, param):
    global selected_id, target_zoom

    if event == cv2.EVENT_LBUTTONDOWN:
        result = param
        if result is None or result.boxes is None:
            return

        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track_id
                target_zoom = MAX_ZOOM
                print(f"Selected ID: {selected_id}")

    # Right click → reset zoom
    if event == cv2.EVENT_RBUTTONDOWN:
        selected_id = None
        target_zoom = 1.0


cv2.namedWindow("Auto Zoom Tracker")
cv2.setMouseCallback("Auto Zoom Tracker", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    result = results[0]
    cv2.setMouseCallback("Auto Zoom Tracker", mouse_callback, result)

    selected_box = None

    if result.boxes is not None:
        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if track_id == selected_id:
                selected_box = (x1, y1, x2, y2)

    if selected_box is not None:
        x1, y1, x2, y2 = selected_box

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        frame_area = w * h

        object_ratio = box_area / frame_area

        # --- Dynamic Zoom Calculation ---
        if object_ratio > 0:
            desired_zoom = (TARGET_FILL / object_ratio) ** 0.5
            desired_zoom = max(MIN_ZOOM, min(MAX_ZOOM, desired_zoom))
        else:
            desired_zoom = 1.0

        target_zoom = desired_zoom

        # --- Smooth center tracking ---
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if smooth_center is None:
            smooth_center = np.array([cx, cy], dtype=np.float32)

        target_center = np.array([cx, cy], dtype=np.float32)
        smooth_center += MOVE_SMOOTH * (target_center - smooth_center)

    else:
        target_zoom = 1.0
        smooth_center = np.array([w // 2, h // 2], dtype=np.float32)


    # Smooth zoom transition
    zoom_factor += ZOOM_SMOOTH * (target_zoom - zoom_factor)

    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)

    cx, cy = int(smooth_center[0]), int(smooth_center[1])

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    crop = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(crop, (w, h))

    # Draw tracking boxes on zoomed frame
    if result.boxes is not None:
        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])

            # Adjust box position relative to crop
            x1b = int((x1b - x1) * w / crop_w)
            x2b = int((x2b - x1) * w / crop_w)
            y1b = int((y1b - y1) * h / crop_h)
            y2b = int((y2b - y1) * h / crop_h)

            if track_id == selected_id:
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(zoomed_frame, (x1b, y1b), (x2b, y2b), color, thickness)
            cv2.putText(
                zoomed_frame,
                f"ID {track_id}",
                (x1b, y1b - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Auto Zoom Tracker", zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
