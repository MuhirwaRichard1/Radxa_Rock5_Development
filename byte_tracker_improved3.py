import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("./yolo26n_rknn_model")

cap = cv2.VideoCapture(0)

PIP_SCALE = 0.3         # PIP width relative to frame width
PIP_PADDING = 20        # extra pixels around object for zoomed view
PIP_MARGIN = 10         # distance from screen edges

selected_id = None
smooth_pip_center = None
MOVE_SPEED = 0.2


def mouse_callback(event, x, y, flags, param):
    global selected_id

    if event == cv2.EVENT_LBUTTONDOWN:
        result = param
        if result is None or result.boxes is None:
            return

        for box in result.boxes:
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

    # Right click to deselect
    if event == cv2.EVENT_RBUTTONDOWN:
        selected_id = None


cv2.namedWindow("PIP Tracker")
cv2.setMouseCallback("PIP Tracker", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    result = results[0]
    cv2.setMouseCallback("PIP Tracker", mouse_callback, result)

    display = frame.copy()
    selected_box = None

    # Draw all tracking boxes on main frame (no zoom)
    if result.boxes is not None:
        for box in result.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            coords = box.xyxy[0]
            if any(c != c for c in coords):
                continue
            x1, y1, x2, y2 = map(int, coords)

            if track_id == selected_id:
                selected_box = (x1, y1, x2, y2)
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---- PIP: always visible, zoomed view in top-right corner ----
    # Default: center of frame. When object selected: center on object.
    if selected_box is not None:
        sx1, sy1, sx2, sy2 = selected_box
        target_cx = (sx1 + sx2) // 2
        target_cy = (sy1 + sy2) // 2
        obj_w = sx2 - sx1
        obj_h = sy2 - sy1
        crop_size = max(obj_w, obj_h) + PIP_PADDING * 2
    else:
        target_cx = w // 2
        target_cy = h // 2
        crop_size = min(w, h) // 3  # default zoom on center

    # Smooth the PIP center
    if smooth_pip_center is None:
        smooth_pip_center = np.array([target_cx, target_cy], dtype=np.float32)
    target = np.array([target_cx, target_cy], dtype=np.float32)
    smooth_pip_center += MOVE_SPEED * (target - smooth_pip_center)

    cx = int(smooth_pip_center[0])
    cy = int(smooth_pip_center[1])

    crop_x1 = max(0, cx - crop_size // 2)
    crop_y1 = max(0, cy - crop_size // 2)
    crop_x2 = min(w, crop_x1 + crop_size)
    crop_y2 = min(h, crop_y1 + crop_size)

    pip_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    pip_width = int(w * PIP_SCALE)
    pip_height = pip_width  # square PIP

    if pip_crop.size != 0:
        pip_img = cv2.resize(pip_crop, (pip_width, pip_height))

        # Draw box around selected object inside PIP
        if selected_box is not None:
            scale_x = pip_width / (crop_x2 - crop_x1)
            scale_y = pip_height / (crop_y2 - crop_y1)
            pip_ox1 = int((sx1 - crop_x1) * scale_x)
            pip_oy1 = int((sy1 - crop_y1) * scale_y)
            pip_ox2 = int((sx2 - crop_x1) * scale_x)
            pip_oy2 = int((sy2 - crop_y1) * scale_y)
            cv2.rectangle(pip_img, (pip_ox1, pip_oy1), (pip_ox2, pip_oy2), (0, 0, 255), 2)

        # Position: top-right
        x_offset = w - pip_width - PIP_MARGIN
        y_offset = PIP_MARGIN

        if y_offset + pip_height <= h and x_offset >= 0:
            # Overlay PIP
            display[y_offset:y_offset + pip_height,
                    x_offset:x_offset + pip_width] = pip_img

            # Label
            if selected_id is not None:
                label = f"ID {selected_id}"
            else:
                label = "Center"
            cv2.putText(display, label,
                        (x_offset + 5, y_offset + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("PIP Tracker", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        selected_id = None

cap.release()
cv2.destroyAllWindows()
