import time

import cv2
import numpy as np
from rknn.api import RKNN

MODEL_PATH = "yolo26n_rknn_model/yolo26n-rk3588.rknn"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Generate a unique color per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)


def letterbox(img, new_shape=(INPUT_SIZE, INPUT_SIZE)):
    """Resize and pad image while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    pad_h = new_shape[0] - new_h
    pad_w = new_shape[1] - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, scale, (top, left)


def postprocess(output, scale, pad, orig_shape):
    """Parse YOLO output (1, 84, 8400) -> list of (x1, y1, x2, y2, conf, class_id)."""
    # output shape: (1, 84, 8400) -> (84, 8400)
    pred = output[0].squeeze(0) if output[0].ndim == 3 else output[0]

    # pred: (84, 8400) = 4 box coords + 80 class scores
    boxes = pred[:4, :]         # (4, 8400) in cx, cy, w, h
    scores = pred[4:, :]        # (80, 8400)

    class_ids = np.argmax(scores, axis=0)           # (8400,)
    confidences = scores[class_ids, np.arange(scores.shape[1])]  # (8400,)

    # Filter by confidence
    mask = confidences > CONF_THRESHOLD
    boxes = boxes[:, mask].T          # (N, 4)
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert cx, cy, w, h -> x1, y1, x2, y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Remove padding and rescale to original image
    pad_top, pad_left = pad
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale

    # Clip to image bounds
    orig_h, orig_w = orig_shape[:2]
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        results.append((int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]),
                        float(confidences[idx]), int(class_ids[idx])))
    return results


def draw_detections(img, detections):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        color = tuple(int(c) for c in COLORS[cls_id])
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    # Init RKNN
    rknn = RKNN()
    print("Loading RKNN model...")
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"Failed to load RKNN model: {ret}")
        return

    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        print(f"Failed to init runtime: {ret}")
        return
    print("RKNN model loaded and runtime initialized.")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        rknn.release()
        return

    print("Running YOLO object detection. Press 'q' to quit.")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess
        img_input, scale, pad = letterbox(frame)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_input, axis=0)

        # Inference
        outputs = rknn.inference(inputs=[img_input])

        # Postprocess and draw
        detections = postprocess(outputs, scale, pad, frame.shape)
        frame = draw_detections(frame, detections)

        # FPS counter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO RKNN Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()


if __name__ == "__main__":
    main()
