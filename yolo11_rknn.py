import cv2
import numpy as np
import sys
from rknnlite.api import RKNNLite

MODEL_PATH = "/home/radxa/Desktop/rknn_models/yolo11s.rknn"
INPUT_SIZE = 640
CONF_THRESH = 0.5
NMS_THRESH = 0.45

COCO_CLASSES = [
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def dfl(box_dist, reg_max=16):
    """Distribution Focal Loss decode: convert distribution to distances."""
    b, c, h, w = box_dist.shape
    box_dist = box_dist.reshape(b, 4, reg_max, h, w)
    box_dist = softmax(box_dist, axis=2)
    proj = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    return np.sum(box_dist * proj, axis=2)  # (b, 4, h, w)


def decode_outputs(outputs, img_shape):
    """Decode the 9 RKNN outputs into boxes, scores, and class IDs."""
    strides = [8, 16, 32]
    all_boxes = []
    all_scores = []
    all_class_ids = []

    for i, stride in enumerate(strides):
        box_dist = outputs[i * 3]       # (1, 64, h, w)
        cls_score = outputs[i * 3 + 1]  # (1, 80, h, w)
        obj_score = outputs[i * 3 + 2]  # (1, 1, h, w)

        _, _, h, w = cls_score.shape

        # Decode box distances via DFL
        distances = dfl(box_dist)  # (1, 4, h, w)

        # Build grid
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Decode to x1, y1, x2, y2
        dist = distances[0]  # (4, h, w) -> left, top, right, bottom
        x1 = (grid_x - dist[0]) * stride
        y1 = (grid_y - dist[1]) * stride
        x2 = (grid_x + dist[2]) * stride
        y2 = (grid_y + dist[3]) * stride

        # Combine scores: class_score * objectness
        scores = 1.0 / (1.0 + np.exp(-cls_score[0]))  # sigmoid, (80, h, w)
        obj = 1.0 / (1.0 + np.exp(-obj_score[0, 0]))  # sigmoid, (h, w)
        scores = scores * obj[np.newaxis, :, :]

        # Flatten
        x1 = x1.flatten()
        y1 = y1.flatten()
        x2 = x2.flatten()
        y2 = y2.flatten()
        scores = scores.reshape(len(COCO_CLASSES), -1).T  # (h*w, 80)

        class_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = max_scores > CONF_THRESH
        all_boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1))
        all_scores.append(max_scores[mask])
        all_class_ids.append(class_ids[mask])

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    return boxes, scores, class_ids


def nms(boxes, scores, class_ids):
    """Apply per-class NMS."""
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRESH,
        NMS_THRESH,
    )
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    indices = indices.flatten()
    return boxes[indices], scores[indices], class_ids[indices]


def preprocess(frame):
    """Letterbox resize and normalize for 640x640 NCHW input."""
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h
    top, left = pad_h // 2, pad_w // 2

    img = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.uint8)
    return img, scale, left, top


def draw_detections(frame, boxes, scores, class_ids, scale, pad_left, pad_top):
    """Draw bounding boxes on the original frame."""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1 = int((box[0] - pad_left) / scale)
        y1 = int((box[1] - pad_top) / scale)
        x2 = int((box[2] - pad_left) / scale)
        y2 = int((box[3] - pad_top) / scale)

        label = f"{COCO_CLASSES[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    # Init RKNN
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"Error: Failed to load RKNN model ({ret})")
        sys.exit(1)

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print(f"Error: Failed to init RKNN runtime ({ret})")
        sys.exit(1)

    # Init webcam
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("YOLO11s RKNN running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img, scale, pad_left, pad_top = preprocess(frame)
        outputs = rknn.inference(inputs=[img])

        boxes, scores, class_ids = decode_outputs(outputs, frame.shape)
        if len(boxes) > 0:
            boxes, scores, class_ids = nms(boxes, scores, class_ids)

        if len(boxes) > 0:
            frame = draw_detections(frame, boxes, scores, class_ids,
                                    scale, pad_left, pad_top)

        cv2.imshow("YOLO11s RKNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    rknn.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
