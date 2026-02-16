#!/usr/bin/env python3
"""
Fast object tracking with servo-mounted webcam on Radxa Rock 5C.

Optimizations vs servo_object_tracker.py:
  - yolo26n (nano) instead of yolo26m (medium)
  - 320x320 inference resolution
  - Threaded inference: servo updates every camera frame, not blocked by YOLO
  - Frame skipping: only runs YOLO when previous inference is done

Controls:
  - Left click: select object to track
  - 'r': reset selection
  - 'q': quit
"""

import threading
import time

import cv2
from ultralytics import YOLO
from periphery import PWM

# --- Servo config (PWM15_IR_M3 = pwmchip4) ---
PWM_CHIP = 4
PWM_CHANNEL = 0
SERVO_FREQ = 50
MIN_PULSE = 0.0005
MAX_PULSE = 0.0025
CENTER_ANGLE = 90.0
MIN_ANGLE = 0.0
MAX_ANGLE = 180.0

# --- Tracking tuning ---
DEADZONE = 0.05
STEP_GAIN = 15.0
SMOOTHING = 0.3
INFER_SIZE = 640   # RKNN model is compiled for 640x640 (static shape)


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


class Servo:
    def __init__(self, chip, channel, frequency=50):
        self.pwm = PWM(chip, channel)
        self.pwm.frequency = frequency
        self.pwm.duty_cycle = 0
        self._fix_polarity(chip, channel)
        self.freq = frequency
        self.angle = CENTER_ANGLE
        self._set_angle(self.angle)
        self.pwm.enable()

    def _fix_polarity(self, chip, channel):
        path = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}/polarity"
        try:
            with open(path, "r") as f:
                if f.read().strip() == "inversed":
                    with open(path, "w") as fw:
                        fw.write("normal")
        except Exception:
            pass

    def _set_angle(self, angle):
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        self.angle = angle
        pulse = MIN_PULSE + (MAX_PULSE - MIN_PULSE) * (angle / 180.0)
        self.pwm.duty_cycle = pulse * self.freq

    def move_to(self, angle):
        self._set_angle(angle)

    def close(self):
        try:
            self.pwm.duty_cycle = 0
            self.pwm.disable()
        except Exception:
            pass
        try:
            self.pwm.close()
        except Exception:
            pass


class AsyncDetector:
    """Runs YOLO inference in a background thread, never blocking the main loop."""

    def __init__(self, model_path, infer_size=320):
        self.model = YOLO(model_path)
        self.infer_size = infer_size

        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.running = True

        self._input_frame = None
        self._result = None
        self._infer_fps = 0.0

        thread = threading.Thread(target=self._worker)
        thread.daemon = True
        thread.start()

    def submit(self, frame):
        """Submit a frame for inference (non-blocking, drops if busy)."""
        with self.lock:
            self._input_frame = frame
        self.frame_ready.set()

    def get_result(self):
        """Get latest detection result and inference FPS (non-blocking)."""
        with self.lock:
            return self._result, self._infer_fps

    def _worker(self):
        while self.running:
            self.frame_ready.wait()
            self.frame_ready.clear()

            with self.lock:
                frame = self._input_frame
            if frame is None:
                continue

            t0 = time.time()
            results = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                imgsz=self.infer_size, verbose=False
            )
            dt = time.time() - t0

            with self.lock:
                self._result = results[0]
                self._infer_fps = 1.0 / dt if dt > 0 else 0

    def close(self):
        self.running = False
        self.frame_ready.set()


def main():
    detector = AsyncDetector("./yolo26n_rknn_model", INFER_SIZE)
    cap = Camera(0)
    servo = Servo(PWM_CHIP, PWM_CHANNEL, SERVO_FREQ)

    selected_id = None
    current_angle = CENTER_ANGLE
    last_result = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_id
        if event == cv2.EVENT_LBUTTONDOWN:
            result = param
            if result is not None and result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    coords = box.xyxy[0]
                    if any(c != c for c in coords):
                        continue
                    x1, y1, x2, y2 = map(int, coords)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_id = int(box.id.item())
                        print(f"Tracking ID: {selected_id}")
                        return

    cv2.namedWindow("Servo Tracker Fast")
    cv2.setMouseCallback("Servo Tracker Fast", mouse_callback)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_cx = frame_w / 2.0

            # Submit frame (non-blocking, drops if detector is busy)
            detector.submit(frame)

            # Get latest result (may be from a previous frame)
            result, infer_fps = detector.get_result()
            if result is not None:
                last_result = result

            cv2.setMouseCallback("Servo Tracker Fast", mouse_callback, last_result)

            annotated = frame.copy()
            target_box = None

            if last_result is not None and last_result.boxes is not None:
                for box in last_result.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    coords = box.xyxy[0]
                    if any(c != c for c in coords):
                        continue
                    x1, y1, x2, y2 = map(int, coords)

                    if track_id == selected_id:
                        color, thickness = (0, 0, 255), 4
                        target_box = (x1, y1, x2, y2)
                    else:
                        color, thickness = (0, 255, 0), 2

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(annotated, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Servo tracking (runs every frame, not blocked by inference) ---
            if selected_id is not None and target_box is not None:
                tx = (target_box[0] + target_box[2]) / 2.0
                error = (tx - frame_cx) / frame_cx

                if abs(error) > DEADZONE:
                    adjustment = -error * STEP_GAIN
                    current_angle += SMOOTHING * adjustment
                    current_angle = max(MIN_ANGLE, min(MAX_ANGLE, current_angle))
                    servo.move_to(current_angle)

                tcx, tcy = int(tx), int((target_box[1] + target_box[3]) / 2)
                cv2.drawMarker(annotated, (tcx, tcy), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)

            # Draw frame center line
            cv2.line(annotated, (int(frame_cx), 0), (int(frame_cx), frame_h),
                     (255, 255, 0), 1)

            # HUD
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(annotated, f"Display: {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detect:  {infer_fps:.1f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Servo: {current_angle:.1f} deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if selected_id is not None:
                status = "TRACKING" if target_box else "LOST"
                cv2.putText(annotated, f"ID {selected_id}: {status}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Servo Tracker Fast", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                selected_id = None
                current_angle = CENTER_ANGLE
                servo.move_to(CENTER_ANGLE)
                print("Selection reset, servo centered.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        detector.close()
        servo.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
