#!/usr/bin/env python3
"""
Interactive Teaching Robot - Radxa Rock 5C

Pan-tilt servo-mounted webcam tracks the largest person by default.
Press 'd' to describe what the camera sees, or type a question in the
terminal to ask about the current view. Responses are overlaid on the
camera feed.

Controls:
  d        - Describe current view (VLM)
  c        - Clear response text
  r        - Reset servos to center
  q        - Quit
  <Enter>  - Type question in terminal, press Enter to ask about view
"""

import sys
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np
from ultralytics import YOLO
from periphery import PWM

# --- Servo config ---
PAN_CHIP = 4       # PWM15_IR_M3, pin 3
PAN_CHANNEL = 0
TILT_CHIP = 3      # PWM14_M2, pin 5
TILT_CHANNEL = 0
SERVO_FREQ = 50

MIN_PULSE = 0.0005
MAX_PULSE = 0.0025
CENTER_ANGLE = 90.0
MIN_ANGLE = 0.0
MAX_ANGLE = 180.0
TILT_MIN = 30.0
TILT_MAX = 150.0

# --- Tracking tuning ---
DEADZONE = 0.05
PAN_GAIN = 15.0
TILT_GAIN = 10.0
SMOOTHING = 0.3
INFER_SIZE = 640
PERSON_CLASS = 0  # COCO class 0 = person

# --- Display ---
RESPONSE_TIMEOUT = 15.0  # seconds before auto-clearing response


class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()


class Servo:
    def __init__(self, chip, channel, frequency=50, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.freq = frequency
        self.pwm = PWM(chip, channel)
        self.pwm.frequency = frequency
        self.pwm.duty_cycle = 0
        self._fix_polarity(chip, channel)
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
        angle = max(self.min_angle, min(self.max_angle, angle))
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
    """Runs YOLO inference in a background thread."""

    def __init__(self, model_path, infer_size=640):
        self.model = YOLO(model_path)
        self.infer_size = infer_size
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.running = True
        self._input_frame = None
        self._result = None
        self._infer_fps = 0.0
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def submit(self, frame):
        with self.lock:
            self._input_frame = frame
        self.frame_ready.set()

    def get_result(self):
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


def _find_largest_person(result):
    """Return (x1, y1, x2, y2) of the largest person box, or None."""
    if result is None or result.boxes is None:
        return None
    best = None
    best_area = 0
    for box in result.boxes:
        cls = int(box.cls.item()) if box.cls is not None else -1
        if cls != PERSON_CLASS:
            continue
        coords = box.xyxy[0]
        if any(c != c for c in coords):  # NaN check
            continue
        x1, y1, x2, y2 = map(int, coords)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def _wrap_text(text, max_chars=50):
    """Word-wrap text to lines of max_chars width."""
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    return lines


def _draw_response_overlay(frame, text, max_lines=6):
    """Draw semi-transparent box with wrapped text at bottom of frame."""
    if not text:
        return
    h, w = frame.shape[:2]
    lines = _wrap_text(text, max_chars=int(w / 12))
    lines = lines[:max_lines]
    if not lines:
        return

    line_h = 28
    pad = 10
    box_h = len(lines) * line_h + 2 * pad
    y_start = h - box_h

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    for i, line in enumerate(lines):
        y = y_start + pad + (i + 1) * line_h - 6
        cv2.putText(frame, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)


def _input_reader(queue):
    """Read lines from stdin in a background thread."""
    try:
        while True:
            line = input()
            if line.strip():
                queue.put(line.strip())
    except EOFError:
        pass


def main():
    print("=== Interactive Teaching Robot ===")
    print("Loading models...")

    # Load VLM (slow - do first so user sees progress)
    from vlm_helper import VLMHelper
    vlm = VLMHelper()

    # Load YOLO + hardware
    detector = AsyncDetector("./yolo26n_rknn_model", INFER_SIZE)
    camera = Camera(0)
    pan = Servo(PAN_CHIP, PAN_CHANNEL, SERVO_FREQ)
    tilt = Servo(TILT_CHIP, TILT_CHANNEL, SERVO_FREQ,
                 min_angle=TILT_MIN, max_angle=TILT_MAX)

    # State
    pan_angle = CENTER_ANGLE
    tilt_angle = CENTER_ANGLE
    mode = "tracking"  # tracking | thinking
    response_text = ""
    response_time = 0.0
    last_result = None

    # Input thread
    input_queue = Queue()
    input_thread = threading.Thread(target=_input_reader, args=(input_queue,), daemon=True)
    input_thread.start()

    # VLM worker thread
    vlm_queue = Queue()      # (frame, question) → VLM thread picks up
    vlm_result = Queue()     # VLM thread puts result string

    def _vlm_worker():
        while True:
            item = vlm_queue.get()
            if item is None:
                break
            frame, question = item
            try:
                answer = vlm.describe(frame, question)
            except Exception as e:
                answer = f"[Error: {e}]"
            vlm_result.put(answer)

    vlm_thread = threading.Thread(target=_vlm_worker, daemon=True)
    vlm_thread.start()

    cv2.namedWindow("Teaching Robot")
    prev_time = time.time()

    print("\nReady! Controls:")
    print("  d = describe view | c = clear | r = reset servos | q = quit")
    print("  Or type a question and press Enter.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]
            frame_cx = frame_w / 2.0
            frame_cy = frame_h / 2.0

            # Submit to YOLO
            detector.submit(frame)
            result, infer_fps = detector.get_result()
            if result is not None:
                last_result = result

            annotated = frame.copy()

            # --- Face tracking ---
            person_box = _find_largest_person(last_result)

            if mode == "tracking" and person_box is not None:
                x1, y1, x2, y2 = person_box
                tx = (x1 + x2) / 2.0
                ty = (y1 + y2) / 2.0

                error_x = (tx - frame_cx) / frame_cx
                error_y = (ty - frame_cy) / frame_cy

                if abs(error_x) > DEADZONE:
                    pan_angle += SMOOTHING * (-error_x * PAN_GAIN)
                    pan_angle = max(MIN_ANGLE, min(MAX_ANGLE, pan_angle))
                    pan.move_to(pan_angle)

                if abs(error_y) > DEADZONE:
                    tilt_angle += SMOOTHING * (-error_y * TILT_GAIN)
                    tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))
                    tilt.move_to(tilt_angle)

                # Draw tracking box + crosshair
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.drawMarker(annotated, (int(tx), int(ty)), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)

            # Draw all person boxes faintly
            if last_result is not None and last_result.boxes is not None:
                for box in last_result.boxes:
                    cls = int(box.cls.item()) if box.cls is not None else -1
                    if cls != PERSON_CLASS:
                        continue
                    coords = box.xyxy[0]
                    if any(c != c for c in coords):
                        continue
                    bx1, by1, bx2, by2 = map(int, coords)
                    if person_box and (bx1, by1, bx2, by2) == person_box:
                        continue  # already drawn above
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (128, 128, 128), 1)

            # --- Check for VLM result ---
            try:
                answer = vlm_result.get_nowait()
                response_text = answer
                response_time = time.time()
                mode = "tracking"
                print(f"VLM: {answer}")
            except Empty:
                pass

            # Auto-clear old responses
            if response_text and (time.time() - response_time > RESPONSE_TIMEOUT):
                response_text = ""

            # --- Check for user input ---
            # Keyboard (OpenCV)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                if mode != "thinking":
                    mode = "thinking"
                    response_text = "Thinking..."
                    vlm_queue.put((frame.copy(), "Describe what you see in this image briefly."))
                    print("Describing view...")
            elif key == ord("c"):
                response_text = ""
            elif key == ord("r"):
                pan_angle = CENTER_ANGLE
                tilt_angle = CENTER_ANGLE
                pan.move_to(CENTER_ANGLE)
                tilt.move_to(CENTER_ANGLE)
                print("Servos centered.")

            # Terminal input
            try:
                question = input_queue.get_nowait()
                if mode != "thinking":
                    mode = "thinking"
                    response_text = "Thinking..."
                    vlm_queue.put((frame.copy(), question))
                    print(f"Asking: {question}")
            except Empty:
                pass

            # --- HUD ---
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Center crosshair
            cv2.line(annotated, (int(frame_cx), 0), (int(frame_cx), frame_h),
                     (255, 255, 0), 1)
            cv2.line(annotated, (0, int(frame_cy)), (frame_w, int(frame_cy)),
                     (255, 255, 0), 1)

            # Status text
            mode_label = "TRACKING" if mode == "tracking" else "THINKING..."
            color = (0, 255, 0) if mode == "tracking" else (0, 165, 255)
            cv2.putText(annotated, mode_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, f"FPS: {fps:.0f} | YOLO: {infer_fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated, f"Pan: {pan_angle:.0f} Tilt: {tilt_angle:.0f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # VLM response overlay
            _draw_response_overlay(annotated, response_text)

            cv2.imshow("Teaching Robot", annotated)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        vlm_queue.put(None)  # signal VLM thread to exit
        detector.close()
        pan.close()
        tilt.close()
        camera.release()
        vlm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
