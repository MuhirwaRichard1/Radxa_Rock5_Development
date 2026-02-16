#!/usr/bin/env python3
"""
Object tracking with servo-mounted webcam on Radxa Rock 5C.

Click on a detected object to select it. The servo (PWM15 on pin 3)
will pan to keep the selected object centered in the frame.

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
MIN_PULSE = 0.0005   # 0.5 ms → 0°
MAX_PULSE = 0.0025   # 2.5 ms → 180°
CENTER_ANGLE = 90.0
MIN_ANGLE = 0.0
MAX_ANGLE = 180.0

# --- Tracking tuning ---
DEADZONE = 0.05       # fraction of frame width; no movement if error is within this
STEP_GAIN = 15.0      # degrees per unit of normalized error
SMOOTHING = 0.3       # low-pass filter (0 = ignore new, 1 = instant response)


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


def main():
    model = YOLO("./yolo26m_rknn_model")
    cap = Camera(0)
    servo = Servo(PWM_CHIP, PWM_CHANNEL, SERVO_FREQ)

    selected_id = None
    current_angle = CENTER_ANGLE

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

    cv2.namedWindow("Servo Tracker")
    cv2.setMouseCallback("Servo Tracker", mouse_callback)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_cx = frame_w / 2.0

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            result = results[0]

            cv2.setMouseCallback("Servo Tracker", mouse_callback, result)

            annotated = frame.copy()
            target_box = None

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
                        color, thickness = (0, 0, 255), 4
                        target_box = (x1, y1, x2, y2)
                    else:
                        color, thickness = (0, 255, 0), 2

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(annotated, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Servo tracking logic ---
            if selected_id is not None and target_box is not None:
                tx = (target_box[0] + target_box[2]) / 2.0
                error = (tx - frame_cx) / frame_cx  # -1.0 to 1.0

                if abs(error) > DEADZONE:
                    # Negative error = object is left = increase angle (pan left)
                    adjustment = -error * STEP_GAIN
                    current_angle += SMOOTHING * adjustment
                    current_angle = max(MIN_ANGLE, min(MAX_ANGLE, current_angle))
                    servo.move_to(current_angle)

                # Draw crosshair on target center
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
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Servo: {current_angle:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if selected_id is not None:
                status = "TRACKING" if target_box else "LOST"
                cv2.putText(annotated, f"ID {selected_id}: {status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Servo Tracker", annotated)

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
        servo.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
