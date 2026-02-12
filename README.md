# Radxa_Rock5_Development

# ✅ 1️⃣ Check CPU Usage (While Script Is Running)

### 🔥 Best Tool:

```bash
htop
```

If not installed:

```bash
sudo apt install htop
```

### What to look for:

* 8 cores visible (RK3588 = 4x A76 + 4x A55)
* If total usage > **300–400%**, CPU is overloaded
* If one core at 100% constantly → single-thread bottleneck

💡 100% = 1 full core
So 300% = 3 cores fully used

For NPU inference, CPU should usually stay under **250%**.

---

# ✅ 2️⃣ Check If NPU Is Being Used (VERY IMPORTANT)

### Method 1 (Best Way):

```bash
sudo cat /sys/kernel/debug/rknpu/load
```

You should see something like:

```
NPU load: 65%
```

If it says:

```
NPU load: 0%
```

⚠️ You're running on CPU, not NPU.

---

### Alternative:

```bash
sudo watch -n 1 cat /sys/kernel/debug/rknpu/load
```

This updates every second.

If load increases when inference runs → NPU is working ✅

---

# ✅ 3️⃣ Show FPS Inside Your Script (Real-Time)

Add this inside your loop:

```python
import time

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    end = time.time()

    fps = 1 / (end - start)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Monitor", frame)
```

Now FPS will display on screen.

---

# ✅ 4️⃣ Check Camera Resolution (Inside Script)

Add:

```python
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

Or show it live:

```python
h, w = frame.shape[:2]
cv2.putText(frame, f"{w}x{h}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 0), 2)
```

---

# ✅ 5️⃣ Bonus: Check CPU Temperature (Important on RK3588)

```bash
cat /sys/class/thermal/thermal_zone0/temp
```

Divide by 1000.

If > 85°C → thermal throttling → FPS drops.
