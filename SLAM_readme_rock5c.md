# 2D SLAM on Radxa Rock 5C with SLAMTEC C1 LIDAR
## ROS2 Humble | Debian Bookworm (aarch64)

This guide documents the full setup and operation of a 2D SLAM system using
`slam_toolbox` and `rf2o_laser_odometry` (laser-based odometry — no wheel
encoders required).

---

## Hardware & Software

| Item | Detail |
|---|---|
| Board | Radxa Rock 5C (RK3588, aarch64) |
| OS | Debian GNU/Linux 12 (Bookworm) |
| LIDAR | SLAMTEC C1, connected via `/dev/ttyUSB0` |
| ROS2 | Humble (built from source at `~/ros2_humble/`) |
| SLAM | slam_toolbox (built from source in `~/ros2_ws/`) |
| Odometry | rf2o_laser_odometry (laser scan-based, no encoders) |

---

## TF Frame Architecture

```
map
 └── odom           (published by slam_toolbox)
      └── base_link  (published by rf2o_laser_odometry)
           └── laser  (static transform: 0 0 0.1 above base_link)
```

---

## One-Time Setup

### 1. Serial port permissions

The LIDAR connects over USB-serial. Without this your user cannot open the port:

```bash
sudo usermod -aG dialout $USER
# Log out and back in, or run:
newgrp dialout
```

Verify:
```bash
ls -l /dev/ttyUSB0   # should show group: dialout
groups               # should include: dialout
```

### 2. Build packages from source

ROS2 is installed from source, so `apt install ros-humble-*` does not work.
All packages must be built in `~/ros2_ws/`.

```bash
cd ~/ros2_ws/src

# slam_toolbox
git clone --branch humble https://github.com/SteveMacenski/slam_toolbox.git

# rf2o_laser_odometry (laser-based odometry, no wheel encoders needed)
git clone --branch ros2 https://github.com/MAPIRlab/rf2o_laser_odometry.git

# System dependency for slam_toolbox
sudo apt install libsuitesparse-dev

# Build both
source ~/ros2_humble/install/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select slam_toolbox rf2o_laser_odometry
```

> **Note:** slam_toolbox takes 10–20 minutes to build on the RK3588.

---

## Running SLAM

### Quick start (recommended)

```bash
~/slam_ws/start_slam.sh
```

With a different serial port:
```bash
~/slam_ws/start_slam.sh serial_port:=/dev/ttyUSB1
```

This single command starts all four components in the correct order:
1. SLAMTEC C1 LIDAR node
2. Static TF publisher (`base_link` → `laser`)
3. rf2o laser odometry (`odom` → `base_link`)
4. slam_toolbox async SLAM node (delayed 2s to let odom initialize)

---

### Manual start (4 terminals)

If you need to debug individual components:

**Terminal 1 — LIDAR:**
```bash
source ~/ros2_humble/install/setup.bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/ttyUSB0
```

**Terminal 2 — Static TF (LIDAR position on robot):**
```bash
source ~/ros2_humble/install/setup.bash
ros2 run tf2_ros static_transform_publisher 0 0 0.1 0 0 0 base_link laser
```
> Adjust `0 0 0.1` (x y z in metres) to match your physical LIDAR mount height.

**Terminal 3 — Laser odometry:**
```bash
source ~/ros2_humble/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py
```

**Terminal 4 — SLAM:**
```bash
source ~/ros2_humble/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=$HOME/slam_ws/mapper_params.yaml \
  use_sim_time:=false
```

---

## Visualising in RViz2

```bash
source ~/ros2_humble/install/setup.bash && rviz2
```

RViz2 configuration:

| Setting | Value |
|---|---|
| Global Options → Fixed Frame | `map` |
| Add → LaserScan → Topic | `/scan` |
| LaserScan → Size (m) | `0.03` |
| Add → Map → Topic | `/map` |
| Add → TF | (optional, shows all frames) |

---

## Saving the Map

While SLAM is running:

```bash
~/slam_ws/save_map.sh my_room
```

Saves two files to `~/maps/`:
- `my_room.pgm` — occupancy grid image
- `my_room.yaml` — map metadata (resolution, origin)

---

## Troubleshooting

### `SL_RESULT_OPERATION_TIMEOUT` on launch

The LIDAR node cannot communicate with the device.

**Check 1 — Port permissions:**
```bash
ls -l /dev/ttyUSB0
groups | grep dialout
# Fix: sudo usermod -aG dialout $USER  (then re-login)
```

**Check 2 — Port is busy:**
```bash
sudo fuser /dev/ttyUSB0
# Kill the PID shown, then retry
```

**Check 3 — Device detected:**
```bash
dmesg | grep -i "tty\|usb" | tail -10
```

**Check 4 — Wrong port:**
```bash
ls /dev/ttyUSB*
```

---

### `Failed to compute odom pose` warnings from slam_toolbox

This appears at startup while rf2o is initialising — it is **not a real error**.
The warnings stop within a few seconds once the `odom` → `base_link` transform
is being published. Confirm SLAM is working with:

```bash
source ~/ros2_humble/install/setup.bash
ros2 topic echo /map --once | grep -E "width|height"
# Should show non-zero width and height
```

**Root cause:** slam_toolbox starts before rf2o has produced its first odometry
estimate. The combined launch file adds a 2-second delay before starting
slam_toolbox to avoid this.

---

### No scans visible in RViz2

- **Fixed Frame is wrong** — must be `map`, not the default `map` or `world`.
  If it shows a red warning, the frame does not exist yet; wait for SLAM to init.
- **LaserScan not added** — Add → By Topic → `/scan` → LaserScan.
- **Size too small** — set Size (m) to `0.03` or larger.

---

### `base_frame: base_footprint` mismatch

The default `slam_toolbox` config (`mapper_params_online_async.yaml`) sets
`base_frame: base_footprint`. This setup uses `base_link` instead.
Always use the custom config at `~/slam_ws/mapper_params.yaml`, which has the
correct value. The `start_slam.sh` script does this automatically.

---

### Verify the TF tree is complete

```bash
source ~/ros2_humble/install/setup.bash
ros2 run tf2_tools view_frames
# Opens frames_*.pdf showing the full transform chain
```

Expected output:
```
map → odom (rf2o, ~10 Hz)
odom → base_link (slam_toolbox, ~10 Hz)
base_link → laser (static)
```

---

## File Reference

```
~/slam_ws/
  start_slam.sh        Launch everything with one command
  save_map.sh          Save current map to ~/maps/
  slam_launch.py       ROS2 launch file (all 4 nodes)
  mapper_params.yaml   slam_toolbox config (base_link, real-time)
  SLAM_readme_rock5c.md  This file

~/maps/                Saved maps (created on first save)
~/ros2_humble/         ROS2 Humble (source install)
~/ros2_ws/             ROS2 workspace
  src/
    slam_toolbox/
    rf2o_laser_odometry/
    rplidar_ros/
```
