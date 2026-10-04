#!/usr/bin/env python3
# =============================================================================
# File:        thermal/thermal_node.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
#              MLX90640 (110 x 75 deg) on I2C bus 1 at 0x33, bus at 400 kHz
# =============================================================================
# DESCRIPTION
# -----------
# Reads the thermal camera and publishes what the rest of the stack needs.
# Everything is stamped in thermal_link_optical (z forward, x right, y down),
# defined in description/thermal.xacro.
#
#   /thermal/image        sensor_msgs/Image, 32FC1, 32 x 24, degrees C.
#                         Raw temperatures: about 3 KB per frame, so cheap
#                         to send over wifi. This is what other nodes use.
#   /thermal/image_color  sensor_msgs/Image, rgb8, upscaled with the "iron"
#                         palette, for looking at in RViz / rqt_image_view.
#                         Only computed while something is subscribed.
#   /thermal/hotspots     sensor_msgs/PointCloud2, one point per warm blob:
#                         x, y, z  = unit direction from the camera to the
#                                    blob's centre (a ray, not a position;
#                                    the camera cannot measure distance)
#                         temperature = hottest pixel in the blob, deg C
#                         pixels      = blob size in pixels
#   /thermal/rays         visualization_msgs/MarkerArray: those directions
#                         drawn as lines with a temperature label, for RViz
#   /thermal/max_c        std_msgs/Float32, hottest valid pixel per frame
#
# PROCESSING, PER FRAME
# ---------------------
#   1. Dead pixels: anything outside the sensor's -40..300 C range is
#      replaced with the mean of its valid neighbours.
#   2. Orientation: flip_v / flip_h so the image reads like a normal camera
#      (on this robot: flip_v = true, flip_h = false).
#   3. Light temporal smoothing (exponential average) to stop flicker.
#   4. Hotspots: pixels above hot_threshold_c AND at least min_contrast_c
#      above the scene's median, grouped into 4-connected blobs. Rows listed
#      in mask_bottom_rows are ignored, for when the frame's bottom edge
#      sees the robot itself.
#   5. Direction: pixel position -> angles with a linear angle-per-pixel
#      model (hfov / 32, vfov / 24), which suits this wide lens better than
#      a pinhole model, then -> unit vector in the optical frame.
#
# PARAMETERS (ros2 run style: --ros-args -p name:=value)
# -------------------------------------------------------
#   flip_v (true) flip_h (false)        image orientation
#   hfov_deg (110.0) vfov_deg (75.0)    lens field of view
#   refresh_hz (16)                     sensor rate; two halves per image,
#                                       so 16 -> ~8 full frames per second
#   smoothing (0.5)                     0 = none, closer to 1 = smoother
#   hot_threshold_c (28.0)              absolute temperature for a hotspot
#   min_contrast_c (4.0)                required rise above scene median
#   min_blob_pixels (1)                 smallest blob reported
#   mask_bottom_rows (0)                ignore this many rows at the bottom
#   color_scale (10)                    image_color upscale factor
#   color_min_c (20.0) color_max_c (40.0)  fixed palette range
#   color_auto (false)                  true: per-frame range instead
#   ray_length_m (3.0)                  length of the RViz direction lines
#
# RUN
# ---
#   python3 ~/robot_ws/src/articubot_one/thermal/thermal_node.py
#   (source ROS first; the robot panel will start it automatically later)
#
# View in RViz: add an Image display on /thermal/image_color, and a
# MarkerArray display on /thermal/rays.
#
# Needs: pip3 install adafruit-circuitpython-mlx90640, python3-numpy
# =============================================================================

import math
import sys
import threading
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

ROWS, COLS = 24, 32
VALID_MIN_C, VALID_MAX_C = -40.0, 300.0
FRAME_ID = 'thermal_link_optical'

# "Iron" palette, 256 entries.
_STOPS = [(0.00, (0, 0, 0)), (0.20, (32, 0, 96)), (0.45, (160, 0, 120)),
          (0.65, (230, 60, 20)), (0.85, (255, 170, 0)), (1.00, (255, 255, 230))]
_t = np.linspace(0.0, 1.0, 256)
_xs = np.array([s for s, _ in _STOPS])
_cs = np.array([c for _, c in _STOPS], dtype=float)
PALETTE = np.stack([np.interp(_t, _xs, _cs[:, i]) for i in range(3)], axis=1).astype(np.uint8)


# --- Pure functions (no ROS, unit-testable) --------------------------------------

def patch_bad_pixels(frame):
    """Replace out-of-range pixels with the mean of their valid neighbours."""
    bad = ~np.isfinite(frame) | (frame < VALID_MIN_C) | (frame > VALID_MAX_C)
    count = int(bad.sum())
    if count == 0:
        return frame, 0
    fixed = frame.copy()
    fixed[bad] = np.nan
    h, w = frame.shape
    for r, c in zip(*np.nonzero(bad)):
        block = fixed[max(r - 1, 0):min(r + 2, h), max(c - 1, 0):min(c + 2, w)]
        neighbours = block[np.isfinite(block)]
        fixed[r, c] = neighbours.mean() if neighbours.size else np.nanmedian(fixed)
    return fixed, count


def find_blobs(frame, threshold_c, min_contrast_c, min_pixels, mask_bottom_rows=0):
    """4-connected warm blobs. Returns dicts with centroid, peak and size."""
    hot = (frame >= threshold_c) & (frame >= np.median(frame) + min_contrast_c)
    if mask_bottom_rows > 0:
        hot[ROWS - mask_bottom_rows:, :] = False
    seen = np.zeros_like(hot)
    blobs = []
    for r0, c0 in zip(*np.nonzero(hot)):
        if seen[r0, c0]:
            continue
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ROWS and 0 <= cc < COLS and hot[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(cells) < min_pixels:
            continue
        rs = np.array([p[0] for p in cells], dtype=float)
        cs = np.array([p[1] for p in cells], dtype=float)
        temps = frame[rs.astype(int), cs.astype(int)]
        weights = temps - threshold_c + 0.1  # hotter pixels pull the centre
        blobs.append({
            'row': float((rs * weights).sum() / weights.sum()),
            'col': float((cs * weights).sum() / weights.sum()),
            'peak_c': float(temps.max()),
            'pixels': len(cells),
        })
    blobs.sort(key=lambda b: b['peak_c'], reverse=True)
    return blobs


def pixel_to_direction(row, col, hfov_deg, vfov_deg):
    """Pixel centre -> unit vector in the optical frame (x right, y down, z fwd).

    Linear angle-per-pixel model: the angle from the optical axis grows in
    proportion to the distance from the image centre.
    """
    azimuth = math.radians(((col + 0.5) / COLS - 0.5) * hfov_deg)    # + right
    elevation = math.radians(((row + 0.5) / ROWS - 0.5) * vfov_deg)  # + down
    return (math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth))


def upscale(img, factor):
    """Bilinear upscale of a small 2-D array."""
    h, w = img.shape
    ys = np.linspace(0, h - 1, h * factor)
    xs = np.linspace(0, w - 1, w * factor)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bottom = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def colorize(frame, factor, lo, hi):
    big = upscale(frame, factor)
    span = max(hi - lo, 0.5)
    index = np.clip((big - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return PALETTE[index]


# --- Sensor reader ---------------------------------------------------------------

class SensorReader(threading.Thread):
    """Pulls frames off the MLX90640 continuously; keeps the newest one."""

    def __init__(self, refresh_hz, logger):
        super().__init__(daemon=True)
        self._refresh_hz = refresh_hz
        self._log = logger
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self.errors = 0
        self.stop_requested = False

    def _open(self):
        import adafruit_mlx90640
        import board
        import busio
        rates = {2: 'REFRESH_2_HZ', 4: 'REFRESH_4_HZ', 8: 'REFRESH_8_HZ',
                 16: 'REFRESH_16_HZ', 32: 'REFRESH_32_HZ'}
        mlx = adafruit_mlx90640.MLX90640(busio.I2C(board.SCL, board.SDA))
        mlx.refresh_rate = getattr(adafruit_mlx90640.RefreshRate,
                                   rates.get(self._refresh_hz, 'REFRESH_16_HZ'))
        return mlx

    def run(self):
        mlx = None
        buffer = [0.0] * (ROWS * COLS)
        while not self.stop_requested:
            if mlx is None:
                try:
                    mlx = self._open()
                    self._log.info('Thermal camera connected.')
                except Exception as exc:  # noqa: BLE001  (keep retrying)
                    self._log.error(f'Thermal camera not available ({exc}); retrying in 2 s.')
                    time.sleep(2.0)
                    continue
            try:
                mlx.getFrame(buffer)
            except (ValueError, RuntimeError, OSError):
                self.errors += 1  # occasional bad read; take the next one
                continue
            frame = np.array(buffer, dtype=float).reshape(ROWS, COLS)
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self):
        with self._lock:
            return self._seq, self._frame


# --- ROS node -------------------------------------------------------------------

class ThermalNode(Node):

    def __init__(self):
        super().__init__('thermal')
        p = self.declare_parameter
        self.flip_v = p('flip_v', True).value
        self.flip_h = p('flip_h', False).value
        self.hfov = p('hfov_deg', 110.0).value
        self.vfov = p('vfov_deg', 75.0).value
        refresh = p('refresh_hz', 16).value
        self.alpha = min(max(p('smoothing', 0.5).value, 0.0), 0.95)
        self.threshold = p('hot_threshold_c', 28.0).value
        self.contrast = p('min_contrast_c', 4.0).value
        self.min_pixels = p('min_blob_pixels', 1).value
        self.mask_rows = p('mask_bottom_rows', 0).value
        self.color_scale = p('color_scale', 10).value
        self.color_min = p('color_min_c', 20.0).value
        self.color_max = p('color_max_c', 40.0).value
        self.color_auto = p('color_auto', False).value
        self.ray_length = p('ray_length_m', 3.0).value

        self.pub_image = self.create_publisher(Image, '/thermal/image', qos_profile_sensor_data)
        self.pub_color = self.create_publisher(Image, '/thermal/image_color', qos_profile_sensor_data)
        self.pub_hot = self.create_publisher(PointCloud2, '/thermal/hotspots', 10)
        self.pub_rays = self.create_publisher(MarkerArray, '/thermal/rays', 10)
        self.pub_max = self.create_publisher(Float32, '/thermal/max_c', 10)

        self._fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='temperature', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='pixels', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        self._smoothed = None
        self._last_seq = 0
        self._frame_times = deque(maxlen=40)
        self._last_report = time.time()

        self.reader = SensorReader(refresh, self.get_logger())
        self.reader.start()
        self.create_timer(0.02, self._tick)
        self.get_logger().info(
            f'Thermal node up: flip_v={self.flip_v} flip_h={self.flip_h}, '
            f'hotspots above {self.threshold:.1f} C and {self.contrast:.1f} C over the scene.')

    def _tick(self):
        seq, raw = self.reader.latest()
        if raw is None or seq == self._last_seq:
            return
        self._last_seq = seq
        frame, patched = patch_bad_pixels(raw)
        if self.flip_v:
            frame = frame[::-1, :]
        if self.flip_h:
            frame = frame[:, ::-1]
        if self._smoothed is None or self.alpha == 0.0:
            self._smoothed = frame
        else:
            self._smoothed = self.alpha * self._smoothed + (1.0 - self.alpha) * frame
        frame = np.ascontiguousarray(self._smoothed)

        header = Header(frame_id=FRAME_ID, stamp=self.get_clock().now().to_msg())
        self._publish_image(header, frame)
        if self.pub_color.get_subscription_count() > 0:
            self._publish_color(header, frame)
        blobs = find_blobs(frame, self.threshold, self.contrast,
                           self.min_pixels, self.mask_rows)
        self._publish_hotspots(header, blobs)
        self.pub_max.publish(Float32(data=float(frame.max())))

        now = time.time()
        self._frame_times.append(now)
        if now - self._last_report > 30.0 and len(self._frame_times) > 1:
            fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
            self.get_logger().info(
                f'{fps:.1f} frames/s, max {frame.max():.1f} C, {len(blobs)} hotspot(s), '
                f'{patched} dead pixel(s) patched, {self.reader.errors} bad read(s) so far.')
            self._last_report = now

    def _publish_image(self, header, frame):
        msg = Image(header=header, height=ROWS, width=COLS, encoding='32FC1',
                    is_bigendian=int(sys.byteorder == 'big'), step=COLS * 4)
        msg.data = frame.astype(np.float32).tobytes()
        self.pub_image.publish(msg)

    def _publish_color(self, header, frame):
        if self.color_auto:
            lo, hi = float(np.percentile(frame, 2)), float(frame.max())
        else:
            lo, hi = self.color_min, self.color_max
        rgb = colorize(frame, self.color_scale, lo, hi)
        msg = Image(header=header, height=rgb.shape[0], width=rgb.shape[1],
                    encoding='rgb8', is_bigendian=0, step=rgb.shape[1] * 3)
        msg.data = rgb.tobytes()
        self.pub_color.publish(msg)

    def _publish_hotspots(self, header, blobs):
        points, markers = [], [Marker(header=header, action=Marker.DELETEALL)]
        for i, blob in enumerate(blobs):
            dx, dy, dz = pixel_to_direction(blob['row'], blob['col'], self.hfov, self.vfov)
            points.append((dx, dy, dz, blob['peak_c'], float(blob['pixels'])))

            heat = min(max((blob['peak_c'] - self.threshold) / 10.0, 0.0), 1.0)
            ray = Marker(header=header, ns='thermal_rays', id=i, type=Marker.LINE_LIST,
                         action=Marker.ADD)
            ray.scale.x = 0.01
            ray.color.r, ray.color.g, ray.color.b, ray.color.a = 1.0, 0.6 - 0.5 * heat, 0.1, 0.9
            ray.points = [Point(x=0.0, y=0.0, z=0.0),
                          Point(x=dx * self.ray_length, y=dy * self.ray_length,
                                z=dz * self.ray_length)]
            label = Marker(header=header, ns='thermal_labels', id=i, type=Marker.TEXT_VIEW_FACING,
                           action=Marker.ADD, text=f'{blob["peak_c"]:.1f} C')
            label.pose.position = Point(x=dx * 0.6, y=dy * 0.6 - 0.05, z=dz * 0.6)
            label.scale.z = 0.06
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            markers += [ray, label]
        self.pub_hot.publish(point_cloud2.create_cloud(header, self._fields, points))
        self.pub_rays.publish(MarkerArray(markers=markers))

    def destroy_node(self):
        self.reader.stop_requested = True
        return super().destroy_node()


def main():
    rclpy.init()
    node = ThermalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
