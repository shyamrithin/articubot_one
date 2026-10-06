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
#                         track       = tracking number; stays the same
#                                       while the same target stays in view
#   /thermal/rays         visualization_msgs/MarkerArray: those directions
#                         drawn as lines with a temperature label, for RViz
#   /thermal/max_c        std_msgs/Float32, hottest valid pixel per frame
#
# PROCESSING, PER FRAME
# ---------------------
#   0. The sensor is read in a separate process. Reading it is heavy pure
#      Python, and sharing one process with the colour drawing halved the
#      frame rate and caused bad reads whenever RViz was watching.
#   1. Dead pixels: anything outside the sensor's -40..300 C range is
#      replaced with the mean of its valid neighbours.
#   2. Orientation: flip_v / flip_h so the image reads like a normal camera
#      (on this robot: flip_v = true, flip_h = false).
#   3. Light temporal smoothing (exponential average) to stop flicker.
#   4. Hotspots, with hysteresis: a blob must contain at least one pixel
#      above hot_threshold_c AND min_contrast_c over the scene's median, but
#      grows into touching pixels (diagonals included) that are up to
#      grow_margin_c below those limits. So a palm joins the fingers into
#      one blob instead of one blob per finger. A band edge_margin_px wide
#      around the whole image is ignored: this sensor's edge pixels read
#      warm in every frame (phantoms that follow the robot around).
#      mask_bottom_rows can hide extra rows at the bottom, if the frame's
#      bottom edge ever sees the robot itself.
#   4b. Confirmation over time: a blob is only published once it has been
#      seen within match_angle_deg of the same direction in confirm_frames
#      of the last confirm_window frames (about half a second), and it is
#      dropped after drop_after_frames frames without a sighting. Sensor
#      noise near the threshold flickers for a frame or two and never
#      confirms; a real person stays put long enough to.
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
#   hot_threshold_c (30.0)              absolute temperature for a hotspot
#   min_contrast_c (2.0)                required rise above scene median
#                                       (30 / 2 tuned in a ~31 C room)
#   min_blob_pixels (2)                 smallest blob reported
#   grow_margin_c (1.0)                 hysteresis: how far below the
#                                       limits a blob may grow
#   confirm_frames (4) confirm_window (5)  seen in 4 of the last 5 frames
#   drop_after_frames (4)               forget after 4 frames unseen
#   match_angle_deg (8.0)               same target if this close
#   edge_margin_px (2)                  ignore this many pixels on every
#                                       edge (2: usable view ~96 x 62 deg)
#   mask_bottom_rows (0)                extra rows to ignore at the bottom
#   color_scale (8)                     image_color upscale factor
#   color_every (2)                     draw image_color every Nth frame
#   color_min_c (20.0) color_max_c (40.0)  fixed palette range
#   color_auto (true)                   true: per-frame range,
#   color_min_span_c (6.0)                but never narrower than this, so
#                                       a uniform scene stays uniform
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
import multiprocessing as mp
import queue
import sys
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


def find_blobs(frame, threshold_c, min_contrast_c, min_pixels,
               mask_bottom_rows=0, grow_margin_c=1.0, edge_margin=0):
    """Warm blobs with hysteresis. Returns dicts with centroid, peak and size.

    Seeds must clear both the absolute threshold and the contrast over the
    scene median; blobs then grow into 8-connected neighbours that clear
    both limits minus grow_margin_c.
    """
    median = float(np.median(frame))
    seed = (frame >= threshold_c) & (frame >= median + min_contrast_c)
    grow = ((frame >= threshold_c - grow_margin_c)
            & (frame >= median + min_contrast_c - grow_margin_c))
    if mask_bottom_rows > 0:
        seed[ROWS - mask_bottom_rows:, :] = False
        grow[ROWS - mask_bottom_rows:, :] = False
    if edge_margin > 0:
        inner = np.zeros_like(seed)
        inner[edge_margin:ROWS - edge_margin, edge_margin:COLS - edge_margin] = True
        seed &= inner
        grow &= inner
    seen = np.zeros_like(grow)
    floor = min(threshold_c, median + min_contrast_c) - grow_margin_c
    blobs = []
    for r0, c0 in zip(*np.nonzero(seed)):
        if seen[r0, c0]:
            continue
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < ROWS and 0 <= cc < COLS and grow[rr, cc]
                            and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        if len(cells) < min_pixels:
            continue
        rs = np.array([p[0] for p in cells], dtype=float)
        cs = np.array([p[1] for p in cells], dtype=float)
        temps = frame[rs.astype(int), cs.astype(int)]
        weights = temps - floor + 0.1  # hotter pixels pull the centre
        blobs.append({
            'row': float((rs * weights).sum() / weights.sum()),
            'col': float((cs * weights).sum() / weights.sum()),
            'peak_c': float(temps.max()),
            'pixels': len(cells),
        })
    blobs.sort(key=lambda b: b['peak_c'], reverse=True)
    return blobs


class HotspotTracker:
    """Confirms blobs that persist in roughly the same direction.

    Directions are compared as (azimuth, elevation) in degrees. Matching is
    greedy, hottest detection first, which is plenty for a handful of blobs.
    """

    def __init__(self, confirm_frames=4, window=5, drop_after=4, match_deg=8.0):
        self.confirm_frames = confirm_frames
        self.window = window
        self.drop_after = drop_after
        self.match_deg = match_deg
        self.tracks = []
        self._next_id = 0

    def update(self, detections):
        """detections: list of (azimuth_deg, elevation_deg, blob). Returns confirmed."""
        unmatched = list(self.tracks)
        for az, el, blob in detections:
            best, best_d = None, self.match_deg
            for track in unmatched:
                d = math.hypot(track['az'] - az, track['el'] - el)
                if d <= best_d:
                    best, best_d = track, d
            if best is None:
                self.tracks.append({'id': self._next_id, 'az': az, 'el': el, 'blob': blob,
                                    'hits': deque([True], maxlen=self.window),
                                    'missed': 0, 'confirmed': False})
                self._next_id += 1
                continue
            unmatched.remove(best)
            best['az'] = 0.5 * best['az'] + 0.5 * az
            best['el'] = 0.5 * best['el'] + 0.5 * el
            best['blob'] = blob
            best['hits'].append(True)
            best['missed'] = 0
        for track in unmatched:
            track['hits'].append(False)
            track['missed'] += 1
        for track in self.tracks:
            if sum(track['hits']) >= self.confirm_frames:
                track['confirmed'] = True
        self.tracks = [t for t in self.tracks if t['missed'] < self.drop_after]
        return [t for t in self.tracks if t['confirmed'] and t['missed'] == 0]


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


# --- Sensor reader (separate process) -------------------------------------------

def _sensor_process(refresh_hz, frames, stop, errors):
    """Runs in its own process: read frames, keep only the newest in the queue."""
    import adafruit_mlx90640
    import board
    import busio
    rates = {2: 'REFRESH_2_HZ', 4: 'REFRESH_4_HZ', 8: 'REFRESH_8_HZ',
             16: 'REFRESH_16_HZ', 32: 'REFRESH_32_HZ'}
    mlx = None
    buffer = [0.0] * (ROWS * COLS)
    while not stop.is_set():
        if mlx is None:
            try:
                mlx = adafruit_mlx90640.MLX90640(busio.I2C(board.SCL, board.SDA))
                mlx.refresh_rate = getattr(adafruit_mlx90640.RefreshRate,
                                           rates.get(refresh_hz, 'REFRESH_16_HZ'))
                print('[thermal sensor] camera connected', file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001  (keep retrying)
                print(f'[thermal sensor] camera not available ({exc}); retrying in 2 s',
                      file=sys.stderr, flush=True)
                time.sleep(2.0)
                continue
        try:
            mlx.getFrame(buffer)
        except (ValueError, RuntimeError, OSError):
            with errors.get_lock():
                errors.value += 1
            continue
        item = list(buffer)
        try:
            frames.put_nowait(item)
        except queue.Full:
            try:
                frames.get_nowait()  # drop the stale frame
            except queue.Empty:
                pass
            try:
                frames.put_nowait(item)
            except queue.Full:
                pass


class SensorReader:
    """Owns the sensor process; hands the newest frame to the ROS side."""

    def __init__(self, refresh_hz):
        ctx = mp.get_context('spawn')  # never fork a process that has ROS running
        self._frames = ctx.Queue(maxsize=2)
        self._stop = ctx.Event()
        self._errors = ctx.Value('i', 0)
        self._proc = ctx.Process(target=_sensor_process, daemon=True,
                                 args=(refresh_hz, self._frames, self._stop, self._errors))

    @property
    def errors(self):
        return self._errors.value

    def start(self):
        self._proc.start()

    def newest(self):
        """Newest frame since the last call, or None."""
        frame = None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
        return None if frame is None else np.array(frame, dtype=float).reshape(ROWS, COLS)

    def stop(self):
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()


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
        self.threshold = p('hot_threshold_c', 30.0).value
        self.contrast = p('min_contrast_c', 2.0).value
        self.min_pixels = p('min_blob_pixels', 2).value
        self.grow_margin = p('grow_margin_c', 1.0).value
        self.tracker = HotspotTracker(
            confirm_frames=p('confirm_frames', 4).value,
            window=p('confirm_window', 5).value,
            drop_after=p('drop_after_frames', 4).value,
            match_deg=p('match_angle_deg', 8.0).value)
        self.mask_rows = p('mask_bottom_rows', 0).value
        self.edge_margin = max(0, p('edge_margin_px', 2).value)
        self.color_scale = p('color_scale', 8).value
        self.color_every = max(1, p('color_every', 2).value)
        self.color_min_span = p('color_min_span_c', 6.0).value
        self.color_min = p('color_min_c', 20.0).value
        self.color_max = p('color_max_c', 40.0).value
        self.color_auto = p('color_auto', True).value
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
            PointField(name='track', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        self._smoothed = None
        self._frame_count = 0
        self._frame_times = deque(maxlen=40)
        self._hottest = None
        self._last_report = time.time()

        self.reader = SensorReader(refresh)
        self.reader.start()
        self.create_timer(0.02, self._tick)
        self.get_logger().info('Reading the camera in a separate process.')
        self.get_logger().info(
            f'Thermal node up: flip_v={self.flip_v} flip_h={self.flip_h}, '
            f'ignoring a {self.edge_margin}-pixel edge band, '
            f'hotspots above {self.threshold:.1f} C and {self.contrast:.1f} C over the scene.')

    def _tick(self):
        raw = self.reader.newest()
        if raw is None:
            return
        self._frame_count += 1
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
        if (self._frame_count % self.color_every == 0
                and self.pub_color.get_subscription_count() > 0):
            self._publish_color(header, frame)
        raw_blobs = find_blobs(frame, self.threshold, self.contrast,
                               self.min_pixels, self.mask_rows, self.grow_margin,
                               self.edge_margin)
        detections = []
        for blob in raw_blobs:
            dx, dy, dz = pixel_to_direction(blob['row'], blob['col'], self.hfov, self.vfov)
            detections.append((math.degrees(math.atan2(dx, dz)),
                               math.degrees(math.asin(max(-1.0, min(1.0, dy)))), blob))
        confirmed = self.tracker.update(detections)
        blobs = [t['blob'] for t in confirmed]
        self._publish_hotspots(header, blobs, [t['id'] for t in confirmed])
        self.pub_max.publish(Float32(data=float(frame.max())))
        if blobs:
            top = blobs[0]
            dx, dy, dz = pixel_to_direction(top['row'], top['col'], self.hfov, self.vfov)
            self._hottest = (top['peak_c'], math.degrees(math.atan2(dx, dz)),
                             math.degrees(math.asin(max(-1.0, min(1.0, dy)))))
        else:
            self._hottest = None

        now = time.time()
        self._frame_times.append(now)
        if now - self._last_report > 30.0 and len(self._frame_times) > 1:
            fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
            where = ''
            if self._hottest is not None:
                temp, az, el = self._hottest
                where = (f' Hottest spot {temp:.1f} C at {abs(az):.0f} deg '
                         f'{"left" if az < 0 else "right"}, {abs(el):.0f} deg '
                         f'{"up" if el < 0 else "down"}.')
            self.get_logger().info(
                f'{fps:.1f} frames/s, max {frame.max():.1f} C, {len(blobs)} hotspot(s), '
                f'{patched} dead pixel(s) patched, {self.reader.errors} bad read(s) so far.'
                + where)
            self._last_report = now

    def _publish_image(self, header, frame):
        msg = Image(header=header, height=ROWS, width=COLS, encoding='32FC1',
                    is_bigendian=int(sys.byteorder == 'big'), step=COLS * 4)
        msg.data = frame.astype(np.float32).tobytes()
        self.pub_image.publish(msg)

    def _publish_color(self, header, frame):
        if self.color_auto:
            lo, hi = float(np.percentile(frame, 2)), float(frame.max())
            if hi - lo < self.color_min_span:  # keep a uniform scene uniform
                middle = (hi + lo) / 2.0
                lo, hi = middle - self.color_min_span / 2.0, middle + self.color_min_span / 2.0
        else:
            lo, hi = self.color_min, self.color_max
        rgb = colorize(frame, self.color_scale, lo, hi)
        msg = Image(header=header, height=rgb.shape[0], width=rgb.shape[1],
                    encoding='rgb8', is_bigendian=0, step=rgb.shape[1] * 3)
        msg.data = rgb.tobytes()
        self.pub_color.publish(msg)

    def _publish_hotspots(self, header, blobs, ids):
        points, markers = [], [Marker(header=header, action=Marker.DELETEALL)]
        for i, (blob, track_id) in enumerate(zip(blobs, ids)):
            dx, dy, dz = pixel_to_direction(blob['row'], blob['col'], self.hfov, self.vfov)
            points.append((dx, dy, dz, blob['peak_c'], float(blob['pixels']), float(track_id)))

            heat = min(max((blob['peak_c'] - self.threshold) / 10.0, 0.0), 1.0)
            ray = Marker(header=header, ns='thermal_rays', id=i, type=Marker.LINE_LIST,
                         action=Marker.ADD)
            ray.scale.x = 0.01
            ray.color.r, ray.color.g, ray.color.b, ray.color.a = 1.0, 0.6 - 0.5 * heat, 0.1, 0.9
            ray.points = [Point(x=0.0, y=0.0, z=0.0),
                          Point(x=dx * self.ray_length, y=dy * self.ray_length,
                                z=dz * self.ray_length)]
            label = Marker(header=header, ns='thermal_labels', id=i, type=Marker.TEXT_VIEW_FACING,
                           action=Marker.ADD, text=f'#{track_id}  {blob["peak_c"]:.1f} C')
            label.pose.position = Point(x=dx * 0.6, y=dy * 0.6 - 0.05, z=dz * 0.6)
            label.scale.z = 0.08
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            markers += [ray, label]
        self.pub_hot.publish(point_cloud2.create_cloud(header, self._fields, points))
        self.pub_rays.publish(MarkerArray(markers=markers))

    def destroy_node(self):
        self.reader.stop()
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