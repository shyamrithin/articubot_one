#!/usr/bin/env python3
# =============================================================================
# File:        vision/vision_node.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
#              Logitech C270 on /dev/video0, MLX90640 via thermal_node.py
# =============================================================================
# DESCRIPTION
# -----------
# The robot's vision node. Reads the C270 webcam ON THE PI and fuses it with
# the thermal camera, sending only small compressed previews over Wi-Fi
# (the old camera.launch.py streamed raw frames and choked the hotspot).
# The webcam is opened once here and will be shared by the cube detector.
#
# CAMERA
#   MJPG 640x480 at camera_fps (10). A reader thread keeps only the newest
#   frame, so processing never lags behind. Reconnects if unplugged.
#
# FUSION (modes, switchable live with `ros2 param set /vision mode <name>`)
#   blend   webcam picture; warm areas glow in thermal colours, cool areas
#           stay as the plain webcam image (alpha scales with temperature)
#   edges   thermal colours with the webcam's outlines drawn on top
#           (FLIR "MSX" style: shapes from the webcam, heat from thermal)
#   side    full thermal view (left) next to the webcam (right); the webcam
#           only covers the middle of the thermal view, marked by a box
#   The hottest point in view is marked with its temperature.
#
# ALIGNMENT
#   Each output pixel is a ray from the webcam. It is extended to align_dist_m
#   (the distance the overlay is exact at), moved by the webcam -> thermal
#   lens offset, and looked up in the thermal image with the same
#   angle-per-pixel model thermal_node.py uses. Because the lenses are a few
#   cm apart, objects much nearer or farther than align_dist_m show a small
#   offset (parallax).
#   Measured on the robot: webcam 2.3 cm ABOVE and 2.0 cm AHEAD of the
#   thermal lens, centred left-right.
#   Fine-tune live, watching a mug of hot water ~1 m away:
#     ros2 param set /vision shift_x_deg 1.5     (+ moves heat to the right)
#     ros2 param set /vision shift_y_deg -1.0    (+ moves heat down)
#     ros2 param set /vision web_hfov_deg 46.0   (bigger = heat shrinks)
#   then save so it is loaded next time:
#     ros2 service call /vision/save_alignment std_srvs/srv/Trigger
#   (writes ~/.ros/vision_alignment.yaml)
#
# OUTPUTS (JPEG, only produced while someone is subscribed)
#   /vision/fused/compressed    sensor_msgs/CompressedImage
#   /camera/image/compressed    sensor_msgs/CompressedImage (plain webcam)
#
# INPUT
#   /thermal/image   32FC1 temperatures from thermal_node.py, already
#                    oriented like a normal camera
#
# VIEWING ON THE LAPTOP
#   rqt_image_view, pick /vision/fused/compressed (simplest), or RViz Image
#   display on /vision/fused with Transport Hint "compressed" (needs
#   ros-humble-compressed-image-transport on the laptop).
#
# RUN
#   python3 ~/robot_ws/src/articubot_one/vision/vision_node.py
#   (thermal_node.py must be running for the thermal half)
# =============================================================================

import math
import os
import threading
import time

import cv2
import numpy as np
import yaml

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger

THERMAL_ROWS, THERMAL_COLS = 24, 32
ALIGN_FILE = os.path.expanduser('~/.ros/vision_alignment.yaml')
ALIGN_KEYS = ('web_hfov_deg', 'web_vfov_deg', 'web_pitch_deg', 'offset_up_m',
              'offset_forward_m', 'offset_right_m', 'align_dist_m',
              'shift_x_deg', 'shift_y_deg')
MODES = ('blend', 'edges', 'side')


# --- Geometry (pure functions, unit-testable) --------------------------------------

def build_thermal_lookup(out_w, out_h, web_hfov, web_vfov, web_pitch, offset_up,
                         offset_forward, offset_right, dist, shift_x, shift_y,
                         th_hfov, th_vfov):
    """For each output (webcam) pixel, the thermal (col, row) it looks at.

    Angles in degrees, offsets in metres: the webcam's position relative to
    the thermal lens (up, forward, right). Returns float32 maps for cv2.remap.
    """
    fx = (out_w / 2.0) / math.tan(math.radians(web_hfov) / 2.0)
    fy = (out_h / 2.0) / math.tan(math.radians(web_vfov) / 2.0)
    u, v = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    # Webcam ray in optical coordinates: x right, y down, z forward.
    x = (u - out_w / 2.0) / fx
    y = (v - out_h / 2.0) / fy
    z = np.ones_like(x)
    # Webcam pitch relative to the thermal camera (+ = webcam tilted down).
    p = math.radians(web_pitch)
    y, z = y * math.cos(p) + z * math.sin(p), -y * math.sin(p) + z * math.cos(p)
    # Point at the alignment distance, then into the thermal camera's frame.
    scale = dist / z
    px = x * scale + offset_right
    py = y * scale - offset_up        # webcam above thermal = smaller y
    pz = z * scale + offset_forward
    rng = np.sqrt(px * px + py * py + pz * pz)
    az = np.degrees(np.arctan2(px, pz)) - shift_x
    el = np.degrees(np.arcsin(np.clip(py / rng, -1.0, 1.0))) - shift_y
    col = (az / th_hfov + 0.5) * THERMAL_COLS - 0.5
    row = (el / th_vfov + 0.5) * THERMAL_ROWS - 0.5
    return col.astype(np.float32), row.astype(np.float32)


def webcam_box_in_thermal(lookup_col, lookup_row):
    """Corners of the webcam's view in thermal pixel coordinates (for 'side')."""
    h, w = lookup_col.shape
    pts = [(lookup_col[0, 0], lookup_row[0, 0]), (lookup_col[0, w - 1], lookup_row[0, w - 1]),
           (lookup_col[h - 1, w - 1], lookup_row[h - 1, w - 1]),
           (lookup_col[h - 1, 0], lookup_row[h - 1, 0])]
    return [(float(c), float(r)) for c, r in pts]


def temperature_range(temps, min_span):
    lo, hi = float(np.percentile(temps, 2)), float(np.max(temps))
    if hi - lo < min_span:
        mid = (hi + lo) / 2.0
        lo, hi = mid - min_span / 2.0, mid + min_span / 2.0
    return lo, hi


def colorize(temps, lo, hi):
    norm = np.clip((temps - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), norm


# --- Camera reader ------------------------------------------------------------------

class CameraReader(threading.Thread):
    """Grabs MJPG frames continuously; keeps only the newest."""

    def __init__(self, device, width, height, fps, logger):
        super().__init__(daemon=True)
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.log = logger
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self.stop_requested = False

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def run(self):
        cap = None
        while not self.stop_requested:
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.log.error(f'Webcam {self.device} not available; retrying in 2 s.')
                    time.sleep(2.0)
                    continue
                self.log.info(f'Webcam {self.device} open: MJPG '
                              f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                              f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at {self.fps} fps.')
            ok, frame = cap.read()
            if not ok or frame is None:
                self.log.warning('Webcam read failed; reopening.')
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame, self._stamp = frame, time.time()
        if cap is not None:
            cap.release()

    def latest(self):
        with self._lock:
            return self._frame, self._stamp


# --- ROS node -------------------------------------------------------------------------

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision')
        p = self.declare_parameter
        self.device = p('device', '/dev/video0').value
        cam_w, cam_h = p('camera_width', 640).value, p('camera_height', 480).value
        cam_fps = p('camera_fps', 10).value
        self.out_w, self.out_h = p('output_width', 480).value, p('output_height', 360).value
        self.publish_hz = p('publish_hz', 5.0).value
        self.jpeg_quality = p('jpeg_quality', 70).value
        p('mode', 'blend')
        p('blend_alpha', 0.75)
        p('min_span_c', 6.0)
        p('thermal_hfov_deg', 110.0)
        p('thermal_vfov_deg', 75.0)
        # Alignment (C270 at 640x480 is roughly 46 x 35 degrees; tune live).
        defaults = {'web_hfov_deg': 46.0, 'web_vfov_deg': 35.0, 'web_pitch_deg': 0.0,
                    'offset_up_m': 0.023, 'offset_forward_m': 0.020, 'offset_right_m': 0.0,
                    'align_dist_m': 1.0, 'shift_x_deg': 0.0, 'shift_y_deg': 0.0}
        saved = self._load_alignment()
        for key, value in defaults.items():
            p(key, float(saved.get(key, value)))
        if saved:
            self.get_logger().info(f'Loaded alignment from {ALIGN_FILE}.')

        self._maps = None
        self._maps_lock = threading.Lock()
        self._rebuild_maps()
        self.add_on_set_parameters_callback(self._on_params)

        self._thermal = None
        self._thermal_time = 0.0
        self.create_subscription(Image, '/thermal/image', self._on_thermal,
                                 qos_profile_sensor_data)
        self.pub_fused = self.create_publisher(CompressedImage, '/vision/fused/compressed',
                                               qos_profile_sensor_data)
        self.pub_cam = self.create_publisher(CompressedImage, '/camera/image/compressed',
                                             qos_profile_sensor_data)
        self.create_service(Trigger, '/vision/save_alignment', self._srv_save)

        self.camera = CameraReader(self.device, cam_w, cam_h, cam_fps, self.get_logger())
        self.camera.start()
        self.create_timer(1.0 / max(0.5, self.publish_hz), self._tick)
        self.get_logger().info(f'Vision node up, mode "{self.get_parameter("mode").value}". '
                               'View /vision/fused/compressed in rqt_image_view.')

    # -- parameters -----------------------------------------------------------

    def _param(self, name):
        return self.get_parameter(name).value

    def _rebuild_maps(self, overrides=None):
        vals = {k: self._param(k) for k in ALIGN_KEYS + ('thermal_hfov_deg', 'thermal_vfov_deg')}
        vals.update(overrides or {})
        col, row = build_thermal_lookup(
            self.out_w, self.out_h, vals['web_hfov_deg'], vals['web_vfov_deg'],
            vals['web_pitch_deg'], vals['offset_up_m'], vals['offset_forward_m'],
            vals['offset_right_m'], max(0.1, vals['align_dist_m']),
            vals['shift_x_deg'], vals['shift_y_deg'],
            vals['thermal_hfov_deg'], vals['thermal_vfov_deg'])
        with self._maps_lock:
            self._maps = (col, row, webcam_box_in_thermal(col, row))

    def _on_params(self, params):
        changed = {}
        for prm in params:
            if prm.name == 'mode' and prm.value not in MODES:
                return SetParametersResult(successful=False,
                                           reason=f'mode must be one of {", ".join(MODES)}')
            if prm.name in ALIGN_KEYS + ('thermal_hfov_deg', 'thermal_vfov_deg'):
                changed[prm.name] = float(prm.value)
        if changed:
            self._rebuild_maps(changed)
        return SetParametersResult(successful=True)

    def _load_alignment(self):
        try:
            with open(ALIGN_FILE) as handle:
                data = yaml.safe_load(handle) or {}
            return {k: float(v) for k, v in data.items() if k in ALIGN_KEYS}
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            return {}

    def _srv_save(self, _request, response):
        data = {k: float(self._param(k)) for k in ALIGN_KEYS}
        try:
            os.makedirs(os.path.dirname(ALIGN_FILE), exist_ok=True)
            with open(ALIGN_FILE, 'w') as handle:
                yaml.safe_dump(data, handle)
            response.success, response.message = True, f'Alignment saved to {ALIGN_FILE}'
        except OSError as exc:
            response.success, response.message = False, f'Could not save: {exc}'
        return response

    # -- inputs ---------------------------------------------------------------

    def _on_thermal(self, msg):
        if msg.encoding != '32FC1' or msg.width != THERMAL_COLS or msg.height != THERMAL_ROWS:
            return
        dtype = '>f4' if msg.is_bigendian else '<f4'
        self._thermal = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(
            THERMAL_ROWS, THERMAL_COLS).astype(np.float32)
        self._thermal_time = time.time()

    # -- rendering --------------------------------------------------------------

    def _encode(self, image):
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
        if not ok:
            return None
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link_optical'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        return msg

    def render(self, frame, thermal):
        """Compose the fused image (BGR, out_w x out_h)."""
        mode = self._param('mode')
        web = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        if thermal is None:
            cv2.putText(web, 'No thermal data - is thermal_node.py running?', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return web
        with self._maps_lock:
            col, row, box = self._maps
        lo, hi = temperature_range(thermal, self._param('min_span_c'))

        if mode == 'side':
            big = cv2.resize(thermal, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)
            color, _ = colorize(big, lo, hi)
            sx, sy = self.out_w / THERMAL_COLS, self.out_h / THERMAL_ROWS
            poly = np.array([[(c + 0.5) * sx, (r + 0.5) * sy] for c, r in box], np.int32)
            cv2.polylines(color, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
            self._mark_hottest(color, thermal, lambda c, r: ((c + 0.5) * sx, (r + 0.5) * sy))
            return np.hstack([color, web])

        temps = cv2.remap(thermal, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        color, norm = colorize(temps, lo, hi)
        if mode == 'edges':
            gray = cv2.cvtColor(web, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110)
            out = color.copy()
            out[edges > 0] = (255, 255, 255)
        else:  # blend: warmer = more thermal colour
            alpha = (np.clip((norm - 0.25) / 0.75, 0.0, 1.0) *
                     float(self._param('blend_alpha')))[..., None]
            out = (web * (1.0 - alpha) + color * alpha).astype(np.uint8)
        self._mark_hottest_mapped(out, temps)
        return out

    @staticmethod
    def _label(img, x, y, temp):
        x, y = int(x), int(y)
        cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        text = f'{temp:.1f} C'
        tx = x + 10 if x < img.shape[1] - 80 else x - 80
        cv2.putText(img, text, (tx, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (tx, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _mark_hottest(self, img, thermal, to_px):
        r, c = np.unravel_index(int(np.argmax(thermal)), thermal.shape)
        x, y = to_px(c, r)
        self._label(img, x, y, float(thermal[r, c]))

    def _mark_hottest_mapped(self, img, temps):
        y, x = np.unravel_index(int(np.argmax(temps)), temps.shape)
        self._label(img, x, y, float(temps[y, x]))

    def _tick(self):
        want_fused = self.pub_fused.get_subscription_count() > 0
        want_cam = self.pub_cam.get_subscription_count() > 0
        if not (want_fused or want_cam):
            return
        frame, stamp = self.camera.latest()
        if frame is None or time.time() - stamp > 2.0:
            return
        if want_cam:
            msg = self._encode(cv2.resize(frame, (self.out_w, self.out_h),
                                          interpolation=cv2.INTER_AREA))
            if msg is not None:
                self.pub_cam.publish(msg)
        if want_fused:
            thermal = self._thermal if time.time() - self._thermal_time < 2.0 else None
            msg = self._encode(self.render(frame, thermal))
            if msg is not None:
                self.pub_fused.publish(msg)

    def destroy_node(self):
        self.camera.stop_requested = True
        return super().destroy_node()


def main():
    rclpy.init()
    node = VisionNode()
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