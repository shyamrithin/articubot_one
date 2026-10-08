#!/usr/bin/env python3
# =============================================================================
# File:        thermal/victim_mapper.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Turns the thermal node's hotspot DIRECTIONS into victim POSITIONS on the
# map. The thermal camera can't measure distance, so for each confirmed
# hotspot this estimates it up to three ways and fuses them:
#
#   Lidar         the nearest lidar return lying in the hotspot's direction,
#                 seen from the camera (so the camera-lidar offset is handled),
#                 pushed lidar_surface_offset_m further along the ray so it
#                 marks the body's centre, not its front surface. Accurate when
#                 the warm spot reaches the lidar's scan plane (18 cm here).
#                 Rejected when the ray hits the floor before the lidar return,
#                 and given a large uncertainty when the warm spot lies BELOW
#                 the scan plane at that range: then the lidar beam passes
#                 over the warm thing and is probably seeing what's behind it,
#                 so triangulation has to decide.
#   Floor         where a downward ray meets the floor, within
#                 floor_max_range_m: with the camera 9.5 cm up, a one-degree
#                 error moves the estimate a long way at distance.
#
# CONSISTENCY: a single noisy ray can fool any per-frame check (e.g. a ray
# tilted up by noise makes the wall behind a low target look like a tall
# person). So for each victim, the floor method only counts if it applied
# in at least 60% of recent sightings, and the lidar is only trusted if it
# looked trustworthy in 60% of them; otherwise its readings are kept with a
# large uncertainty and triangulation decides.
#   Triangulation as the robot moves, sightings of the same target from
#                 different places are lines on the map; their least-squares
#                 crossing point needs no assumption about height. Used once
#                 the sightings span tri_min_baseline_m and tri_min_spread_deg.
#                 When available, direct estimates that disagree with it are
#                 dropped, which is what catches lidar looking past a target.
#
# Estimates are combined weighted by 1 / sigma^2, each sigma growing with
# range in the way that method's error does.
#
# Victims are kept apart by position (assoc_radius_m), by the thermal node's
# track number for the same target, and by whether a ray passes close to a
# known victim. Victims that drift within merge_radius_m are merged.
#
# Temperature sorts what was found: a median above heat_source_c is a heat
# source (lamp, charger, motor), not a person. People read ~30-37 C.
#
# INPUTS
# ------
#   /thermal/hotspots   PointCloud2 (x, y, z unit ray, temperature, pixels,
#                       track) in thermal_link_optical
#   /scan               LaserScan
#   TF                  map -> thermal_link_optical, map -> laser_frame
#                       (needs SLAM or AMCL running, i.e. a 'map' frame)
#
# MOVING TARGETS
# --------------
# Victims are assumed to stay put, so every reading is averaged into one
# position. If move_confirm_frames confident readings in a row land more
# than max(move_min_m, 3 sigma) from that position, the target has moved:
# its old evidence is discarded and it is re-located from the new readings,
# keeping the same victim number.
#
# CAMERA-FIXED PHANTOMS
# ---------------------
# A faulty sensor pixel reads warm in every frame and stays at the same
# place in the IMAGE however the robot moves, so it would be projected onto
# a different wall every time the robot turns. Real objects stay put in the
# ROOM. So when one thermal track keeps the same camera bearing (within
# fixed_bearing_tol_deg) while the camera turns by fixed_min_turn_deg or
# more, that bearing is learned as faulty: hotspots there are ignored from
# then on, victims built from it are removed, and the list is saved to
# ~/.ros/thermal_faulty_directions.yaml so it is remembered next run.
# (The thermal node also ignores a band around the image edge, where this
# sensor's known phantoms are.)
#
# OUTPUTS
# -------
#   /victims/markers    MarkerArray. Bright, larger, "seeing now" = in view
#                       right now (seen within active_window_s); dimmed,
#                       "last seen N s ago" = remembered. Per victim:
#                       sphere (red = person, orange =
#                       heat source), translucent disc = 2 sigma uncertainty,
#                       label: number, temperature, confidence, sightings
#   /victims            PoseArray of located people (not heat sources)
#   /victims/active     PoseArray of people being seen right now
#   /victims/save       std_srvs/Trigger: write ~/maps/victims_<time>.yaml
#   /victims/clear      std_srvs/Trigger: forget everything
#   Auto-saved every 30 s to ~/.ros/victims_autosave.yaml
#
# FRAMES NOTE
# -----------
# In this robot's URDF base_footprint sits on the wheel axle, 3.25 cm above
# the floor, not on it. The map frame's z = 0 follows base_footprint, so the
# floor is at z = -0.0325 in the map frame: floor_z_m.
#
# Frames are skipped while the robot turns faster than max_turn_rate_rad_s,
# when TF and the camera image are most likely to be out of step.
#
# RUN (with a mode running that provides the map frame, and the thermal node)
# ---------------------------------------------------------------------------
#   python3 ~/robot_ws/src/articubot_one/thermal/victim_mapper.py
# RViz: Fixed Frame map, add MarkerArray /victims/markers
# =============================================================================

import math
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import yaml

import rclpy
import tf2_ros
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

MAP_DIR = os.path.expanduser('~/maps')
AUTOSAVE = os.path.expanduser('~/.ros/victims_autosave.yaml')


# --- Geometry (pure functions, unit-testable) --------------------------------------

def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def floor_estimate(origin, direction, floor_z, max_range, min_depression, sigma_angle):
    """Where a downward ray meets the floor -> (x, y, sigma, range) or None."""
    ox, oy, oz = origin
    dx, dy, dz = direction
    h = math.hypot(dx, dy)
    depression = -math.asin(max(-1.0, min(1.0, dz)))
    if h < 1e-6 or depression < min_depression:
        return None
    t = (floor_z - oz) / dz
    if t <= 0:
        return None
    rng = t * h
    if rng > max_range:
        return None
    height = oz - floor_z
    sigma_r = rng * rng / height * sigma_angle + 0.03   # dr/d(angle) ~ r^2 / H
    return (ox + t * dx, oy + t * dy, math.hypot(sigma_r, rng * sigma_angle), rng)


def floor_range_upper(origin, direction, floor_z, sigma_angle):
    """Furthest the ray could plausibly meet the floor (inf if it doesn't)."""
    _, _, oz = origin
    dx, dy, dz = direction
    depression = -math.asin(max(-1.0, min(1.0, dz)))
    shallowest = depression - sigma_angle
    if shallowest <= 1e-3:
        return math.inf
    return (oz - floor_z) / math.tan(shallowest)


def lidar_estimate(origin, direction, scan_xy, window, sigma_angle, floor_upper,
                   lidar_z=None, max_height_above=1.5, surface_offset=0.10):
    """Nearest lidar return along the ray's heading.

    Returns (x, y, sigma, range, trusted, ambiguous_sigma) or None. 'trusted'
    is a per-frame judgement; the Victim only believes it if it holds in most
    sightings, because a single noisy ray can look trustworthy by chance.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    h = math.hypot(dx, dy)
    if h < 1e-6 or scan_xy is None or len(scan_xy) == 0:
        return None
    heading = math.atan2(dy, dx)
    rel = scan_xy - np.array([ox, oy])
    ranges = np.hypot(rel[:, 0], rel[:, 1])
    bearings = np.arctan2(rel[:, 1], rel[:, 0])
    off = np.abs((bearings - heading + np.pi) % (2 * np.pi) - np.pi)
    candidates = np.where((off <= window) & (ranges > 0.12), ranges, np.inf)
    i = int(np.argmin(candidates))
    rng = float(candidates[i])
    if not math.isfinite(rng):
        return None
    if rng > floor_upper + 0.10:
        return None  # the ray reaches the floor first: lidar sees past a low target
    warm_z = oz + rng * dz / h  # height of the warm spot if it were at that range
    if lidar_z is not None and warm_z > lidar_z + max_height_above:
        return None  # ray far above the lidar plane there; not the same object
    sigma = math.hypot(0.05 + 0.02 * rng, rng * sigma_angle)
    ambiguous_sigma = max(sigma, 0.4 * rng)
    # Trusted only if the warm spot is confidently at or above the scan plane,
    # allowing for two sigma of bearing error in its height at this range;
    # otherwise the beam may pass over the warm spot and see what's behind.
    trusted = lidar_z is None or warm_z - 2.0 * rng * sigma_angle >= lidar_z - 0.03
    rng += surface_offset
    t = rng / h
    return (ox + t * dx, oy + t * dy, sigma if trusted else ambiguous_sigma, rng,
            trusted, ambiguous_sigma)


def triangulate(rays, min_spread, min_baseline):
    """Least-squares crossing point of 2-D rays [(ox, oy, heading)] -> (x, y, sigma)."""
    if len(rays) < 3:
        return None
    arr = np.array(rays, dtype=float)
    origins, headings = arr[:, :2], arr[:, 2]
    mean = math.atan2(np.sin(headings).mean(), np.cos(headings).mean())
    deviation = (headings - mean + np.pi) % (2 * np.pi) - np.pi
    if deviation.max() - deviation.min() < min_spread:
        return None
    gaps = origins[:, None, :] - origins[None, :, :]
    if np.sqrt((gaps ** 2).sum(-1)).max() < min_baseline:
        return None
    normals = np.stack([-np.sin(headings), np.cos(headings)], axis=1)
    A = (normals[:, :, None] * normals[:, None, :]).sum(0)
    if np.linalg.cond(A) > 1e4:
        return None
    b = (normals * (normals * origins).sum(1, keepdims=True)).sum(0)
    p = np.linalg.solve(A, b)
    ahead = ((p - origins) * np.stack([np.cos(headings), np.sin(headings)], 1)).sum(1)
    if (ahead <= 0).mean() > 0.2:
        return None  # crossing point behind the camera: not a real target
    residual = (normals * (p - origins)).sum(1)
    rms = max(0.05, float(np.sqrt((residual ** 2).mean())))
    sigma = rms * math.sqrt(float(np.trace(np.linalg.inv(A))) * len(rays) / 2.0)
    return float(p[0]), float(p[1]), max(0.05, sigma)


# --- Victim bookkeeping ----------------------------------------------------------

class Victim:

    def __init__(self, vid, now):
        self.id = vid
        self.x = self.y = None
        self.sigma = math.inf
        self.direct = deque(maxlen=40)     # dicts: x, y, sigma, method, ...
        self.history = deque(maxlen=20)    # per sighting: (ray_points_down, lidar_trusted)
        self.rays = deque(maxlen=80)       # (ox, oy, heading)
        self.temps = deque(maxlen=60)
        self.tracks = deque(maxlen=8)
        self.sightings = 0
        self.first_seen = self.last_seen = now
        self.triangulated = False
        self.disagree = 0   # consecutive confident readings far from the estimate

    @property
    def located(self):
        return self.x is not None

    def kind(self, heat_source_c):
        return 'heat_source' if self.temps and float(np.median(self.temps)) > heat_source_c else 'person'

    def confidence(self):
        if self.sightings >= 15 and self.sigma < 0.25:
            return 'high'
        if self.sightings >= 6 and self.sigma < 0.5:
            return 'medium'
        return 'low'

    def add(self, ray, floor, lidar, temp, track, now, points_down=None):
        ox, oy, heading = ray
        if not self.rays or math.hypot(ox - self.rays[-1][0], oy - self.rays[-1][1]) >= 0.05:
            self.rays.append(ray)   # only keep rays from new viewpoints
        if floor is not None:
            self.direct.append({'x': floor[0], 'y': floor[1], 'sigma': floor[2],
                                'method': 'floor'})
        if lidar is not None:
            self.direct.append({'x': lidar[0], 'y': lidar[1], 'sigma': lidar[2],
                                'method': 'lidar', 'trusted': lidar[4],
                                'ambiguous_sigma': lidar[5]})
        down = (floor is not None) if points_down is None else points_down
        self.history.append((down, lidar is not None and lidar[4]))
        self.temps.append(temp)
        if track is not None and (not self.tracks or self.tracks[-1] != track):
            self.tracks.append(track)
        self.sightings += 1
        self.last_seen = now

    def _usable_estimates(self, persistence=0.6):
        """Direct estimates, keeping each method only if it is consistent."""
        n = len(self.history)
        floor_ok = n and sum(h[0] for h in self.history) / n >= persistence
        lidar_ok = n and sum(h[1] for h in self.history) / n >= persistence
        confident, ambiguous = [], []
        for e in self.direct:
            if e['method'] == 'floor':
                if floor_ok:
                    confident.append((e['x'], e['y'], e['sigma']))
            elif lidar_ok:
                confident.append((e['x'], e['y'], e['sigma']))
            else:  # lidar not consistently trustworthy for this target
                ambiguous.append((e['x'], e['y'], e['ambiguous_sigma']))
        return confident, ambiguous

    def recompute(self, min_spread, min_baseline):
        tri = triangulate(list(self.rays), min_spread, min_baseline)
        self.triangulated = tri is not None
        estimates, ambiguous = self._usable_estimates()
        if tri is not None:
            gate = max(3.0 * tri[2], 0.5)
            estimates = [e for e in estimates if math.hypot(e[0] - tri[0], e[1] - tri[1]) <= gate]
            ambiguous = [e for e in ambiguous if math.hypot(e[0] - tri[0], e[1] - tri[1]) <= gate]
        # Successive direct estimates share the same biases, so don't let
        # their count shrink the uncertainty without limit.
        scale = min(1.0, 5.0 / len(estimates)) if estimates else 1.0
        weights = [scale / (e[2] ** 2) for e in estimates]
        xs = [e[0] for e in estimates]
        ys = [e[1] for e in estimates]
        if ambiguous:
            # Ambiguous lidar readings tend to be wrong the SAME way (the wall
            # behind a low target), so repeating them adds no confidence: they
            # count together as one reading with the best single uncertainty.
            aw = [1.0 / e[2] ** 2 for e in ambiguous]
            weights.append(max(aw))
            xs.append(sum(w * e[0] for w, e in zip(aw, ambiguous)) / sum(aw))
            ys.append(sum(w * e[1] for w, e in zip(aw, ambiguous)) / sum(aw))
        if tri is not None:
            weights.append(1.0 / tri[2] ** 2)
            xs.append(tri[0])
            ys.append(tri[1])
        if not weights:
            return
        total = sum(weights)
        self.x = sum(w * x for w, x in zip(weights, xs)) / total
        self.y = sum(w * y for w, y in zip(weights, ys)) / total
        self.sigma = max(0.05, 1.0 / math.sqrt(total))

    def relocate(self):
        """The target has moved: drop the old position evidence, keep identity."""
        self.direct.clear()
        self.rays.clear()
        self.history.clear()
        self.x = self.y = None
        self.sigma = math.inf
        self.triangulated = False
        self.disagree = 0

    def merge(self, other):
        self.direct.extend(other.direct)
        self.history.extend(other.history)
        self.rays.extend(other.rays)
        self.temps.extend(other.temps)
        self.tracks.extend(other.tracks)
        self.sightings += other.sightings
        self.first_seen = min(self.first_seen, other.first_seen)
        self.last_seen = max(self.last_seen, other.last_seen)

    def as_dict(self, heat_source_c):
        return {'id': self.id, 'kind': self.kind(heat_source_c),
                'x': round(self.x, 3), 'y': round(self.y, 3),
                'sigma_m': round(self.sigma, 3), 'confidence': self.confidence(),
                'median_temp_c': round(float(np.median(self.temps)), 1),
                'peak_temp_c': round(float(max(self.temps)), 1),
                'sightings': self.sightings, 'triangulated': self.triangulated}


class CameraFixedFilter:
    """Learns camera-frame bearings that never move with the robot (bad pixels)."""

    def __init__(self, tol_deg=3.0, min_turn_deg=25.0, faulty=None):
        self.tol = tol_deg
        self.min_turn = math.radians(min_turn_deg)
        self.faulty = list(faulty or [])      # [(azimuth_deg, elevation_deg)]
        self._tracks = {}                     # track -> stats

    def is_faulty(self, az, el):
        return any(math.hypot(az - fa, el - fe) <= self.tol for fa, fe in self.faulty)

    def observe(self, track, az, el, cam_yaw, now):
        """Returns the newly learned faulty bearing for this track, or None."""
        if track is None:
            return None
        t = self._tracks.get(track)
        if t is None or now - t['last'] > 2.0:
            t = {'az0': az, 'el0': el, 'yaw0': cam_yaw, 'yaw_min': 0.0, 'yaw_max': 0.0,
                 'drift': 0.0, 'last': now}
            self._tracks[track] = t
        t['last'] = now
        rel = wrap(cam_yaw - t['yaw0'])
        t['yaw_min'], t['yaw_max'] = min(t['yaw_min'], rel), max(t['yaw_max'], rel)
        t['drift'] = max(t['drift'], math.hypot(az - t['az0'], el - t['el0']))
        if t['drift'] > self.tol:
            return None  # it moves in the image, as real objects do while turning
        if t['yaw_max'] - t['yaw_min'] < self.min_turn:
            return None  # the robot hasn't turned enough yet to judge
        bearing = (round(t['az0'], 1), round(t['el0'], 1))
        del self._tracks[track]
        if self.is_faulty(*bearing):
            return None
        self.faulty.append(bearing)
        return bearing

    def prune(self, now):
        self._tracks = {k: t for k, t in self._tracks.items() if now - t['last'] <= 2.0}


# --- ROS node ---------------------------------------------------------------------

class VictimMapper(Node):

    def __init__(self):
        super().__init__('victim_mapper')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.floor_z = p('floor_z_m', -0.0325).value
        self.floor_max_range = p('floor_max_range_m', 1.5).value
        self.min_depression = math.radians(p('floor_min_depression_deg', 2.5).value)
        self.sigma_angle = math.radians(p('bearing_sigma_deg', 1.5).value)
        self.lidar_window = math.radians(p('lidar_window_deg', 2.5).value)
        self.surface_offset = p('lidar_surface_offset_m', 0.10).value
        self.tri_spread = math.radians(p('tri_min_spread_deg', 8.0).value)
        self.tri_baseline = p('tri_min_baseline_m', 0.25).value
        self.assoc_radius = p('assoc_radius_m', 0.6).value
        self.merge_radius = p('merge_radius_m', 0.4).value
        self.heat_source_c = p('heat_source_c', 45.0).value
        self.max_turn = p('max_turn_rate_rad_s', 1.0).value
        self.active_window = p('active_window_s', 1.5).value
        self.move_frames = p('move_confirm_frames', 5).value
        self.move_min_m = p('move_min_m', 0.5).value
        self.move_max_sigma = p('move_max_sigma_m', 0.4).value
        self.faulty_file = os.path.expanduser(
            p('faulty_directions_file', '~/.ros/thermal_faulty_directions.yaml').value)
        self.fixed_filter = CameraFixedFilter(
            tol_deg=p('fixed_bearing_tol_deg', 3.0).value,
            min_turn_deg=p('fixed_min_turn_deg', 25.0).value,
            faulty=self._load_faulty())
        self._track_victims = {}   # thermal track -> {victim id: sightings}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.victims = []
        self._next_id = 1
        self._scan_xy = None
        self._scan_time = 0.0
        self._lidar_z = None
        self._last_cam_yaw = None
        self._last_cam_t = None
        self._skipped_turning = 0
        self._last_status = time.time()

        self.create_subscription(PointCloud2, '/thermal/hotspots', self._on_hotspots, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.pub_markers = self.create_publisher(MarkerArray, '/victims/markers', 10)
        self.pub_poses = self.create_publisher(PoseArray, '/victims', 10)
        self.pub_active = self.create_publisher(PoseArray, '/victims/active', 10)
        self.create_service(Trigger, '/victims/save', self._srv_save)
        self.create_service(Trigger, '/victims/clear', self._srv_clear)
        self.create_timer(0.5, self._publish)
        self.create_timer(30.0, self._autosave)
        self.get_logger().info('Victim mapper up. Needs the map frame (SLAM or AMCL) and '
                               'the thermal node running.')

    # -- inputs --------------------------------------------------------------

    def _lookup(self, source):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, source, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)

    def _on_scan(self, msg):
        pose = self._lookup(msg.header.frame_id)
        if pose is None:
            return
        (lx, ly, lz), q = pose
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        ok = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        yaw = yaw_of(q)
        a = angles[ok] + yaw
        self._scan_xy = np.stack([lx + ranges[ok] * np.cos(a), ly + ranges[ok] * np.sin(a)], 1)
        self._scan_time = time.time()
        self._lidar_z = lz

    @staticmethod
    def _read_cloud(msg):
        names = {f.name: f.offset // 4 for f in msg.fields}
        count = msg.width * msg.height
        if count == 0:
            return [], names
        dtype = '>f4' if msg.is_bigendian else '<f4'
        rows = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(count, msg.point_step // 4)
        return rows, names

    def _on_hotspots(self, msg):
        rows, cols = self._read_cloud(msg)
        cam = self._lookup(msg.header.frame_id)
        if cam is None:
            return
        origin, q = cam
        now = time.time()

        # Skip frames taken mid-turn: TF and image are most likely out of step.
        cam_yaw = yaw_of(q)
        if self._last_cam_yaw is not None and now > self._last_cam_t:
            rate = abs(wrap(cam_yaw - self._last_cam_yaw)) / (now - self._last_cam_t)
            self._last_cam_yaw, self._last_cam_t = cam_yaw, now
            if rate > self.max_turn:
                self._skipped_turning += 1
                return
        self._last_cam_yaw, self._last_cam_t = cam_yaw, now

        scan_xy = self._scan_xy if now - self._scan_time < 0.5 else None
        for row in rows:
            direction = quat_rotate(q, (float(row[cols['x']]), float(row[cols['y']]),
                                        float(row[cols['z']])))
            temp = float(row[cols['temperature']])
            track = int(row[cols['track']]) if 'track' in cols else None
            cx, cy, cz = float(row[cols['x']]), float(row[cols['y']]), float(row[cols['z']])
            az = math.degrees(math.atan2(cx, cz))                   # + right, camera frame
            el = math.degrees(math.asin(max(-1.0, min(1.0, cy))))   # + down
            if self.fixed_filter.is_faulty(az, el):
                continue
            learned = self.fixed_filter.observe(track, az, el, cam_yaw, now)
            if learned is not None:
                self._forget_track(track, learned)
                continue
            victim = self._handle_ray(origin, direction, temp, track, scan_xy, now)
            if track is not None and victim is not None:
                counts = self._track_victims.setdefault(track, {})
                counts[victim.id] = counts.get(victim.id, 0) + 1

        self.fixed_filter.prune(now)
        self._merge_close()

    def _forget_track(self, track, bearing):
        """A track turned out to be a faulty pixel: drop what it created."""
        counts = self._track_victims.pop(track, {})
        removed = [v for v in self.victims
                   if counts.get(v.id, 0) >= 0.5 * max(1, v.sightings)]
        self.victims = [v for v in self.victims if v not in removed]
        az, el = bearing
        self.get_logger().warning(
            f'Ignoring a camera-fixed hotspot at {abs(az):.0f} deg '
            f'{"right" if az >= 0 else "left"}, {abs(el):.0f} deg '
            f'{"down" if el >= 0 else "up"} (it never moved while the robot turned: a '
            f'faulty pixel). Removed {len(removed)} victim(s) built from it.')
        self._save_faulty()

    def _load_faulty(self):
        try:
            with open(self.faulty_file) as handle:
                data = yaml.safe_load(handle) or {}
            bearings = [tuple(b) for b in data.get('faulty_bearings_deg', [])]
            if bearings:
                self.get_logger().info(f'Loaded {len(bearings)} known faulty thermal '
                                       f'direction(s) from {self.faulty_file}.')
            return bearings
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            return []

    def _save_faulty(self):
        try:
            os.makedirs(os.path.dirname(self.faulty_file), exist_ok=True)
            with open(self.faulty_file, 'w') as handle:
                yaml.safe_dump({'faulty_bearings_deg': [list(b) for b in self.fixed_filter.faulty]},
                               handle)
        except OSError as exc:
            self.get_logger().warning(f'Could not save faulty directions: {exc}')

    # -- core ----------------------------------------------------------------

    def _handle_ray(self, origin, direction, temp, track, scan_xy, now):
        heading = math.atan2(direction[1], direction[0])
        floor = floor_estimate(origin, direction, self.floor_z, self.floor_max_range,
                               self.min_depression, self.sigma_angle)
        upper = floor_range_upper(origin, direction, self.floor_z, self.sigma_angle)
        lidar = lidar_estimate(origin, direction, scan_xy, self.lidar_window,
                               self.sigma_angle, upper, self._lidar_z,
                               surface_offset=self.surface_offset)
        options = [e for e in (lidar, floor) if e is not None]
        estimate = min(options, key=lambda e: e[2]) if options else None  # for association

        victim = self._associate(origin, heading, estimate, track, now)
        if victim is None:
            victim = Victim(self._next_id, now)
            self._next_id += 1
            self.victims.append(victim)
        elif victim.located:
            self._check_moved(victim, estimate)
        points_down = -math.asin(max(-1.0, min(1.0, direction[2]))) >= self.min_depression
        victim.add((origin[0], origin[1], heading), floor, lidar, temp, track, now, points_down)
        victim.recompute(self.tri_spread, self.tri_baseline)
        return victim

    def _check_moved(self, victim, estimate):
        """Relocate a victim when confident readings keep landing elsewhere."""
        if estimate is None or estimate[2] > self.move_max_sigma:
            return  # only confident readings can say the target moved
        gap = math.hypot(estimate[0] - victim.x, estimate[1] - victim.y)
        limit = max(self.move_min_m, 3.0 * math.hypot(estimate[2], victim.sigma))
        victim.disagree = victim.disagree + 1 if gap > limit else 0
        if victim.disagree >= self.move_frames:
            victim.relocate()
            self.get_logger().info(f'Victim {victim.id} has moved; re-locating it.')

    def _associate(self, origin, heading, estimate, track, now):
        # 1. The thermal node is still tracking this same target.
        if track is not None:
            for v in self.victims:
                if v.tracks and v.tracks[-1] == track and now - v.last_seen < 2.0:
                    return v
        located = [v for v in self.victims if v.located]
        # 2. A position estimate close to a known victim.
        if estimate is not None and located:
            best = min(located, key=lambda v: math.hypot(v.x - estimate[0], v.y - estimate[1]))
            if math.hypot(best.x - estimate[0], best.y - estimate[1]) <= self.assoc_radius + best.sigma:
                return best
        # 3. The ray passes close in front of a known victim.
        ux, uy = math.cos(heading), math.sin(heading)
        best, best_d = None, self.assoc_radius
        for v in located:
            rx, ry = v.x - origin[0], v.y - origin[1]
            along = rx * ux + ry * uy
            if along <= 0:
                continue
            across = abs(-uy * rx + ux * ry)
            if across < best_d:
                best, best_d = v, across
        return best

    def _merge_close(self):
        located = [v for v in self.victims if v.located]
        for i, a in enumerate(located):
            for b in located[i + 1:]:
                if a in self.victims and b in self.victims and \
                        math.hypot(a.x - b.x, a.y - b.y) < self.merge_radius:
                    keep, drop = (a, b) if a.sightings >= b.sightings else (b, a)
                    keep.merge(drop)
                    keep.recompute(self.tri_spread, self.tri_baseline)
                    self.victims.remove(drop)
        # Forget unlocated stragglers that never firmed up.
        now = time.time()
        self.victims = [v for v in self.victims
                        if v.located or now - v.last_seen < 15.0]

    # -- outputs -------------------------------------------------------------

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id = self.map_frame
        markers.markers.append(clear)
        poses = PoseArray()
        poses.header.frame_id, poses.header.stamp = self.map_frame, stamp
        active_poses = PoseArray()
        active_poses.header = poses.header
        now = time.time()

        for v in [v for v in self.victims if v.located]:
            kind = v.kind(self.heat_source_c)
            person = kind == 'person'
            age = now - v.last_seen
            active = age <= self.active_window
            alpha = {'high': 1.0, 'medium': 0.75, 'low': 0.45}[v.confidence()]
            if active:
                r, g, b = (1.0, 0.10, 0.08) if person else (1.0, 0.60, 0.05)
            else:  # remembered: darker and see-through
                r, g, b = (0.50, 0.10, 0.08) if person else (0.60, 0.36, 0.05)
                alpha *= 0.55

            sphere = Marker(type=Marker.SPHERE, action=Marker.ADD, ns='victims', id=v.id)
            sphere.header.frame_id, sphere.header.stamp = self.map_frame, stamp
            sphere.pose.position = Point(x=v.x, y=v.y, z=0.12)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.30 if active else 0.20
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, alpha

            disc = Marker(type=Marker.CYLINDER, action=Marker.ADD, ns='uncertainty', id=v.id)
            disc.header = sphere.header
            disc.pose.position = Point(x=v.x, y=v.y, z=0.0)
            disc.pose.orientation.w = 1.0
            disc.scale.x = disc.scale.y = max(0.1, 4.0 * v.sigma)   # diameter = 2 sigma radius
            disc.scale.z = 0.01
            disc.color.r, disc.color.g, disc.color.b, disc.color.a = r, g, b, 0.18

            label = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns='labels', id=v.id)
            label.header = sphere.header
            label.pose.position = Point(x=v.x, y=v.y, z=0.42)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            name = f'Person {v.id}' if person else f'Heat source {v.id}'
            when = 'SEEING NOW' if active else f'last seen {self._ago(age)} ago'
            label.text = (f'{name} - {when}\n{float(np.median(v.temps)):.1f} C, '
                          f'{v.confidence()}, seen {v.sightings}')
            markers.markers += [sphere, disc, label]

            if person:
                pose = Pose()
                pose.position = Point(x=v.x, y=v.y, z=0.0)
                pose.orientation.w = 1.0
                poses.poses.append(pose)
                if active:
                    active_poses.poses.append(pose)

        self.pub_markers.publish(markers)
        self.pub_active.publish(active_poses)
        self.pub_poses.publish(poses)

        if time.time() - self._last_status > 30.0:
            self._last_status = time.time()
            people = [v for v in self.victims if v.located and v.kind(self.heat_source_c) == 'person']
            self.get_logger().info(
                f'{len(people)} person(s) located, '
                f'{sum(1 for v in self.victims if v.located) - len(people)} heat source(s), '
                f'{sum(1 for v in self.victims if not v.located)} still being located. '
                f'Frames skipped while turning: {self._skipped_turning}.')

    @staticmethod
    def _ago(seconds):
        if seconds < 60:
            return f'{seconds:.0f} s'
        if seconds < 3600:
            return f'{seconds / 60:.0f} min'
        return f'{seconds / 3600:.1f} h'

    def _snapshot(self):
        return {'saved': datetime.now().isoformat(timespec='seconds'),
                'frame': self.map_frame,
                'victims': [v.as_dict(self.heat_source_c) for v in self.victims if v.located]}

    def _write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            yaml.safe_dump(self._snapshot(), handle, sort_keys=False)

    def _autosave(self):
        if any(v.located for v in self.victims):
            self._write(AUTOSAVE)

    def _srv_save(self, _request, response):
        path = os.path.join(MAP_DIR, datetime.now().strftime('victims_%Y%m%d_%H%M%S.yaml'))
        self._write(path)
        response.success, response.message = True, f'Saved to {path}'
        return response

    def _srv_clear(self, _request, response):
        self.victims.clear()
        response.success, response.message = True, 'All victims cleared.'
        return response


def main():
    rclpy.init()
    node = VictimMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if any(v.located for v in node.victims):
            node._write(AUTOSAVE)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()