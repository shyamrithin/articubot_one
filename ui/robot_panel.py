#!/usr/bin/env python3
# =============================================================================
# File:        ui/robot_panel.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Full-screen touchscreen panel for the robot's own 7" display. It starts and
# stops the robot stack and shows health at a glance. Visualisation stays in
# RViz on the laptop, so this sends nothing over wifi.
#
#   Top bar      IP address (large), hostname, current mode
#   Status line  what is happening now, or the latest warning / error
#   Mode tiles   Map           drivetrain + lidar + SLAM, new map
#                Explore       drivetrain + lidar + SLAM + Nav2
#                Navigate      drivetrain + lidar + AMCL + Nav2 on the chosen map
#                Continue map  drivetrain + lidar + SLAM resumed from the chosen
#                              map (only maps saved by this panel can resume)
#   Actions      saved-map picker, Save map, Robot is at map start, Stop all
#   Stop robot   software e-stop: zero twist on /cmd_vel_estop at 20 Hz while
#                engaged, which twist_mux ranks above everything (priority 255)
#   Diagnostics  pose in map and odom frames, wheel ticks and speeds, commanded
#                vs measured speed, topic rates, CPU / RAM / temperature /
#                disk / power, and each layer's state
#   Log bar      latest line from the most relevant layer's log
#
# HOW LAYERS ARE STARTED
# ----------------------
# Each layer is one `ros2 launch` in its own process group, started in order
# and checked for readiness before the next one:
#   drivetrain   -> /joint_states arriving
#   lidar        -> /scan arriving
#   SLAM         -> /map arriving
#   localisation -> map_server and amcl lifecycle state ACTIVE (one retry)
#   navigation   -> bt_navigator lifecycle state ACTIVE
# In Navigate mode Nav2 cannot finish until the robot's position is set, either
# with RViz's 2D Pose Estimate or the "Robot is at map start" button.
#
# SAVED MAPS
# ----------
# Save map writes, into ~/maps:
#   <name>.yaml + <name>.pgm            map_saver image  -> Navigate
#   <name>.posegraph + <name>.data      slam_toolbox graph -> Continue map
# Continue map starts SLAM with map_file_name set to that graph and
# map_start_at_dock on, so put the robot back where that map was first begun.
#
# RUNNING
# -------
#   From ssh:    ~/robot_ws/src/articubot_one/ui/start_panel.sh
#   On screen:   autostart via ui/robot-panel.desktop
#   Exit:        Escape on a keyboard, Ctrl-C in the ssh session, or SIGTERM.
#                Exiting stops every layer the panel started.
#
# Layer logs: ~/.ros/robot_panel_logs/<layer>.log
#
# Only one control program may run at once. A lock file stops a second panel,
# and the earlier web UI (robot_supervisor.py) must not run alongside it.
# =============================================================================

import fcntl
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from datetime import datetime

import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan

try:
    from slam_toolbox.srv import SerializePoseGraph
except ImportError:  # panel still works; maps just can't be resumed
    SerializePoseGraph = None

# --- Configuration ------------------------------------------------------------

PACKAGE = 'articubot_one'
SLAM_PARAMS_FILE = 'mapper_params_async.yaml'
MAP_DIR = os.path.expanduser(os.environ.get('ROBOT_MAP_DIR', '~/maps'))
RUN_DIR = os.path.expanduser('~/.ros/robot_panel')
LOG_DIR = os.path.expanduser('~/.ros/robot_panel_logs')

MAP_FRAME = 'map'
ODOM_FRAME = 'odom'
BASE_FRAME = 'base_footprint'
ODOM_TOPIC = '/diff_cont/odom'
CMD_TOPIC = '/diff_cont/cmd_vel_unstamped'
LEFT_JOINT = 'left_wheel_joint'
RIGHT_JOINT = 'right_wheel_joint'
WHEEL_RADIUS_M = 0.0325
ENCODER_CPR = float(os.environ.get('ROBOT_ENCODER_CPR', '510'))

FULLSCREEN = os.environ.get('PANEL_WINDOWED', '0') != '1'
HIDE_CURSOR = os.environ.get('PANEL_SHOW_CURSOR', '0') != '1'


def _launch(*args):
    return ['ros2', 'launch', PACKAGE, *args]


LAYER_COMMANDS = {
    'drive': _launch('launch_robot.launch.py'),
    'lidar': _launch('rplidar.launch.py'),
    'slam': _launch('online_async_launch.py', 'use_sim_time:=false'),
    'navigation': _launch('navigation_launch.py', 'use_sim_time:=false',
                          'map_subscribe_transient_local:=true'),
}
LAYER_LABELS = {
    'drive': 'Drivetrain',
    'lidar': 'Lidar',
    'slam': 'Mapping',
    'localization': 'Localisation',
    'navigation': 'Navigation',
}
MODES = {
    'map': ['drive', 'lidar', 'slam'],
    'explore': ['drive', 'lidar', 'slam', 'navigation'],
    'navigate': ['drive', 'lidar', 'localization', 'navigation'],
    'resume': ['drive', 'lidar', 'slam'],
}
MODE_LABELS = {'map': 'Map', 'explore': 'Explore', 'navigate': 'Navigate',
               'resume': 'Continue map'}
START_ORDER = ['drive', 'lidar', 'slam', 'localization', 'navigation']
STOP_ORDER = list(reversed(START_ORDER))

READY_TIMEOUT_S = {'drive': 25.0, 'lidar': 20.0, 'slam': 40.0,
                   'localization': 45.0, 'navigation': 90.0}
NAV_WAIT_FOR_POSE_S = 600.0

SAFE_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

# Palette: cool concrete panel, ink text, hi-vis orange for "active".
BG = '#E4E9EC'
PANEL = '#F6F8F9'
INK = '#13222E'
MUTED = '#56666F'
LINE = '#C6D0D6'
HIVIS = '#F26B1D'
STOP = '#C3201C'
HAZARD = '#F2C200'
OK = '#17855A'
WARN = '#9A6408'
WHITE = '#FFFFFF'
DIM = '#9AA6AE'


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Aborted(Exception):
    """Raised inside a start-up wait when Stop all is pressed."""


class RateMeter:
    """Thread-safe message rate over a sliding window."""

    def __init__(self, window_s=3.0):
        self._window = window_s
        self._times = deque(maxlen=400)
        self._lock = threading.Lock()
        self.last = 0.0

    def tick(self):
        now = time.time()
        with self._lock:
            self._times.append(now)
        self.last = now

    def hz(self):
        now = time.time()
        with self._lock:
            recent = [t for t in self._times if now - t < self._window]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0])


# --- Process management -------------------------------------------------------

class ManagedProcess:
    """One command (usually `ros2 launch`) running in its own process group."""

    def __init__(self, name, cmd):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.started_at = None
        self.log_path = os.path.join(LOG_DIR, f'{name}.log')
        self._log = None

    def start(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log = open(self.log_path, 'w', buffering=1)
        self._log.write(f'$ {" ".join(self.cmd)}\n')
        self.proc = subprocess.Popen(self.cmd, stdout=self._log,
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        self.started_at = time.time()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace_s=12.0):
        """SIGINT (lets ros2 launch shut its nodes down), then TERM, then KILL."""
        if self.proc is None:
            return
        pgid = self.proc.pid  # equals the group id because of start_new_session
        if self.proc.poll() is None:
            for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0),
                                (signal.SIGKILL, 3.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        try:
            os.killpg(pgid, signal.SIGKILL)  # sweep nodes that outlived launch
        except ProcessLookupError:
            pass
        if self._log is not None and not self._log.closed:
            self._log.close()

    def last_line(self):
        try:
            with open(self.log_path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 4096))
                text = handle.read().decode(errors='replace')
        except OSError:
            return ''
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return ANSI_ESCAPE.sub('', lines[-1]) if lines else ''


# --- ROS side -------------------------------------------------------------------

class RobotCore(Node):

    def __init__(self):
        super().__init__('robot_panel')
        self._cb = ReentrantCallbackGroup()

        self._transition_lock = threading.Lock()
        self._abort = threading.Event()
        self._shut_down = False
        self.procs = {}
        self.mode = 'idle'
        self.phase = 'Idle. Choose a mode to start.'
        self.busy = False
        self.active_map = None
        self._nav_source = None
        self.events = deque(maxlen=12)
        self.estop = False

        self.rates = {name: RateMeter() for name in ('scan', 'odom', 'joints', 'cmd', 'map')}
        self.joints = {}
        self.odom_twist = None
        self.cmd = None
        self.sys = {'ip': '', 'host': socket.gethostname()}
        self._sys_slow_t = 0.0
        self._cpu_prev = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.estop_pub = self.create_publisher(Twist, '/cmd_vel_estop', 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        sensor = qos_profile_sensor_data
        self.create_subscription(LaserScan, '/scan', lambda _m: self.rates['scan'].tick(),
                                 sensor, callback_group=self._cb)
        self.create_subscription(OccupancyGrid, '/map', lambda _m: self.rates['map'].tick(),
                                 latched, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states', self._on_joints, sensor,
                                 callback_group=self._cb)
        self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, sensor,
                                 callback_group=self._cb)
        self.create_subscription(Twist, CMD_TOPIC, self._on_cmd, 10,
                                 callback_group=self._cb)

        self._state_clients = {
            name: self.create_client(GetState, f'/{name}/get_state', callback_group=self._cb)
            for name in ('map_server', 'amcl', 'bt_navigator')
        }
        self._serialize_client = None
        if SerializePoseGraph is not None:
            self._serialize_client = self.create_client(
                SerializePoseGraph, '/slam_toolbox/serialize_map', callback_group=self._cb)

        self.create_timer(0.05, self._publish_estop, callback_group=self._cb)
        self.create_timer(1.0, self._update_system, callback_group=self._cb)
        self._update_system()
        self._event('Panel ready.')

    # -- events ------------------------------------------------------------

    def _event(self, text, level='info'):
        self.events.appendleft({'wall': time.time(),
                                't': datetime.now().strftime('%H:%M:%S'),
                                'level': level, 'text': text})
        # rclpy refuses different severities from one call site.
        logger = self.get_logger()
        if level == 'error':
            logger.error(text)
        elif level == 'warn':
            logger.warning(text)
        else:
            logger.info(text)

    # -- modes -------------------------------------------------------------

    def request_mode(self, target, map_name=None):
        if target != 'idle' and target not in MODES:
            self._event(f'Unknown mode "{target}".', 'error')
            return
        if target == 'navigate' and not self._map_file(map_name, '.yaml'):
            self._event('Pick a saved map before starting Navigate.', 'error')
            return
        if target == 'resume' and not self._map_file(map_name, '.posegraph'):
            self._event('That map has no SLAM data to continue from. '
                        'Only maps saved from this panel can be continued.', 'error')
            return
        if target == 'idle':
            self._abort.set()  # interrupt any start-up in progress
            threading.Thread(target=self._run_transition, args=('idle', None, True),
                             daemon=True).start()
            return
        if not self._transition_lock.acquire(blocking=False):
            self._event('Still changing modes. Wait for it, or press Stop all.', 'warn')
            return
        threading.Thread(target=self._run_transition, args=(target, map_name, False),
                         daemon=True).start()

    def _run_transition(self, target, map_name, need_lock):
        if need_lock:
            self._transition_lock.acquire()
        self._abort.clear()
        self.busy = True
        try:
            self._transition(target, map_name)
        except Aborted:
            self._event('Start-up interrupted.', 'warn')
        except Exception as exc:  # noqa: BLE001  (show anything on the screen)
            self.phase = f'Could not start: {exc}'
            self._event(f'Mode change failed: {exc}', 'error')
        finally:
            self.busy = False
            self._transition_lock.release()

    def _transition(self, target, map_name):
        wanted = [] if target == 'idle' else MODES[target]
        desired = {name: self._desired_cmd(name, target, map_name) for name in wanted}
        nav_source = None if target == 'idle' else (
            'amcl' if 'localization' in wanted else 'slam')

        for name in STOP_ORDER:
            proc = self.procs.get(name)
            if proc is None:
                continue
            must_stop = (name not in wanted
                         or proc.cmd != desired[name]
                         or (name == 'navigation' and nav_source != self._nav_source))
            if must_stop:
                self.phase = f'Stopping {LAYER_LABELS[name].lower()}...'
                self.procs.pop(name).stop()

        if target == 'idle':
            self.mode = 'idle'
            self.active_map = None
            self._nav_source = None
            self.phase = 'Idle. Choose a mode to start.'
            self._event('Everything stopped.')
            return

        self.mode = target
        self.active_map = map_name if target in ('navigate', 'resume') else None
        self._nav_source = nav_source
        for name in START_ORDER:
            if name not in wanted:
                continue
            proc = self.procs.get(name)
            if proc is not None and proc.alive():
                continue
            self._start_layer(name, desired[name])

        self.phase = {
            'map': 'Mapping. Drive around slowly, then save the map.',
            'explore': 'Exploring. Send goals from RViz; the map grows as it goes.',
            'navigate': f'Navigating on {map_name}. Send goals from RViz.',
            'resume': f'Continuing {map_name}. Drive on, then save the map.',
        }[target]
        self._event(f'{MODE_LABELS[target]} ready.')

    def _desired_cmd(self, name, target, map_name):
        if name == 'localization':
            return _launch('localization_launch.py', 'use_sim_time:=false',
                           f'map:={self._map_file(map_name, ".yaml")}')
        if name == 'slam' and target == 'resume':
            return _launch('online_async_launch.py', 'use_sim_time:=false',
                           f'slam_params_file:={self._resume_params(map_name)}')
        return LAYER_COMMANDS[name]

    def _resume_params(self, map_name):
        """Copy the SLAM params with map_file_name pointing at a saved graph."""
        source = os.path.join(get_package_share_directory(PACKAGE), 'config', SLAM_PARAMS_FILE)
        with open(source) as handle:
            params = yaml.safe_load(handle)
        p = params['slam_toolbox']['ros__parameters']
        p['mode'] = 'mapping'
        p['map_file_name'] = os.path.join(MAP_DIR, map_name)
        p['map_start_at_dock'] = True
        p.pop('map_start_pose', None)
        os.makedirs(RUN_DIR, exist_ok=True)
        path = os.path.join(RUN_DIR, f'slam_resume_{map_name}.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(params, handle, default_flow_style=False)
        return path

    def _start_layer(self, name, cmd):
        old = self.procs.pop(name, None)
        if old is not None:
            old.stop()
        label = LAYER_LABELS[name].lower()
        waiting_on_person = name == 'navigation' and self.mode == 'navigate'
        attempts = 1 if waiting_on_person else 2
        for attempt in range(1, attempts + 1):
            self.phase = f'Starting {label}...' if attempt == 1 else f'Retrying {label}...'
            proc = ManagedProcess(name, cmd)
            proc.start()
            self.procs[name] = proc
            if self._wait_ready(name, proc, time.time()):
                return
            if attempt < attempts:
                self._event(f'{LAYER_LABELS[name]} did not come up; retrying once.', 'warn')
                self.procs.pop(name).stop()
        raise RuntimeError(f'{label} did not start. See {proc.log_path}')

    def _wait_ready(self, name, proc, started):
        if name == 'drive':
            return self._wait(lambda: self.rates['joints'].last > started,
                              READY_TIMEOUT_S['drive'], proc)
        if name == 'lidar':
            return self._wait(lambda: self.rates['scan'].last > started,
                              READY_TIMEOUT_S['lidar'], proc)
        if name == 'slam':
            return self._wait(lambda: self.rates['map'].last > started,
                              READY_TIMEOUT_S['slam'], proc)
        if name == 'localization':
            return self._wait(lambda: (self._node_active('map_server')
                                       and self._node_active('amcl')),
                              READY_TIMEOUT_S['localization'], proc)
        if name == 'navigation':
            needs_pose = self.mode == 'navigate'
            timeout = NAV_WAIT_FOR_POSE_S if needs_pose else READY_TIMEOUT_S['navigation']
            return self._wait(lambda: self._node_active('bt_navigator'),
                              timeout, proc, pose_hint=needs_pose)
        return True

    def _wait(self, condition, timeout_s, proc, pose_hint=False):
        deadline = time.time() + timeout_s
        base_phase = self.phase
        while time.time() < deadline:
            if self._abort.is_set():
                raise Aborted()
            if not proc.alive():
                self._event(f'{LAYER_LABELS[proc.name]} exited: {proc.last_line()[-120:]}',
                            'error')
                return False
            if condition():
                self.phase = base_phase
                return True
            if pose_hint and self._pose(MAP_FRAME) is None:
                self.phase = ('Set the robot\u2019s position: 2D Pose Estimate in RViz, '
                              'or Robot is at map start.')
            time.sleep(0.25)
        return False

    def _call(self, client, request, timeout_s):
        if client is None or not client.service_is_ready():
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            future.cancel()
            return None
        return future.result()

    def _node_active(self, node_name):
        result = self._call(self._state_clients[node_name], GetState.Request(), 2.0)
        return result is not None and result.current_state.id == State.PRIMARY_STATE_ACTIVE

    # -- maps --------------------------------------------------------------

    @staticmethod
    def _map_file(name, ext):
        if not name or not SAFE_NAME.fullmatch(str(name)):
            return None
        path = os.path.join(MAP_DIR, f'{name}{ext}')
        return path if os.path.isfile(path) else None

    def list_maps(self):
        """Newest first: [{'name': ..., 'resumable': bool}]."""
        try:
            names = [e[:-5] for e in os.listdir(MAP_DIR) if e.endswith('.yaml')]
        except FileNotFoundError:
            return []
        names.sort(key=lambda n: os.path.getmtime(os.path.join(MAP_DIR, f'{n}.yaml')),
                   reverse=True)
        return [{'name': n,
                 'resumable': os.path.isfile(os.path.join(MAP_DIR, f'{n}.posegraph'))}
                for n in names]

    def save_map(self):
        threading.Thread(target=self._save_map, daemon=True).start()

    def _save_map(self):
        slam = self.procs.get('slam')
        if slam is None or not slam.alive():
            self._event('Start Map, Explore or Continue map first; there is no live map.',
                        'error')
            return
        base = datetime.now().strftime('map_%Y%m%d_%H%M')
        name, suffix = base, 2
        os.makedirs(MAP_DIR, exist_ok=True)
        while os.path.exists(os.path.join(MAP_DIR, f'{name}.yaml')):
            name, suffix = f'{base}_{suffix}', suffix + 1
        stem = os.path.join(MAP_DIR, name)
        self._event(f'Saving map as {name}...')

        try:
            result = subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                                     '-f', stem],
                                    capture_output=True, text=True, timeout=30)
            saved_image = result.returncode == 0 and os.path.isfile(f'{stem}.yaml')
        except subprocess.TimeoutExpired:
            saved_image = False
        if not saved_image:
            self._event('Map image could not be saved.', 'error')
            return

        if SerializePoseGraph is None:
            self._event(f'Saved {name} for Navigate (slam_toolbox services not found, '
                        'so it cannot be continued).', 'warn')
            return
        request = SerializePoseGraph.Request()
        request.filename = stem
        self._call(self._serialize_client, request, 20.0)
        if os.path.isfile(f'{stem}.posegraph'):
            self._event(f'Saved {name}: ready for Navigate and Continue map.')
        else:
            self._event(f'Saved {name} for Navigate only; the SLAM graph did not save.',
                        'warn')

    # -- safety and pose ---------------------------------------------------

    def set_estop(self, on):
        if on and not self.estop:
            self.estop = True
            self._event('Stop engaged. The robot will not move until you release it.',
                        'warn')
        elif not on and self.estop:
            self.estop = False
            self._event('Stop released.')

    def _publish_estop(self):
        if self.estop:
            self.estop_pub.publish(Twist())

    def place_at_map_start(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = 0.25   # same defaults as RViz
        covariance[35] = 0.0685
        msg.pose.covariance = covariance
        self.initial_pose_pub.publish(msg)
        self._event('Position set to the map start.')

    # -- sensor callbacks --------------------------------------------------

    def _on_joints(self, msg):
        self.rates['joints'].tick()
        joints = dict(self.joints)
        for i, name in enumerate(msg.name):
            pos = msg.position[i] if i < len(msg.position) else None
            vel = msg.velocity[i] if i < len(msg.velocity) else None
            joints[name] = (pos, vel)
        self.joints = joints

    def _on_odom(self, msg):
        self.rates['odom'].tick()
        self.odom_twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _on_cmd(self, msg):
        self.rates['cmd'].tick()
        self.cmd = (msg.linear.x, msg.angular.z)

    def _pose(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(frame, BASE_FRAME, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        return t.x, t.y, _yaw(tf.transform.rotation)

    # -- system ------------------------------------------------------------

    def _update_system(self):
        info = dict(self.sys)
        try:
            with open('/proc/stat') as handle:
                values = [int(v) for v in handle.readline().split()[1:]]
            idle, total = values[3] + values[4], sum(values)
            if self._cpu_prev is not None:
                d_total = total - self._cpu_prev[1]
                d_idle = idle - self._cpu_prev[0]
                info['cpu'] = 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else None
            self._cpu_prev = (idle, total)
            info['load'] = os.getloadavg()[0]
        except (OSError, ValueError, IndexError):
            pass
        try:
            mem = {}
            with open('/proc/meminfo') as handle:
                for line in handle:
                    key, value = line.split(':', 1)
                    mem[key] = int(value.split()[0])
            info['ram_used'] = (mem['MemTotal'] - mem['MemAvailable']) / 1048576.0
            info['ram_total'] = mem['MemTotal'] / 1048576.0
        except (OSError, ValueError, KeyError):
            pass
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as handle:
                info['temp'] = int(handle.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
        try:
            with open('/proc/uptime') as handle:
                info['uptime'] = float(handle.read().split()[0])
        except (OSError, ValueError):
            pass
        try:
            info['disk_free'] = shutil.disk_usage(os.path.expanduser('~')).free / 1e9
        except OSError:
            pass

        if time.time() - self._sys_slow_t > 5.0:  # the subprocess calls, less often
            self._sys_slow_t = time.time()
            try:
                out = subprocess.run(['hostname', '-I'], capture_output=True, text=True,
                                     timeout=2).stdout.split()
                usable = [a for a in out if ':' not in a
                          and not a.startswith(('172.17.', '192.0.2.'))]
                info['ip'] = usable[0] if usable else ''
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                     text=True, timeout=2).stdout
                info['throttled'] = out.strip().split('=')[-1] if '=' in out else None
            except (OSError, subprocess.SubprocessError):
                pass
        self.sys = info

    # -- snapshot for the screen ------------------------------------------

    def snapshot(self):
        layers = {}
        for name in START_ORDER:
            proc = self.procs.get(name)
            if proc is None:
                layers[name] = ('off', None)
            elif proc.alive():
                layers[name] = ('running', time.time() - proc.started_at)
            else:
                layers[name] = ('exited', None)

        log_line = ''
        for name in STOP_ORDER:  # an exited layer first, else the newest running one
            proc = self.procs.get(name)
            if proc is not None and not proc.alive():
                log_line = f'{LAYER_LABELS[name]}: {proc.last_line()}'
                break
        if not log_line:
            for name in STOP_ORDER:
                proc = self.procs.get(name)
                if proc is not None:
                    log_line = f'{LAYER_LABELS[name]}: {proc.last_line()}'
                    break

        return {
            'mode': self.mode,
            'phase': self.phase,
            'busy': self.busy,
            'estop': self.estop,
            'active_map': self.active_map,
            'slam_running': layers['slam'][0] == 'running',
            'pose_map': self._pose(MAP_FRAME),
            'pose_odom': self._pose(ODOM_FRAME),
            'joints': self.joints,
            'odom_twist': self.odom_twist if time.time() - self.rates['odom'].last < 1.0 else None,
            'cmd': self.cmd if time.time() - self.rates['cmd'].last < 1.0 else None,
            'rates': {k: m.hz() for k, m in self.rates.items()},
            'sys': self.sys,
            'layers': layers,
            'log': log_line,
            'events': list(self.events),
        }

    # -- shutdown ----------------------------------------------------------

    def shutdown_all(self):
        if self._shut_down:
            return
        self._shut_down = True
        self._abort.set()
        self._transition_lock.acquire(timeout=20.0)
        for name in STOP_ORDER:
            proc = self.procs.pop(name, None)
            if proc is not None:
                proc.stop()


# --- Screen -----------------------------------------------------------------------

class DiagColumn:
    """A titled block of key / value rows in the diagnostics strip."""

    def __init__(self, parent, title, keys, fonts):
        self.frame = tk.Frame(parent, bg=PANEL)
        tk.Label(self.frame, text=title, font=fonts['head'], bg=PANEL, fg=MUTED,
                 anchor='w').grid(row=0, column=0, columnspan=2, sticky='w')
        self.values = {}
        for row, key in enumerate(keys, start=1):
            tk.Label(self.frame, text=key, font=fonts['diag'], bg=PANEL, fg=MUTED,
                     anchor='w').grid(row=row, column=0, sticky='w', padx=(0, 8))
            value = tk.Label(self.frame, text='\u2013', font=fonts['diag'], bg=PANEL,
                             fg=INK, anchor='w')
            value.grid(row=row, column=1, sticky='w')
            self.values[key] = value

    def set(self, key, text, color=INK):
        label = self.values[key]
        if label.cget('text') != text:
            label.configure(text=text)
        if label.cget('fg') != color:
            label.configure(fg=color)


class Panel:
    REFRESH_MS = 300

    def __init__(self, root, core):
        self.root = root
        self.core = core
        self._quit_requested = False
        self.selected_map = None
        self._maps = []

        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        if not FULLSCREEN:
            width, height = 800, 480
        self.k = max(0.8, min(width / 800.0, height / 480.0))
        families = set(tkfont.families())

        def pick(*names):
            return next((n for n in names if n in families), 'TkDefaultFont')

        body = pick('Ubuntu', 'DejaVu Sans')
        condensed = pick('Ubuntu Condensed', body)
        mono = pick('Ubuntu Mono', 'DejaVu Sans Mono')

        def font(family, px, weight='normal'):
            return tkfont.Font(family=family, size=-int(round(px * self.k)), weight=weight)

        self.fonts = {
            'ip': font(body, 22, 'bold'),
            'bar': font(body, 14),
            'phase': font(body, 15),
            'tile': font(body, 21, 'bold'),
            'btn': font(body, 14),
            'map': font(body, 15, 'bold'),
            'stop': font(body, 23, 'bold'),
            'head': font(condensed, 11, 'bold'),
            'diag': font(condensed, 12),
            'log': font(mono, 12),
        }

        root.title('Robot panel')
        root.configure(bg=BG)
        if FULLSCREEN:
            root.attributes('-fullscreen', True)
        else:
            root.geometry('800x480')
        if HIDE_CURSOR:
            root.configure(cursor='none')
        root.bind('<Escape>', lambda _e: self.request_quit())
        root.protocol('WM_DELETE_WINDOW', self.request_quit)

        self._build()
        root.after(50, self._refresh)

    # -- widgets -----------------------------------------------------------

    def _button(self, parent, text, command, font, bg=WHITE, fg=INK):
        return tk.Button(parent, text=text, command=command, font=font, bg=bg, fg=fg,
                         activebackground=bg, activeforeground=fg, relief='flat', bd=0,
                         highlightthickness=1, highlightbackground=LINE,
                         highlightcolor=LINE, disabledforeground=DIM,
                         cursor='none' if HIDE_CURSOR else 'hand2')

    def _build(self):
        r = self.root
        f = self.fonts
        pad = int(8 * self.k)
        r.grid_columnconfigure(0, weight=1)
        r.grid_rowconfigure(2, weight=1)

        bar = tk.Frame(r, bg=INK)
        bar.grid(row=0, column=0, sticky='ew')
        self.lbl_ip = tk.Label(bar, text='\u2013', font=f['ip'], fg=WHITE, bg=INK)
        self.lbl_ip.pack(side='left', padx=(pad * 2, pad), pady=pad // 2)
        self.lbl_host = tk.Label(bar, font=f['bar'], fg='#A9B8C2', bg=INK)
        self.lbl_host.pack(side='left')
        self.lbl_mode = tk.Label(bar, font=f['bar'], fg=INK, bg=LINE,
                                 padx=pad * 2, pady=pad // 3)
        self.lbl_mode.pack(side='right', padx=pad * 2)

        self.lbl_phase = tk.Label(r, font=f['phase'], bg=BG, fg=INK, anchor='w',
                                  justify='left')
        self.lbl_phase.grid(row=1, column=0, sticky='ew', padx=pad * 2, pady=(pad, 0))

        main = tk.Frame(r, bg=BG)
        main.grid(row=2, column=0, sticky='nsew', padx=pad, pady=pad)
        for col, weight in enumerate((5, 5, 6, 4)):
            main.grid_columnconfigure(col, weight=weight, uniform='main')
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        self.tiles = {}
        for mode, (row, col) in {'map': (0, 0), 'explore': (0, 1),
                                 'navigate': (1, 0), 'resume': (1, 1)}.items():
            tile = self._button(main, MODE_LABELS[mode],
                                lambda m=mode: self._start(m), f['tile'])
            tile.grid(row=row, column=col, sticky='nsew', padx=pad // 2, pady=pad // 2)
            self.tiles[mode] = tile

        actions = tk.Frame(main, bg=BG)
        actions.grid(row=0, column=2, rowspan=2, sticky='nsew', padx=pad // 2)
        actions.grid_columnconfigure(1, weight=1)
        for row in range(4):
            actions.grid_rowconfigure(row, weight=1, uniform='act')

        self.btn_prev = self._button(actions, '\u25C0', lambda: self._step_map(-1), f['btn'])
        self.btn_prev.grid(row=0, column=0, sticky='nsew', pady=pad // 2)
        self.lbl_map = tk.Label(actions, font=f['map'], bg=WHITE, fg=INK,
                                highlightthickness=1, highlightbackground=LINE)
        self.lbl_map.grid(row=0, column=1, sticky='nsew', pady=pad // 2)
        self.btn_next = self._button(actions, '\u25B6', lambda: self._step_map(1), f['btn'])
        self.btn_next.grid(row=0, column=2, sticky='nsew', pady=pad // 2)

        self.btn_save = self._button(actions, 'Save map', self.core.save_map, f['btn'])
        self.btn_start = self._button(actions, 'Robot is at map start',
                                      self.core.place_at_map_start, f['btn'])
        self.btn_stopall = self._button(actions, 'Stop all',
                                        lambda: self.core.request_mode('idle'), f['btn'])
        for row, button in enumerate((self.btn_save, self.btn_start, self.btn_stopall), 1):
            button.grid(row=row, column=0, columnspan=3, sticky='nsew', pady=pad // 2)

        self.btn_estop = self._button(main, 'Stop\nrobot', self._toggle_estop,
                                      f['stop'], bg=STOP, fg=WHITE)
        self.btn_estop.configure(highlightthickness=0)
        self.btn_estop.grid(row=0, column=3, rowspan=2, sticky='nsew',
                            padx=pad // 2, pady=pad // 2)

        diag = tk.Frame(r, bg=PANEL)
        diag.grid(row=3, column=0, sticky='ew', padx=pad, pady=(0, pad))
        self.col_pose = DiagColumn(diag, 'Position', ['Map', 'Odom', 'Offset'], f)
        self.col_wheels = DiagColumn(diag, 'Wheels', ['Left', 'Right', 'Command', 'Measured'], f)
        self.col_rates = DiagColumn(diag, 'Topics', ['Lidar', 'Odom', 'Joints', 'Map'], f)
        self.col_sys = DiagColumn(diag, 'System', ['CPU', 'Temp', 'RAM', 'Disk', 'Power'], f)
        self.col_layers = DiagColumn(diag, 'Layers', [LAYER_LABELS[n] for n in START_ORDER], f)
        for col, block in enumerate((self.col_pose, self.col_wheels, self.col_rates,
                                     self.col_sys, self.col_layers)):
            diag.grid_columnconfigure(col, weight=(6, 6, 3, 4, 4)[col], uniform='diag')
            block.frame.grid(row=0, column=col, sticky='nw', padx=(pad, 0), pady=pad // 2)

        self.lbl_log = tk.Label(r, font=f['log'], bg=INK, fg='#CFE3EE', anchor='w')
        self.lbl_log.grid(row=4, column=0, sticky='ew')

    # -- actions -----------------------------------------------------------

    def _start(self, mode):
        self.core.request_mode(mode, self.selected_map)

    def _step_map(self, step):
        names = [m['name'] for m in self._maps]
        if not names:
            return
        index = names.index(self.selected_map) if self.selected_map in names else 0
        self.selected_map = names[(index + step) % len(names)]
        self._render_map_picker()

    def _toggle_estop(self):
        self.core.set_estop(not self.core.estop)
        self._render_estop(self.core.estop)

    def request_quit(self):
        self._quit_requested = True

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _set(widget, **options):
        changed = {k: v for k, v in options.items() if str(widget.cget(k)) != str(v)}
        if changed:
            widget.configure(**changed)

    def _render_map_picker(self):
        entry = next((m for m in self._maps if m['name'] == self.selected_map), None)
        if entry is None:
            self._set(self.lbl_map, text='No saved maps', fg=DIM)
        else:
            self._set(self.lbl_map, text=entry['name'], fg=INK)
        state = 'normal' if len(self._maps) > 1 else 'disabled'
        self._set(self.btn_prev, state=state)
        self._set(self.btn_next, state=state)

    def _render_estop(self, engaged):
        if engaged:
            self._set(self.btn_estop, text='Release\nstop', bg=HAZARD, fg=INK,
                      activebackground=HAZARD, activeforeground=INK)
        else:
            self._set(self.btn_estop, text='Stop\nrobot', bg=STOP, fg=WHITE,
                      activebackground=STOP, activeforeground=WHITE)

    @staticmethod
    def _fmt_pose(pose):
        if pose is None:
            return '\u2013'
        x, y, yaw = pose
        return f'{x:+.2f}  {y:+.2f}  {math.degrees(yaw):+.0f}\u00b0'

    def _refresh(self):
        if self._quit_requested:
            self._do_quit()
            return
        try:
            self._render(self.core.snapshot())
        except Exception as exc:  # noqa: BLE001  (never let the screen die)
            self._set(self.lbl_log, text=f'Screen update error: {exc}')
        self.root.after(self.REFRESH_MS, self._refresh)

    def _render(self, s):
        sysinfo = s['sys']
        self._set(self.lbl_ip, text=sysinfo.get('ip') or 'No network')
        self._set(self.lbl_host, text=sysinfo.get('host', ''))
        if s['mode'] == 'idle':
            self._set(self.lbl_mode, text='Idle', bg=LINE, fg=INK)
        else:
            self._set(self.lbl_mode, text=MODE_LABELS[s['mode']], bg=HIVIS, fg=INK)

        # Status line: a recent warning or error wins over the normal phase.
        recent = next((e for e in s['events'] if e['level'] in ('warn', 'error')
                       and time.time() - e['wall'] < 12.0), None)
        if recent is not None:
            color = STOP if recent['level'] == 'error' else WARN
            self._set(self.lbl_phase, text=recent['text'], fg=color)
        else:
            self._set(self.lbl_phase, text=s['phase'], fg=INK)
        self._set(self.lbl_phase, wraplength=max(200, self.root.winfo_width() - 40))

        # Map picker
        self._maps = self.core.list_maps()
        names = [m['name'] for m in self._maps]
        if self.selected_map not in names:
            self.selected_map = s['active_map'] if s['active_map'] in names else (
                names[0] if names else None)
        self._render_map_picker()
        selected = next((m for m in self._maps if m['name'] == self.selected_map), None)

        # Mode tiles
        for mode, tile in self.tiles.items():
            enabled = not s['busy']
            if mode == 'navigate':
                enabled = enabled and selected is not None
            if mode == 'resume':
                enabled = enabled and selected is not None and selected['resumable']
            active = s['mode'] == mode
            self._set(tile, state='normal' if enabled else 'disabled',
                      bg=INK if active else WHITE, fg=WHITE if active else INK,
                      disabledforeground='#B8C4CC' if active else DIM,
                      activebackground=INK if active else WHITE,
                      activeforeground=WHITE if active else INK)

        self._set(self.btn_save, state='normal' if s['slam_running'] and not s['busy'] else 'disabled')
        self._set(self.btn_start, state='normal' if s['mode'] == 'navigate' else 'disabled')
        self._set(self.btn_stopall, state='normal' if s['mode'] != 'idle' or s['busy'] else 'disabled')
        self._render_estop(s['estop'])

        # Position
        pm, po = s['pose_map'], s['pose_odom']
        self.col_pose.set('Map', self._fmt_pose(pm))
        self.col_pose.set('Odom', self._fmt_pose(po))
        if pm is not None and po is not None:
            dist = math.hypot(pm[0] - po[0], pm[1] - po[1])
            dyaw = math.degrees(_wrap_angle(pm[2] - po[2]))
            self.col_pose.set('Offset', f'{dist:.2f} m  {dyaw:+.0f}\u00b0')
        else:
            self.col_pose.set('Offset', '\u2013')

        # Wheels
        for key, joint in (('Left', LEFT_JOINT), ('Right', RIGHT_JOINT)):
            pos, vel = s['joints'].get(joint, (None, None))
            if pos is None:
                self.col_wheels.set(key, '\u2013')
                continue
            ticks = pos / (2.0 * math.pi) * ENCODER_CPR
            speed = f'  {vel * WHEEL_RADIUS_M:+.2f} m/s' if vel is not None else ''
            self.col_wheels.set(key, f'{ticks:.0f} tk{speed}')
        for key, pair in (('Command', s['cmd']), ('Measured', s['odom_twist'])):
            if pair is None:
                self.col_wheels.set(key, '\u2013')
            else:
                self.col_wheels.set(key, f'v {pair[0]:+.2f}  \u03c9 {pair[1]:+.2f}')

        # Topic rates
        for key, rate_key, healthy in (('Lidar', 'scan', 8.0), ('Odom', 'odom', 20.0),
                                       ('Joints', 'joints', 20.0), ('Map', 'map', 0.0)):
            hz = s['rates'][rate_key]
            if hz <= 0.0:
                self.col_rates.set(key, 'off', DIM)
            else:
                self.col_rates.set(key, f'{hz:.1f} Hz', INK if hz >= healthy else WARN)

        # System
        cpu = sysinfo.get('cpu')
        self.col_sys.set('CPU', f'{cpu:.0f} %' if cpu is not None else '\u2013',
                         WARN if cpu is not None and cpu > 85 else INK)
        temp = sysinfo.get('temp')
        self.col_sys.set('Temp', f'{temp:.0f} \u00b0C' if temp is not None else '\u2013',
                         WARN if temp is not None and temp >= 75 else INK)
        if sysinfo.get('ram_total'):
            self.col_sys.set('RAM', f'{sysinfo["ram_used"]:.1f}/{sysinfo["ram_total"]:.1f} GB')
        disk = sysinfo.get('disk_free')
        self.col_sys.set('Disk', f'{disk:.0f} GB free' if disk is not None else '\u2013',
                         WARN if disk is not None and disk < 2 else INK)
        power, power_color = 'OK', INK
        raw = sysinfo.get('throttled')
        try:
            flags = int(raw, 16) if raw else None
        except ValueError:
            flags = None
        if flags is None:
            power, power_color = '\u2013', DIM
        elif flags & 0x1:
            power, power_color = 'Low now', STOP
        elif flags & 0x4:
            power, power_color = 'Throttled', WARN
        elif flags & 0x10000:
            power, power_color = 'Was low', WARN
        self.col_sys.set('Power', power, power_color)

        # Layers
        for name in START_ORDER:
            state, uptime = s['layers'][name]
            if state == 'running':
                minutes = int(uptime // 60)
                text = f'on {minutes} min' if minutes else 'on'
                self.col_layers.set(LAYER_LABELS[name], text, OK)
            elif state == 'exited':
                self.col_layers.set(LAYER_LABELS[name], 'exited', STOP)
            else:
                self.col_layers.set(LAYER_LABELS[name], 'off', DIM)

        self._set(self.lbl_log, text=f'  {s["log"]}' if s['log'] else '  No layers running.')

    def _do_quit(self):
        self._set(self.lbl_phase, text='Stopping everything...', fg=INK)
        self.root.update_idletasks()
        self.core.shutdown_all()
        self.root.destroy()


# --- Entry point --------------------------------------------------------------------

def _single_instance_lock():
    os.makedirs(RUN_DIR, exist_ok=True)
    handle = open(os.path.join(RUN_DIR, 'panel.lock'), 'w')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('The robot panel is already running. Close it before starting another.')
    return handle  # keep the handle open for the life of the process


def main():
    lock = _single_instance_lock()  # noqa: F841  (held until exit)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        sys.exit(f'No display available ({exc}). From ssh, start it with ui/start_panel.sh, '
                 'which points it at the robot\'s screen.')

    rclpy.init()
    core = RobotCore()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(core)
    threading.Thread(target=executor.spin, daemon=True).start()

    panel = Panel(root, core)

    def _on_signal(*_):
        panel.request_quit()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        root.mainloop()
    finally:
        core.shutdown_all()
        executor.shutdown()
        core.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
