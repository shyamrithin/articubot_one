#!/usr/bin/env python3
# =============================================================================
# File:        ui/robot_supervisor.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Back end for the robot's touchscreen / phone control page. One process that:
#
#   1. Serves the web page in ui/web/ on http://<robot>:8080
#   2. Starts rosbridge (ws://<robot>:9090) so the page can talk to ROS
#   3. Starts and stops the robot stack in "modes", replacing the tmux windows:
#
#        Map       drivetrain + lidar + SLAM             (drive by hand, save map)
#        Explore   drivetrain + lidar + SLAM + Nav2      (tap goals in unknown space)
#        Navigate  drivetrain + lidar + AMCL + Nav2      (saved map, tap goals)
#
#      Each layer is one `ros2 launch` in its own process group. Layers are
#      started in order and each is checked for readiness before the next:
#        drivetrain   -> /joint_states arriving
#        lidar        -> /scan arriving
#        SLAM         -> /map arriving
#        localisation -> map_server and amcl lifecycle state ACTIVE (one retry)
#        navigation   -> bt_navigator lifecycle state ACTIVE
#      This sequencing is what avoids the lifecycle-manager timeout hit when
#      localisation and Nav2 were started together by hand.
#
#   4. Turns page commands into ROS actions: Nav2 goals (with progress
#      feedback), AMCL initial pose, map saving, and a software e-stop.
#
# TOPICS
# ------
#   Subscribes  /ui/command    std_msgs/String   JSON commands from the page
#               /map /plan /scan /joint_states   and TF map -> base_footprint
#   Publishes   /ui/status     std_msgs/String   JSON, 1 Hz  (mode, health, events)
#               /ui/live       std_msgs/String   JSON, 5 Hz  (pose, scan points, goal)
#               /ui/map        std_msgs/String   JSON, on change + every 3 s
#                                                (grid as base64 bytes, ~4x smaller
#                                                 than a raw OccupancyGrid in JSON)
#               /ui/plan       std_msgs/String   JSON, downsampled global plan
#               /cmd_vel_estop geometry_msgs/Twist  zero twist at 20 Hz while the
#                                                   stop is engaged (twist_mux 255)
#               /initialpose   for AMCL
#
# COMMANDS (JSON on /ui/command)
# ------------------------------
#   {"cmd":"start","mode":"map|explore|navigate","map":"room_v2"}
#   {"cmd":"stop"}                         stop every layer (also aborts start-up)
#   {"cmd":"save_map","name":"optional"}   needs SLAM running
#   {"cmd":"estop","on":true|false}
#   {"cmd":"goal","x":..,"y":..,"yaw":..}
#   {"cmd":"cancel"}
#   {"cmd":"initial_pose","x":..,"y":..,"yaw":..}
#
# LOGS
# ----
#   Each layer's output goes to ~/.ros/robot_ui_logs/<layer>.log. The status
#   message carries the last line of each, so failures show up on the page.
#
# CAVEATS
# -------
#   - Don't also start the stack by hand in tmux while this is running; two
#     drivetrains fighting over one serial port fails in confusing ways.
#   - The map view assumes the map origin has zero yaw, which holds for
#     slam_toolbox and map_saver output.
#   - rosbridge has no authentication. Anyone on the same network who opens
#     the page can drive the robot.
# =============================================================================

import array
import base64
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String

# --- Configuration ------------------------------------------------------------

PACKAGE = 'articubot_one'
UI_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(UI_DIR, 'web')
MAP_DIR = os.path.expanduser(os.environ.get('ROBOT_MAP_DIR', '~/maps'))
LOG_DIR = os.path.expanduser('~/.ros/robot_ui_logs')
HTTP_PORT = int(os.environ.get('ROBOT_UI_PORT', '8080'))
BRIDGE_PORT = int(os.environ.get('ROBOT_BRIDGE_PORT', '9090'))

MAP_FRAME = 'map'
BASE_FRAME = 'base_footprint'


def _launch(*args):
    return ['ros2', 'launch', PACKAGE, *args]


LAYER_COMMANDS = {
    'drive': _launch('launch_robot.launch.py'),
    'lidar': _launch('rplidar.launch.py'),
    'slam': _launch('online_async_launch.py', 'use_sim_time:=false'),
    'navigation': _launch('navigation_launch.py', 'use_sim_time:=false',
                          'map_subscribe_transient_local:=true'),
    # 'localization' is built per map in _layer_command().
}

LAYER_LABELS = {
    'drive': 'drivetrain',
    'lidar': 'lidar',
    'slam': 'mapping',
    'localization': 'localisation',
    'navigation': 'navigation',
}

MODES = {
    'map': ['drive', 'lidar', 'slam'],
    'explore': ['drive', 'lidar', 'slam', 'navigation'],
    'navigate': ['drive', 'lidar', 'localization', 'navigation'],
}
MODE_LABELS = {'map': 'Map', 'explore': 'Explore', 'navigate': 'Navigate'}

START_ORDER = ['drive', 'lidar', 'slam', 'localization', 'navigation']
STOP_ORDER = list(reversed(START_ORDER))

# Seconds to wait for each layer to report ready.
READY_TIMEOUT_S = {
    'drive': 25.0,
    'lidar': 20.0,
    'slam': 30.0,
    'localization': 45.0,
    'navigation': 90.0,
}
# In Navigate mode Nav2 cannot finish activating until someone sets the
# robot's position, so the wait is long and shows a hint instead.
NAV_WAIT_FOR_POSE_S = 600.0

SAFE_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Aborted(Exception):
    """Raised inside a start-up wait when Stop all is pressed."""


# --- Process management -------------------------------------------------------

class ManagedProcess:
    """One command (usually `ros2 launch`) running in its own process group."""

    def __init__(self, name, cmd):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f'{name}.log')
        self._log = None

    def start(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log = open(self.log_path, 'w', buffering=1)
        self._log.write(f'$ {" ".join(self.cmd)}\n')
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group, so we can stop all of it
        )

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace_s=12.0):
        """SIGINT (lets ros2 launch shut its nodes down), then TERM, then KILL."""
        if self.proc is None:
            return
        pgid = self.proc.pid  # equals the group id because of start_new_session
        if self.proc.poll() is None:
            for sig, wait_s in ((signal.SIGINT, grace_s),
                                (signal.SIGTERM, 5.0),
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
        # Sweep any node that outlived its launch process.
        try:
            os.killpg(pgid, signal.SIGKILL)
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
        return ANSI_ESCAPE.sub('', lines[-1])[-180:] if lines else ''


# --- Supervisor node ----------------------------------------------------------

class RobotSupervisor(Node):

    def __init__(self):
        super().__init__('robot_ui_supervisor')
        self._cb = ReentrantCallbackGroup()

        # Mode state. The lock is held for the whole of a mode change.
        self._transition_lock = threading.Lock()
        self._abort = threading.Event()
        self.procs = {}
        self.bridge = None
        self.mode = 'idle'
        self.phase = 'Idle. Choose a mode to start.'
        self.busy = False
        self.active_map = None
        self.events = deque(maxlen=8)

        self.estop = False

        # Navigation goal tracking. _goal_seq lets late callbacks from a
        # replaced or cancelled goal be ignored.
        self._goal_handle = None
        self._goal_seq = 0
        self.nav_state = 'idle'
        self.nav_remaining = None
        self.goal_xy_yaw = None

        # Sensor bookkeeping.
        self._scan_times = deque(maxlen=60)
        self._last_scan = None
        self._last_scan_wall = 0.0
        self._last_joint_state = 0.0
        self._map_wall_time = 0.0
        self._map_payload = None
        self._sys_cache = {'t': 0.0, 'ip': '', 'temp_c': None, 'throttled': None}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_pub = self.create_publisher(String, '/ui/status', 10)
        self.live_pub = self.create_publisher(String, '/ui/live', 10)
        self.map_pub = self.create_publisher(String, '/ui/map', 10)
        self.plan_pub = self.create_publisher(String, '/ui/plan', 10)
        self.estop_pub = self.create_publisher(Twist, '/cmd_vel_estop', 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/ui/command', self._on_command, 10,
                                 callback_group=self._cb)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, latched,
                                 callback_group=self._cb)
        self.create_subscription(Path, '/plan', self._on_plan, 10,
                                 callback_group=self._cb)
        self.create_subscription(LaserScan, '/scan', self._on_scan,
                                 qos_profile_sensor_data, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state,
                                 qos_profile_sensor_data, callback_group=self._cb)

        self._state_clients = {
            name: self.create_client(GetState, f'/{name}/get_state',
                                     callback_group=self._cb)
            for name in ('map_server', 'amcl', 'bt_navigator')
        }
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                       callback_group=self._cb)

        self.create_timer(1.0, self._publish_status, callback_group=self._cb)
        self.create_timer(0.2, self._publish_live, callback_group=self._cb)
        self.create_timer(3.0, self._republish_map, callback_group=self._cb)
        self.create_timer(0.05, self._publish_estop, callback_group=self._cb)

        self.bridge = ManagedProcess('rosbridge', [
            'ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml',
            f'port:={BRIDGE_PORT}'])
        self.bridge.start()
        self._event('Robot UI ready.')

    # -- events ------------------------------------------------------------

    def _event(self, text, level='info'):
        self.events.appendleft({'t': datetime.now().strftime('%H:%M:%S'),
                                'level': level, 'text': text})
        logger = self.get_logger()
        if level == 'error':
            logger.error(text)
        elif level == 'warn':
            logger.warning(text)
        else:
            logger.info(text)

    # -- commands ----------------------------------------------------------

    def _on_command(self, msg):
        try:
            cmd = json.loads(msg.data)
            kind = cmd.get('cmd')
            if kind == 'start':
                self.request_mode(str(cmd.get('mode', '')), cmd.get('map'))
            elif kind == 'stop':
                self.request_mode('idle')
            elif kind == 'save_map':
                threading.Thread(target=self._save_map, args=(cmd.get('name'),),
                                 daemon=True).start()
            elif kind == 'estop':
                self._set_estop(bool(cmd.get('on')))
            elif kind == 'goal':
                self._send_goal(float(cmd['x']), float(cmd['y']),
                                float(cmd.get('yaw', 0.0)))
            elif kind == 'cancel':
                self._cancel_goal('Goal cancelled.')
            elif kind == 'initial_pose':
                self._set_initial_pose(float(cmd['x']), float(cmd['y']),
                                       float(cmd.get('yaw', 0.0)))
            else:
                self._event(f'Unknown command "{kind}".', 'error')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self._event(f'Ignored a malformed command ({exc}).', 'error')

    # -- modes -------------------------------------------------------------

    def request_mode(self, target, map_name=None):
        if target != 'idle' and target not in MODES:
            self._event(f'Unknown mode "{target}".', 'error')
            return
        map_path = None
        if target == 'navigate':
            map_path = self._map_yaml(map_name)
            if map_path is None:
                self._event('Pick a saved map before starting Navigate.', 'error')
                return
        if target == 'idle':
            # Interrupt any start-up in progress, then stop once it lets go.
            self._abort.set()
            threading.Thread(target=self._run_transition, args=('idle', None, True),
                             daemon=True).start()
            return
        if not self._transition_lock.acquire(blocking=False):
            self._event('Still changing modes. Wait for it, or press Stop all.', 'warn')
            return
        threading.Thread(target=self._run_transition, args=(target, map_path, False),
                         daemon=True).start()

    def _run_transition(self, target, map_path, need_lock):
        if need_lock:
            self._transition_lock.acquire()
        self._abort.clear()
        self.busy = True
        try:
            self._transition(target, map_path)
        except Aborted:
            self._event('Start-up interrupted.', 'warn')
        except Exception as exc:  # noqa: BLE001  (surface anything to the page)
            self.phase = f'Could not start: {exc}'
            self._event(f'Mode change failed: {exc}', 'error')
        finally:
            self.busy = False
            self._transition_lock.release()

    def _transition(self, target, map_path):
        wanted = [] if target == 'idle' else MODES[target]
        # Nav2 must restart when its map source changes (SLAM <-> saved map),
        # and localisation must restart when the map file changes.
        restart_nav = target != self.mode
        restart_loc = map_path != self.active_map

        for name in STOP_ORDER:
            if name not in self.procs:
                continue
            must_stop = (name not in wanted
                         or (name == 'navigation' and restart_nav)
                         or (name == 'localization' and restart_loc))
            if must_stop:
                if name == 'navigation':
                    self._cancel_goal(None)
                self.phase = f'Stopping {LAYER_LABELS[name]}...'
                self.procs.pop(name).stop()

        if target == 'idle':
            self.mode = 'idle'
            self.active_map = None
            self.phase = 'Idle. Choose a mode to start.'
            self._event('Everything stopped.')
            return

        self.mode = target
        self.active_map = map_path
        for name in START_ORDER:
            if name not in wanted:
                continue
            proc = self.procs.get(name)
            if proc is not None and proc.alive():
                continue
            self._start_layer(name, map_path)

        if target == 'map':
            self.phase = 'Mapping. Drive around slowly, then save the map.'
        elif target == 'explore':
            self.phase = 'Exploring. Tap the map to send the robot somewhere.'
        else:
            self.phase = (f'Navigating on {self._map_name(map_path)}. '
                          'Tap the map to send a goal.')
        self._event(f'{MODE_LABELS[target]} mode ready.')

    def _layer_command(self, name, map_path):
        if name == 'localization':
            return _launch('localization_launch.py', 'use_sim_time:=false',
                           f'map:={map_path}')
        return LAYER_COMMANDS[name]

    def _start_layer(self, name, map_path):
        old = self.procs.pop(name, None)
        if old is not None:
            old.stop()
        label = LAYER_LABELS[name]
        waiting_on_person = name == 'navigation' and self.mode == 'navigate'
        attempts = 1 if waiting_on_person else 2

        for attempt in range(1, attempts + 1):
            self.phase = f'Starting {label}...' if attempt == 1 else f'Retrying {label}...'
            proc = ManagedProcess(name, self._layer_command(name, map_path))
            proc.start()
            self.procs[name] = proc
            if self._wait_ready(name, proc, time.time()):
                return
            if attempt < attempts:
                self._event(f'{label.capitalize()} did not come up; retrying once.', 'warn')
                self.procs.pop(name).stop()
        raise RuntimeError(f'{label} did not start. Log: {proc.log_path}')

    def _wait_ready(self, name, proc, started):
        if name == 'drive':
            return self._wait(lambda: self._last_joint_state > started,
                              READY_TIMEOUT_S['drive'], proc)
        if name == 'lidar':
            return self._wait(lambda: self._last_scan_wall > started,
                              READY_TIMEOUT_S['lidar'], proc)
        if name == 'slam':
            return self._wait(lambda: self._map_wall_time > started,
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
                self._event(f'{LAYER_LABELS[proc.name].capitalize()} exited: '
                            f'{proc.last_line()}', 'error')
                return False
            if condition():
                self.phase = base_phase
                return True
            if pose_hint and not self._localized():
                self.phase = ('Set the robot\u2019s position on the map to finish '
                              'starting navigation.')
            time.sleep(0.25)
        return False

    def _node_active(self, node_name):
        client = self._state_clients[node_name]
        if not client.service_is_ready():
            return False
        future = client.call_async(GetState.Request())
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            future.cancel()
            return False
        result = future.result()
        return result is not None and result.current_state.id == State.PRIMARY_STATE_ACTIVE

    # -- maps --------------------------------------------------------------

    @staticmethod
    def _map_name(path):
        return os.path.basename(path)[:-5] if path else None

    def _list_maps(self):
        try:
            entries = [e for e in os.listdir(MAP_DIR) if e.endswith('.yaml')]
        except FileNotFoundError:
            return []
        entries.sort(key=lambda e: os.path.getmtime(os.path.join(MAP_DIR, e)),
                     reverse=True)
        return [e[:-5] for e in entries]

    def _map_yaml(self, name):
        if not name or not SAFE_NAME.fullmatch(str(name)):
            return None
        path = os.path.join(MAP_DIR, f'{name}.yaml')
        return path if os.path.isfile(path) else None

    def _save_map(self, name):
        slam = self.procs.get('slam')
        if slam is None or not slam.alive():
            self._event('Start Map or Explore first; there is no live map to save.',
                        'error')
            return
        name = (str(name).strip() if name else '') or datetime.now().strftime('map_%Y%m%d_%H%M')
        if not SAFE_NAME.fullmatch(name):
            self._event('Map names can use letters, numbers, - and _ only.', 'error')
            return
        os.makedirs(MAP_DIR, exist_ok=True)
        candidate, suffix = name, 2
        while os.path.exists(os.path.join(MAP_DIR, f'{candidate}.yaml')):
            candidate, suffix = f'{name}_{suffix}', suffix + 1
        self._event(f'Saving map as {candidate}...')
        try:
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', os.path.join(MAP_DIR, candidate)],
                capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self._event('Saving the map timed out.', 'error')
            return
        if result.returncode == 0 and os.path.isfile(os.path.join(MAP_DIR, f'{candidate}.yaml')):
            self._event(f'Saved map {candidate}.')
        else:
            tail = (result.stderr or result.stdout).strip().splitlines()
            self._event(f'Map save failed: {tail[-1] if tail else "no output"}', 'error')

    # -- safety ------------------------------------------------------------

    def _set_estop(self, on):
        if on and not self.estop:
            self.estop = True
            self._cancel_goal(None)
            self._event('Stop engaged. The robot will not move until you release it.',
                        'warn')
        elif not on and self.estop:
            self.estop = False
            self._event('Stop released.')

    def _publish_estop(self):
        if self.estop:
            self.estop_pub.publish(Twist())

    # -- navigation --------------------------------------------------------

    def _send_goal(self, x, y, yaw):
        if self.estop:
            self._event('Release the stop before sending a goal.', 'warn')
            return
        if not self.nav_client.server_is_ready():
            self._event('Navigation is not running. Start Explore or Navigate first.',
                        'error')
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = MAP_FRAME
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._goal_seq += 1
        seq = self._goal_seq
        self.goal_xy_yaw = (round(x, 3), round(y, 3), round(yaw, 3))
        self.nav_state = 'sending'
        self.nav_remaining = None
        future = self.nav_client.send_goal_async(
            goal, feedback_callback=partial(self._on_nav_feedback, seq))
        future.add_done_callback(partial(self._on_goal_response, seq))

    def _on_goal_response(self, seq, future):
        handle = future.result()
        if seq != self._goal_seq:
            # Cancelled or replaced while this goal was in flight.
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            return
        if handle is None or not handle.accepted:
            self.nav_state = 'rejected'
            self.goal_xy_yaw = None
            self._event('Navigation rejected that goal. Pick a spot in open space.', 'warn')
            return
        self._goal_handle = handle
        self.nav_state = 'driving'
        handle.get_result_async().add_done_callback(partial(self._on_goal_result, seq))

    def _on_nav_feedback(self, seq, feedback_msg):
        if seq == self._goal_seq:
            self.nav_remaining = float(feedback_msg.feedback.distance_remaining)

    def _on_goal_result(self, seq, future):
        if seq != self._goal_seq:
            return
        status = future.result().status
        self._goal_handle = None
        self.nav_remaining = None
        self.plan_pub.publish(String(data=json.dumps({'points': []})))
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_state = 'arrived'
            self.goal_xy_yaw = None
            self._event('Arrived at the goal.')
        elif status == GoalStatus.STATUS_CANCELED:
            self.nav_state = 'cancelled'
        else:
            self.nav_state = 'failed'
            self._event('Navigation could not reach the goal.', 'warn')

    def _cancel_goal(self, reason):
        handle = self._goal_handle
        was_active = handle is not None or self.nav_state == 'sending'
        self._goal_seq += 1  # ignore anything still in flight
        self._goal_handle = None
        if handle is not None:
            handle.cancel_goal_async()
        if was_active:
            self.nav_state = 'cancelled'
            self.nav_remaining = None
            self.goal_xy_yaw = None
            self.plan_pub.publish(String(data=json.dumps({'points': []})))
            if reason:
                self._event(reason)

    def _set_initial_pose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        covariance = [0.0] * 36
        covariance[0] = 0.25   # x   (same defaults as RViz's 2D Pose Estimate)
        covariance[7] = 0.25   # y
        covariance[35] = 0.0685  # yaw
        msg.pose.covariance = covariance
        self.initial_pose_pub.publish(msg)
        self._event('Position set. Drive a little so the scan settles onto the walls.')

    # -- sensor callbacks --------------------------------------------------

    def _on_map(self, msg):
        info = msg.info
        data = msg.data
        raw = data.tobytes() if isinstance(data, array.array) else array.array('b', data).tobytes()
        # Bytes are two's complement, so unknown (-1) arrives as 255.
        self._map_payload = json.dumps({
            'w': info.width,
            'h': info.height,
            'res': info.resolution,
            'ox': info.origin.position.x,
            'oy': info.origin.position.y,
            'stamp': time.time(),
            'data': base64.b64encode(raw).decode('ascii'),
        })
        self._map_wall_time = time.time()
        self.map_pub.publish(String(data=self._map_payload))

    def _republish_map(self):
        # Volatile topic, so new page loads pick the map up within 3 s.
        if self._map_payload is not None:
            self.map_pub.publish(String(data=self._map_payload))

    def _on_plan(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        step = max(1, len(points) // 200)
        sampled = points[::step]
        if points and sampled[-1] != points[-1]:
            sampled.append(points[-1])
        self.plan_pub.publish(String(data=json.dumps(
            {'points': [[round(x, 3), round(y, 3)] for x, y in sampled]})))

    def _on_scan(self, msg):
        now = time.time()
        self._scan_times.append(now)
        self._last_scan = msg
        self._last_scan_wall = now

    def _on_joint_state(self, _msg):
        self._last_joint_state = time.time()

    # -- live view ---------------------------------------------------------

    def _lookup_2d(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        return t.x, t.y, _yaw(tf.transform.rotation)

    def _localized(self):
        return self._lookup_2d(MAP_FRAME, BASE_FRAME) is not None

    @staticmethod
    def _project_scan(scan, pose, max_points=240):
        x0, y0, yaw0 = pose
        c, s = math.cos(yaw0), math.sin(yaw0)
        step = max(1, len(scan.ranges) // max_points)
        points = []
        for i in range(0, len(scan.ranges), step):
            r = scan.ranges[i]
            if not scan.range_min < r < scan.range_max:  # also drops inf / nan
                continue
            a = scan.angle_min + i * scan.angle_increment
            lx, ly = r * math.cos(a), r * math.sin(a)
            points.append([round(x0 + c * lx - s * ly, 2),
                           round(y0 + s * lx + c * ly, 2)])
        return points

    def _publish_live(self):
        live = {'pose': None, 'scan': [], 'goal': list(self.goal_xy_yaw) if self.goal_xy_yaw else None}
        pose = self._lookup_2d(MAP_FRAME, BASE_FRAME)
        if pose is not None:
            live['pose'] = [round(v, 3) for v in pose]
        scan = self._last_scan
        if scan is not None and time.time() - self._last_scan_wall < 1.0:
            laser = self._lookup_2d(MAP_FRAME, scan.header.frame_id)
            if laser is not None:
                live['scan'] = self._project_scan(scan, laser)
        self.live_pub.publish(String(data=json.dumps(live)))

    # -- status ------------------------------------------------------------

    def _scan_rate(self):
        now = time.time()
        recent = [t for t in self._scan_times if now - t < 3.0]
        if len(recent) < 2:
            return 0.0
        return round((len(recent) - 1) / (recent[-1] - recent[0]), 1)

    def _system_info(self):
        now = time.time()
        if now - self._sys_cache['t'] < 5.0:
            return self._sys_cache
        info = {'t': now, 'ip': '', 'temp_c': None, 'throttled': None}
        try:
            out = subprocess.run(['hostname', '-I'], capture_output=True,
                                 text=True, timeout=2).stdout.split()
            # Skip IPv6, Docker's bridge, and the Zephyr test interface.
            usable = [a for a in out if ':' not in a
                      and not a.startswith(('172.17.', '192.0.2.'))]
            info['ip'] = usable[0] if usable else ''
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as handle:
                info['temp_c'] = round(int(handle.read().strip()) / 1000.0, 1)
        except (OSError, ValueError):
            pass
        try:
            out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                 text=True, timeout=2).stdout
            if '=' in out:
                info['throttled'] = out.strip().split('=')[-1]
        except (OSError, subprocess.SubprocessError):
            pass
        self._sys_cache = info
        return info

    def _layer_running(self, name):
        proc = self.procs.get(name)
        return proc is not None and proc.alive()

    def _publish_status(self):
        info = self._system_info()
        layers = {}
        for name in START_ORDER:
            proc = self.procs.get(name)
            if proc is not None:
                layers[name] = {'running': proc.alive(), 'last': proc.last_line()}
        status = {
            'mode': self.mode,
            'phase': self.phase,
            'busy': self.busy,
            'estop': self.estop,
            'hostname': socket.gethostname(),
            'ip': info['ip'],
            'http_port': HTTP_PORT,
            'temp_c': info['temp_c'],
            'throttled': info['throttled'],
            'scan_hz': self._scan_rate(),
            'localized': self._localized(),
            'maps': self._list_maps(),
            'active_map': self._map_name(self.active_map),
            'slam_running': self._layer_running('slam'),
            'nav_running': self._layer_running('navigation'),
            'nav': {'state': self.nav_state,
                    'remaining': (round(self.nav_remaining, 2)
                                  if self.nav_remaining is not None else None)},
            'layers': layers,
            'events': list(self.events),
        }
        self.status_pub.publish(String(data=json.dumps(status)))

    # -- shutdown ----------------------------------------------------------

    def shutdown_all(self):
        self._abort.set()
        self._transition_lock.acquire(timeout=20.0)
        for name in STOP_ORDER:
            proc = self.procs.pop(name, None)
            if proc is not None:
                proc.stop()
        if self.bridge is not None:
            self.bridge.stop()


# --- Web server ---------------------------------------------------------------

class _WebHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the journal quiet
        pass


def _start_web_server():
    server = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), _WebHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    rclpy.init()
    node = RobotSupervisor()
    web = _start_web_server()
    node.get_logger().info(
        f'Open http://{socket.gethostname()}.local:{HTTP_PORT} (or http://localhost:{HTTP_PORT} on the robot)')

    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    def _on_sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown_all()
        web.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
