#!/usr/bin/env python3
# =============================================================================
# File:        grabber/grabber_teleop.py
# Package:     articubot_one (standalone script, no colcon build step needed)
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Keyboard control for the grabber, with grabber_node.py running. Stays
# running, so every key press goes out immediately instead of waiting for a
# new `ros2 topic pub` / `ros2 service call` process to start each time.
#
# KEYS
# ----
#   a / d   or  Left / Right    move by the current step (- / +)
#   1  2  3                      step size: 1, 5 or 15 degrees
#   o / c                        open / close (the node's open_deg / closed_deg)
#   h                            go to 90 degrees (centre)
#   m                            remember the current angle as OPEN
#   n                            remember the current angle as CLOSED
#   q   or  Ctrl-C               quit (prints what you remembered)
#
# FINDING THE GRABBER'S LIMITS
# ----------------------------
#   1. Press h, then use a / d to swing the jaws.
#   2. When fully open, press m. When the jaws just meet, press n.
#   3. On quit it prints the values, backed off 4 degrees so the servo never
#      strains against the jaws, as --ros-args ready to paste.
#
# The angle shown is the last one SENT. The node goes limp idle_release_s
# after each move (default 3 s); the jaws may sag a little when it does.
#
# RUN (in its own ssh session, with grabber_node.py running)
# --------------------------------------------------------
#   python3 ~/robot_ws/src/articubot_one/grabber/grabber_teleop.py
# =============================================================================

import select
import shutil
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

RANGE_DEG = 180.0
STEPS = {'1': 1.0, '2': 5.0, '3': 15.0}
BACK_OFF_DEG = 4.0
ARROWS = {'\x1b[D': 'a', '\x1b[C': 'd'}  # Left, Right


def apply_key(key, angle, step):
    """Pure key handling. Returns (new_angle or None, new_step, action)."""
    key = ARROWS.get(key, key)
    if key in STEPS:
        return None, STEPS[key], 'step'
    if key == 'a':
        return max(0.0, angle - step), step, 'move'
    if key == 'd':
        return min(RANGE_DEG, angle + step), step, 'move'
    if key == 'h':
        return 90.0, step, 'move'
    action = {'o': 'open', 'c': 'close', 'm': 'mark_open', 'n': 'mark_closed',
              'q': 'quit', '\x03': 'quit'}.get(key, 'none')
    return None, step, action


def suggested_limits(open_mark, closed_mark):
    """Back each mark off toward the other so the servo never pushes the jaws."""
    if open_mark is None or closed_mark is None:
        return open_mark, closed_mark
    direction = 1.0 if open_mark > closed_mark else -1.0
    return open_mark - direction * BACK_OFF_DEG, closed_mark + direction * BACK_OFF_DEG


class GrabberTeleop(Node):

    def __init__(self):
        super().__init__('grabber_teleop')
        self.pub = self.create_publisher(Float32, '/grabber/angle', 10)
        self.open_client = self.create_client(Trigger, '/grabber/open')
        self.close_client = self.create_client(Trigger, '/grabber/close')
        self.create_subscription(Float32, '/grabber/state', self._on_state, 10)
        self.angle = 90.0
        self.step = 5.0
        self._last_key_time = 0.0
        self._synced = False
        self.open_mark = None
        self.closed_mark = None
        self.status = 'Ready. Press h to centre, a / d to move.'

    def _on_state(self, msg):
        # Adopt the node's angle only at start-up and after open / close.
        # During nudging the report lags behind (2 Hz), and taking it then
        # made the next nudge start from a stale angle.
        if not self._synced or time.time() - self._last_key_time > 1.5:
            self.angle = float(msg.data)
            self._synced = True

    def send(self, angle):
        self.angle = angle
        self.pub.publish(Float32(data=float(angle)))

    def call(self, client, label):
        # Ask even if discovery hasn't finished yet: the request is delivered
        # as soon as the node is found.
        client.call_async(Trigger.Request())
        self.status = label

    def on_key(self, key):
        angle, self.step, action = apply_key(key, self.angle, self.step)
        if action == 'move':
            self._last_key_time = time.time()
            self.send(angle)  # always send; discovery can lag a second or two
            self.status = 'Moving.'
        elif action == 'step':
            self.status = f'Step size {self.step:.0f} deg.'
        elif action == 'open':
            self.call(self.open_client, 'Opening.')
            self._last_key_time = 0.0  # take the node's angle once it's there
        elif action == 'close':
            self.call(self.close_client, 'Closing.')
            self._last_key_time = 0.0
        elif action == 'mark_open':
            self.open_mark = self.angle
            self.status = f'Open remembered at {self.angle:.0f} deg.'
        elif action == 'mark_closed':
            self.closed_mark = self.angle
            self.status = f'Closed remembered at {self.angle:.0f} deg.'
        return action != 'quit'

    def node_seen(self):
        # Same graph lookup `ros2 topic info` uses; the publisher's matched
        # count lagged behind and kept reporting 0 on the robot.
        return self.count_subscribers('/grabber/angle') > 0

    def line(self):
        link = '' if self.node_seen() else ' [node not seen]'
        marks = (f'open {self.open_mark:.0f}' if self.open_mark is not None else 'open -') + \
                (f', closed {self.closed_mark:.0f}' if self.closed_mark is not None else ', closed -')
        text = (f' angle {self.angle:5.1f} | step {self.step:2.0f} | '
                f'{marks} | {self.status}{link}')
        # Never wider than the terminal: a wrapped line can't be overwritten
        # in place, and every redraw then prints a new copy.
        width = shutil.get_terminal_size((80, 24)).columns - 1
        return '\r\x1b[K' + text[:max(10, width)]


def read_key(timeout):
    """One key press (arrow keys arrive as 3-character sequences), or ''."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ''
    key = sys.stdin.read(1)
    if key == '\x1b':
        more, _, _ = select.select([sys.stdin], [], [], 0.01)
        if more:
            key += sys.stdin.read(2)
    return key


def main():
    if not sys.stdin.isatty():
        sys.exit('Run this in an interactive terminal (an ssh session).')
    rclpy.init()
    node = GrabberTeleop()
    print('Grabber teleop:  a/d or arrows move | 1/2/3 step 1/5/15 | o open | c close | '
          'h centre | m mark open | n mark closed | q quit')
    saved = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = read_key(0.05)
            if key:
                running = node.on_key(key)
            sys.stdout.write(node.line())
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        print()
        open_deg, closed_deg = suggested_limits(node.open_mark, node.closed_mark)
        if open_deg is not None and closed_deg is not None:
            print(f'Marked: open {node.open_mark:.0f}, closed {node.closed_mark:.0f}. '
                  f'Backed off {BACK_OFF_DEG:.0f} deg, use:')
            print(f'  --ros-args -p open_deg:={open_deg:.1f} -p closed_deg:={closed_deg:.1f}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()