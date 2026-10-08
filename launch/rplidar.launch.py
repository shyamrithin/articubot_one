#!/usr/bin/env python3
# =============================================================================
# File:        launch/rplidar.launch.py
# Package:     articubot_one
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Starts the RPLIDAR A2M8 driver on its CP2102 USB adapter (found by its
# stable /dev/serial/by-id name, so USB enumeration order doesn't matter).
#
#   frame_id          laser_frame (matches description/lidar.xacro)
#   angle_compensate  True: a fixed number of points per scan. With it off the
#                     point count varies and slam_toolbox's map freezes.
#
# The A2M8 sometimes times out on its first connection while its motor spins
# up (SL_RESULT_OPERATION_TIMEOUT). The node then exits, so it is restarted
# automatically after 3 s instead of leaving the robot without a lidar.
#
# USAGE
# -----
#   ros2 launch articubot_one rplidar.launch.py
# Also started by the touchscreen panel and sar_bringup.launch.py.
# =============================================================================

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            respawn=True,
            respawn_delay=3.0,
            parameters=[{
                'serial_port': '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0',
                'serial_baudrate': 115200,
                'frame_id': 'laser_frame',
                'angle_compensate': True,
            }],
        ),
    ])
