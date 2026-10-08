#!/usr/bin/env python3
# =============================================================================
# File:        launch/manipulation.launch.py
# Package:     articubot_one
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Starts the front grabber: grabber_node.py driving the MG90S servo from the
# Pi's hardware PWM on GPIO 18. Settings (open_deg, closed_deg, speed, idle
# release) come from the 'grabber' section of config/sar_params.yaml.
#
# Needs the one-time PWM setup described in grabber/grabber_node.py
# (dtoverlay=pwm in config.txt and the udev rule). Without it the node runs
# in simulated mode and says so.
#
# USAGE
# -----
#   ros2 launch articubot_one manipulation.launch.py
#   ros2 service call /grabber/open  std_srvs/srv/Trigger
#   ros2 service call /grabber/close std_srvs/srv/Trigger
# Keyboard control stays a separate, interactive program:
#   ros2 run articubot_one grabber_teleop.py
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = 'articubot_one'


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory(PACKAGE),
                                  'config', 'sar_params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('sar_params_file', default_value=default_params,
                              description='Parameter file for the SAR nodes'),
        Node(package=PACKAGE, executable='grabber_node.py', name='grabber',
             output='screen', parameters=[LaunchConfiguration('sar_params_file')]),
    ])
