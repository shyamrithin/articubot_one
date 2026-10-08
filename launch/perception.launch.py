#!/usr/bin/env python3
# =============================================================================
# File:        launch/perception.launch.py
# Package:     articubot_one
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Starts the search-and-rescue perception nodes, each switchable:
#
#   thermal:=true    thermal_node.py    MLX90640 -> /thermal/hotspots, images
#   victims:=true    victim_mapper.py   hot spots + /scan + TF -> victims on map
#   vision:=true     vision_node.py     C270 webcam + thermal overlay
#
# Settings come from config/sar_params.yaml (sar_params_file:= to use another).
#
# thermal_node restarts itself if it dies (an I2C glitch can take the sensor
# process down); the others stay down so a real bug is not hidden.
#
# victim_mapper needs a 'map' frame, so run this alongside a mode that has
# SLAM or AMCL (Map, Explore, Navigate, Continue map). Without one it runs
# but places nothing.
#
# USAGE
# -----
#   ros2 launch articubot_one perception.launch.py
#   ros2 launch articubot_one perception.launch.py vision:=false
# Usually started for you by sar_bringup.launch.py or the touchscreen panel.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = 'articubot_one'


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory(PACKAGE),
                                  'config', 'sar_params.yaml')
    sar_params = LaunchConfiguration('sar_params_file')

    args = [
        DeclareLaunchArgument('sar_params_file', default_value=default_params,
                              description='Parameter file for the SAR nodes'),
        DeclareLaunchArgument('thermal', default_value='true',
                              description='Start the thermal camera node'),
        DeclareLaunchArgument('victims', default_value='true',
                              description='Start the victim mapper'),
        DeclareLaunchArgument('vision', default_value='true',
                              description='Start the webcam / fusion node'),
    ]

    thermal = Node(
        package=PACKAGE, executable='thermal_node.py', name='thermal',
        output='screen', parameters=[sar_params],
        respawn=True, respawn_delay=3.0,
        condition=IfCondition(LaunchConfiguration('thermal')))

    victims = Node(
        package=PACKAGE, executable='victim_mapper.py', name='victim_mapper',
        output='screen', parameters=[sar_params],
        condition=IfCondition(LaunchConfiguration('victims')))

    vision = Node(
        package=PACKAGE, executable='vision_node.py', name='vision',
        output='screen', parameters=[sar_params],
        condition=IfCondition(LaunchConfiguration('vision')))

    return LaunchDescription(args + [thermal, victims, vision])
