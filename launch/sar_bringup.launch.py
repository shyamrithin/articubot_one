#!/usr/bin/env python3
# =============================================================================
# File:        launch/sar_bringup.launch.py
# Package:     articubot_one
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# The whole search-and-rescue robot from one command, for ssh sessions and
# demos. Pick a mode and it starts the same layers the touchscreen panel
# does, plus perception and the grabber:
#
#   mode:=map        drivetrain + lidar + SLAM (new map)
#   mode:=explore    drivetrain + lidar + SLAM + Nav2 (goals from RViz)
#   mode:=navigate   drivetrain + lidar + AMCL on map:=<name> + Nav2
#   mode:=resume     drivetrain + lidar + SLAM continuing map:=<name>
#                    (needs <name>.posegraph, i.e. a map saved by the panel;
#                    put the robot back where that map was first started)
#
#   perception:=true   thermal + victim mapper + vision (perception.launch.py)
#     thermal:= victims:= vision:=   switch single perception nodes off
#   grabber:=true      grabber servo (manipulation.launch.py)
#   at_map_start:=true navigate only: tell AMCL the robot is at the map's
#                      start point (same as the panel's "Robot is at map
#                      start"); set false and use RViz's 2D Pose Estimate
#                      when it is somewhere else
#   map_dir:=~/maps    where saved maps live (same folder as the panel)
#
# START ORDER (seconds after launch)
#   0   drivetrain, lidar, grabber
#   6   SLAM or AMCL   (the controller manager itself waits 3 s)
#   8   perception
#   10  Nav2 (explore)        12  initial pose, 14  Nav2 (navigate)
# The panel checks each layer is ready before the next; this file uses fixed
# delays instead, so if something is slow, watch the log and relaunch.
# The lidar restarts itself if its first connection times out.
#
# Each included file runs in its own launch scope, so arguments one file
# declares (e.g. a params_file default) cannot leak into the next.
#
# Do not run this while the panel has a mode running: both would start the
# same nodes. Use one or the other.
#
# USAGE
# -----
#   ros2 launch articubot_one sar_bringup.launch.py mode:=map
#   ros2 launch articubot_one sar_bringup.launch.py mode:=explore vision:=false
#   ros2 launch articubot_one sar_bringup.launch.py mode:=navigate map:=room_v2
#   ros2 launch articubot_one sar_bringup.launch.py mode:=resume map:=room_v2
# RViz runs on the laptop, not here.
# =============================================================================

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

PACKAGE = 'articubot_one'
MODES = ('map', 'explore', 'navigate', 'resume')
SLAM_PARAMS_FILE = 'mapper_params_async.yaml'
RUN_DIR = os.path.expanduser('~/.ros/sar_bringup')

T_LOCALISE = 6.0
T_PERCEPTION = 8.0
T_NAV_EXPLORE = 10.0
T_INITIAL_POSE = 12.0
T_NAV_NAVIGATE = 14.0


def _share():
    return get_package_share_directory(PACKAGE)


def _include(file_name, **launch_args):
    """Include a launch file inside its own scope.

    Without the GroupAction, an argument one included file declares (for
    example a 'params_file' default) leaks into every file included after it,
    and Nav2 then starts with the wrong parameter file.
    """
    source = PythonLaunchDescriptionSource(os.path.join(_share(), 'launch', file_name))
    include = IncludeLaunchDescription(source, launch_arguments=launch_args.items())
    return GroupAction(actions=[include], scoped=True)


def _later(seconds, action):
    return TimerAction(period=seconds, actions=[action])


def _is_true(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _resume_params(map_dir, map_name):
    """Copy the SLAM params with map_file_name pointing at a saved pose graph.

    Same rewrite the touchscreen panel does for Continue map.
    """
    with open(os.path.join(_share(), 'config', SLAM_PARAMS_FILE)) as handle:
        params = yaml.safe_load(handle)
    p = params['slam_toolbox']['ros__parameters']
    p['mode'] = 'mapping'
    p['map_file_name'] = os.path.join(map_dir, map_name)
    p['map_start_at_dock'] = True
    p.pop('map_start_pose', None)
    os.makedirs(RUN_DIR, exist_ok=True)
    path = os.path.join(RUN_DIR, f'slam_resume_{map_name}.yaml')
    with open(path, 'w') as handle:
        yaml.safe_dump(params, handle, default_flow_style=False)
    return path


def _initial_pose_at_origin():
    """Publish 'robot is at the map origin' to AMCL a few times."""
    cov = [0.0] * 36
    cov[0], cov[7], cov[35] = 0.0625, 0.0625, 0.0685  # 25 cm, 25 cm, 15 degrees
    cov_text = ', '.join(f'{c:.4f}' for c in cov)
    msg = ('{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, '
           'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [' + cov_text + ']}}')
    return ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--times', '3', '--rate', '0.5', '/initialpose',
             'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
        output='log')


def _plan(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    mode = get('mode').strip().lower()
    map_name = get('map').strip()
    map_dir = os.path.expanduser(get('map_dir'))

    if mode not in MODES:
        raise RuntimeError(f'mode:={mode!r} is not one of {", ".join(MODES)}.')
    if mode in ('navigate', 'resume') and not map_name:
        raise RuntimeError(f'mode:={mode} needs map:=<name> (a map saved in {map_dir}).')

    actions = [
        LogInfo(msg=f'SAR bringup: mode {mode}' + (f', map {map_name}' if map_name else '')),
        _include('launch_robot.launch.py'),
        _include('rplidar.launch.py'),
    ]

    if _is_true(get('grabber')):
        actions.append(_include('manipulation.launch.py'))

    if mode in ('map', 'explore'):
        actions.append(_later(T_LOCALISE, _include('online_async_launch.py',
                                                   use_sim_time='false')))
    elif mode == 'resume':
        graph = os.path.join(map_dir, map_name + '.posegraph')
        if not os.path.isfile(graph):
            raise RuntimeError(f'{graph} not found. Continue map needs a map saved '
                               'by the panel (it writes the .posegraph).')
        actions.append(_later(T_LOCALISE, _include(
            'online_async_launch.py', use_sim_time='false',
            slam_params_file=_resume_params(map_dir, map_name))))
    else:  # navigate
        map_yaml = os.path.join(map_dir, map_name + '.yaml')
        if not os.path.isfile(map_yaml):
            raise RuntimeError(f'{map_yaml} not found.')
        actions.append(_later(T_LOCALISE, _include('localization_launch.py',
                                                   use_sim_time='false', map=map_yaml)))
        if _is_true(get('at_map_start')):
            actions.append(_later(T_INITIAL_POSE, _initial_pose_at_origin()))

    if mode in ('explore', 'navigate'):
        nav_time = T_NAV_EXPLORE if mode == 'explore' else T_NAV_NAVIGATE
        actions.append(_later(nav_time, _include('navigation_launch.py',
                                                 use_sim_time='false',
                                                 map_subscribe_transient_local='true')))

    if _is_true(get('perception')):
        actions.append(_later(T_PERCEPTION, _include(
            'perception.launch.py',
            thermal=get('thermal'), victims=get('victims'), vision=get('vision'))))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='explore',
                              description='map | explore | navigate | resume'),
        DeclareLaunchArgument('map', default_value='',
                              description='Saved map name (navigate, resume)'),
        DeclareLaunchArgument('map_dir', default_value='~/maps',
                              description='Folder with saved maps'),
        DeclareLaunchArgument('at_map_start', default_value='true',
                              description='navigate: robot starts at the map origin'),
        DeclareLaunchArgument('perception', default_value='true',
                              description='Start thermal, victim mapper and vision'),
        DeclareLaunchArgument('thermal', default_value='true'),
        DeclareLaunchArgument('victims', default_value='true'),
        DeclareLaunchArgument('vision', default_value='true'),
        DeclareLaunchArgument('grabber', default_value='true',
                              description='Start the grabber servo node'),
        OpaqueFunction(function=_plan),
    ])
