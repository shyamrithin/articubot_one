#!/usr/bin/env bash
# =============================================================================
# File:        ui/start_ui.sh
# Package:     articubot_one
# =============================================================================
# DESCRIPTION
# -----------
# Entry point for the robot UI. Sources ROS and the workspace, sets the same
# network variables as ~/.bashrc (systemd does not read .bashrc), then hands
# over to robot_supervisor.py with exec so systemd's stop signal reaches it
# directly and it can shut every launched layer down cleanly.
#
# Run by hand for testing:   ~/robot_ws/src/articubot_one/ui/start_ui.sh
# Run at boot:               see ui/robot-ui.service
# =============================================================================

source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$UI_DIR/robot_supervisor.py"
