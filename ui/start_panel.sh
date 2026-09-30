#!/usr/bin/env bash
# =============================================================================
# File:        ui/start_panel.sh
# Package:     articubot_one
# =============================================================================
# DESCRIPTION
# -----------
# Starts the touchscreen panel (robot_panel.py) on the robot's own display.
# Works both from an ssh session and from the desktop autostart.
#
#   - Sources ROS 2 Humble and the robot workspace, and sets the same network
#     variables as ~/.bashrc.
#   - From ssh there is no display set, so this points at the robot's screen
#     (:0) and finds the X authority file: GNOME on Wayland keeps it under
#     /run/user/<uid>/.mutter-Xwaylandauth.*, an Xorg session in ~/.Xauthority.
#   - exec hands over to Python, so Ctrl-C or SIGTERM reaches the panel and it
#     stops every layer it started before exiting.
#
# Usage:        ~/robot_ws/src/articubot_one/ui/start_panel.sh
# Windowed:     PANEL_WINDOWED=1 PANEL_SHOW_CURSOR=1 ./start_panel.sh
# In background from ssh (survives logout):
#               nohup ./start_panel.sh > ~/.ros/robot_panel.out 2>&1 &
# =============================================================================

source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

export DISPLAY="${DISPLAY:-:0}"
if [ -z "$XAUTHORITY" ]; then
    for candidate in /run/user/"$(id -u)"/.mutter-Xwaylandauth.* "$HOME/.Xauthority"; do
        if [ -f "$candidate" ]; then
            export XAUTHORITY="$candidate"
            break
        fi
    done
fi

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$UI_DIR/robot_panel.py"
