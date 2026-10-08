# =============================================================================
# File:        CMakeLists.txt
# Package:     articubot_one
# Platform:    Raspberry Pi 4 (shyam-rpi4) / Ubuntu 22.04 / ROS 2 Humble
# =============================================================================
# DESCRIPTION
# -----------
# Build and install rules for the robot package. Nothing is compiled; this
# only installs files into install/articubot_one so `ros2 launch` and
# `ros2 run` can find them.
#
#   share/articubot_one/   config, description, launch, worlds, maps
#   lib/articubot_one/     the Python nodes, installed as executables so a
#                          launch file can start them with Node(...):
#                            thermal_node.py   victim_mapper.py
#                            vision_node.py    grabber_node.py
#                            grabber_teleop.py
#
# The nodes keep their .py names, so these both work after a build:
#   ros2 run articubot_one thermal_node.py
#   python3 ~/robot_ws/src/articubot_one/thermal/thermal_node.py
#
# With `colcon build --symlink-install` the installed nodes are links to the
# source files, so the source files must be executable (chmod +x, see README).
# =============================================================================

cmake_minimum_required(VERSION 3.5)
project(articubot_one)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY config description launch worlds maps
  DESTINATION share/${PROJECT_NAME}
)

# Python nodes, installed as executables for launch files and `ros2 run`.
install(
  PROGRAMS
    thermal/thermal_node.py
    thermal/victim_mapper.py
    vision/vision_node.py
    grabber/grabber_node.py
    grabber/grabber_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
