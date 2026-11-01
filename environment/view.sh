#!/usr/bin/env bash
set -euo pipefail
PAI_NAME="${PAI_CONTAINER_NAME:-pai_ros2}"
docker exec -d -u ubuntu -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 "$PAI_NAME" bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/ubuntu/airobot_ws/install/setup.bash
  exec rviz2 -d /workspace/environment/pick_place.rviz'
docker exec -u ubuntu -e OPENBLAS_NUM_THREADS=1 "$PAI_NAME" bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/ubuntu/airobot_ws/install/setup.bash
  exec python3 /workspace/tools/view_reference.py --goal "$1"' bash "${1:-E6}"
