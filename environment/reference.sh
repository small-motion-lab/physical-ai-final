#!/usr/bin/env bash
set -euo pipefail
docker exec -u ubuntu -e OPENBLAS_NUM_THREADS=1 "${PAI_CONTAINER_NAME:-pai_ros2}" bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/ubuntu/airobot_ws/install/setup.bash
  exec python3 /workspace/tools/reference.py "$@"' bash "$@"
