#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"; shift || true
case "$mode" in
  prepare|launch)
    docker exec -u ubuntu "${PAI_CONTAINER_NAME:-pai_ros2}" bash -lc '
      source /opt/ros/humble/setup.bash
      source /home/ubuntu/airobot_ws/install/setup.bash
      python3 /workspace/src/prepare_cartesian.py
      cmake -S /tmp/pai-xy/teacher_src -B /tmp/pai-xy/build -DBUILD_TESTING=OFF > /tmp/pai-xy/build.log 2>&1
      cmake --build /tmp/pai-xy/build --target pick_and_place -j2 >> /tmp/pai-xy/build.log 2>&1'
    if [[ "$mode" == launch ]]; then
      docker exec -it -u ubuntu "${PAI_CONTAINER_NAME:-pai_ros2}" bash -lc '
        source /opt/ros/humble/setup.bash
        source /home/ubuntu/airobot_ws/install/setup.bash
        exec ros2 launch /tmp/pai-xy/sim.launch.py'
    fi
    ;;
  collect)
    docker exec -u ubuntu -e OPENBLAS_NUM_THREADS=1 "${PAI_CONTAINER_NAME:-pai_ros2}" bash -lc '
      source /opt/ros/humble/setup.bash
      source /home/ubuntu/airobot_ws/install/setup.bash
      exec python3 /workspace/src/cartesian_transport.py "$@"' bash "$mode" "$@"
    ;;
  *) echo 'Usage: bash environment/cartesian.sh {prepare|launch|collect} [arguments]' >&2; exit 2;;
esac
