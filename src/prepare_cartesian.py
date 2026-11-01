"""既存の動作計画サンプルに、置き先指定と記録用の区切りを追加する。"""

from pathlib import Path
import shutil

# 講義用の元ソースは変更せず、作業用コピーに置き先指定を追加する。
src = Path("/home/ubuntu/airobot_ws/src/crane_plus/crane_plus_examples")
out = Path("/tmp/pai-xy")
out.mkdir(exist_ok=True)
shutil.copytree(src, out / "teacher_src", dirs_exist_ok=True)
p = out / "teacher_src/src/pick_and_place.cpp"
s = p.read_text()
s = s.replace(
    "#include <cmath>",
    "#include <cmath>\n#include <iostream>\n#include <iomanip>\n#include <chrono>\n#include <thread>",
)
s = s.replace(
    "  // For current state monitor",
    """  const double gx = move_group_arm_node->get_parameter("goal_x").as_double();
  const double gy = move_group_arm_node->get_parameter("goal_y").as_double();
  const double yaw = std::atan2(gy,gx);
  const double radius = std::hypot(gx,gy);
  auto mark = [&](int stage) {
    std::cout << "XY_STAGE " << stage << " " << std::setprecision(15)
              << move_group_arm_node->now().seconds() << std::endl;
  };
  // For current state monitor""",
)
s = s.replace("setMaxVelocityScalingFactor(1.0)", "setMaxVelocityScalingFactor(0.5)")
s = s.replace(
    "setMaxAccelerationScalingFactor(1.0)", "setMaxAccelerationScalingFactor(0.5)"
)
s = s.replace(
    "target_pose.position.x = 0.15;",
    "target_pose.position.x = (radius - 0.10) * std::cos(yaw);",
)
s = s.replace(
    "target_pose.position.y = 0.0;",
    "target_pose.position.y = (radius - 0.10) * std::sin(yaw);",
)
s = s.replace("to_radians(90), to_radians(0)", "to_radians(90), yaw")
s = s.replace(
    "  // Release",
    "  std::this_thread::sleep_for(std::chrono::milliseconds(1000));\n\n  // Release",
)
lines = []
stage = 0
for line in s.splitlines():
    lines.append(line)
    if line.strip() in ("move_group_arm.move();", "move_group_gripper.move();"):
        stage += 1
        lines.append(f"  mark({stage});")
p.write_text("\n".join(lines) + "\n")
launch = (src / "launch/example.launch.py").read_text()
launch = launch.replace(
    "import yaml",
    "import yaml\nfrom launch_ros.parameter_descriptions import ParameterValue",
)
launch = launch.replace("package='crane_plus_examples',", "package=None,")
launch = launch.replace(
    "executable=LaunchConfiguration('example'),",
    "executable='/tmp/pai-xy/build/pick_and_place',",
)
launch = launch.replace(
    "                                    kinematics_yaml])",
    "                                    kinematics_yaml, {'goal_x': ParameterValue(LaunchConfiguration('goal_x'), value_type=float), 'goal_y': ParameterValue(LaunchConfiguration('goal_y'), value_type=float)}])",
)
launch = launch.replace(
    "        declare_use_sim_time,",
    "        DeclareLaunchArgument('goal_x', default_value='0.23'),\n        DeclareLaunchArgument('goal_y', default_value='0.0'),\n        declare_use_sim_time,",
)
(out / "teacher.launch.py").write_text(launch)
print("Isolated Cartesian-goal teacher prepared.")

# Gazeboは物理計算だけを起動し、表示はRVizで行う。
source_root = Path("/home/ubuntu/airobot_ws/src/crane_plus")
launch = (
    source_root / "crane_plus_gazebo/launch/crane_plus_with_table.launch.py"
).read_text()
launch = launch.replace(
    "['ign gazebo -r', world_file, '--gui-config', gui_config]",
    "['ign gazebo -r -s', world_file]",
)
launch = launch.replace(
    "get_package_share_directory('crane_plus_moveit_config'),\n                '/launch/run_move_group.launch.py'",
    "'/tmp/pai-xy/movegroup.launch.py'",
)
(out / "sim.launch.py").write_text(launch)
movegroup = (
    source_root / "crane_plus_moveit_config/launch/run_move_group.launch.py"
).read_text()
movegroup = movegroup.replace("                              rviz_node,\n", "")
(out / "movegroup.launch.py").write_text(movegroup)
