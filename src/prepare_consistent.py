"""手本を作る際の、最初の接近目標を共通の関節角に変更する。"""

import hashlib, json, shutil
from pathlib import Path


def prepare(baseline_dir, output_dir):
    """最初の接近目標を共通関節角に変更した手本ソースを作る。

    Args:
        baseline_dir: 変更前の手本JSONを保存したフォルダ。
        output_dir: 変更後のC++ソースとlaunchファイルを保存するフォルダ。

    Returns:
        共通の目標角、参照した手本、変更後ソースのハッシュなどの辞書。"""
    src = Path("/tmp/pai-xy")
    out = Path("/tmp/pai-consistent")
    out.mkdir(exist_ok=True)
    teacher = json.loads((baseline_dir / "teacher_corner1_attempt1.json").read_text())
    # 最初の手本で接近を終えた時点の関節指令を、共通の目標にする。
    stamp = dict(teacher["stages"])[5]
    row = min(teacher["samples"], key=lambda r: abs(r["t"] - stamp))
    target = row["action"][:4]
    shutil.copytree(src / "teacher_src", out / "teacher_src", dirs_exist_ok=True)
    path = out / "teacher_src/src/pick_and_place.cpp"
    text = path.read_text()
    old = "  move_group_arm.setPoseTarget(target_pose);"
    assert text.count(old) == 5, "Unexpected stock teacher shape"
    assert "std::vector<double>" not in text
    text = text.replace("#include <cmath>", "#include <cmath>\n#include <vector>", 1)
    new = (
        "  move_group_arm.setJointValueTarget(std::vector<double>{"
        + ", ".join(format(x, ".17g") for x in target)
        + "});"
    )
    # 最初の接近だけを変更し、その後の把持・搬送の計画は残す。
    text = text.replace(old, new, 1)
    path.write_text(text)
    launch = (
        (src / "teacher.launch.py")
        .read_text()
        .replace(
            "/tmp/pai-xy/build/pick_and_place",
            "/tmp/pai-consistent/build/pick_and_place",
        )
    )
    (out / "teacher.launch.py").write_text(launch)
    shutil.copy2(path, output_dir / "teacher_pick_and_place.cpp")
    shutil.copy2(out / "teacher.launch.py", output_dir / "teacher.launch.py")
    return dict(
        source_teacher="teacher_corner1_attempt1",
        source_sha256=hashlib.sha256(
            (baseline_dir / "teacher_corner1_attempt1.json").read_bytes()
        ).hexdigest(),
        source_stage=5,
        approach_joint_target_rad=target,
        teacher_cpp_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        change="Only the first picking approach uses the observed fixed joint target; remaining planning stages unchanged",
    )
