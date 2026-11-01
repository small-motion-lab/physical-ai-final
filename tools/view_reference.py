"""指定した置き先と、動作中のアーム・ブロックをRVizに表示する。"""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import cartesian_transport as c


def marker(i, kind, xyz, scale, rgb):
    """RVizに表示する物体や枠のMarkerメッセージを作る。

    Args:
        i: 同じ名前空間内で重複しないマーカー番号。
        kind: Marker.CUBEなどの形状種別。
        xyz: base_linkを基準とする表示位置。単位はm。
        scale: マーカーの幅・高さなどを指定する3要素。
        rgb: 0〜1で表す赤・緑・青の色成分。

    Returns:
        位置・大きさ・色を設定したMarkerメッセージ。"""
    m = Marker()
    m.header.frame_id = "base_link"
    m.ns = "transport"
    m.id = i
    m.type = kind
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in xyz]
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = 1.0
    return m


def main():
    """CLIの--goalで指定した目標と実測状態を表示し、Ctrl+Cで終了する。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", choices=[f"E{i}" for i in range(1, 7)], default="E6")
    a = p.parse_args()
    goal = c.TEST[int(a.goal[1:]) - 1]
    rclpy.init()
    arm = c.b.Arm()
    pub = arm.create_publisher(MarkerArray, "/pick_place/scene", 10)
    try:
        arm.ready()
        last = -1
        print("Live measured view ready: " + a.goal, flush=True)
        while rclpy.ok():
            arm.spin()
            if arm.sim - last < 0.1:
                continue
            # ブロックはGazeboの実測位置、ロボットはROSの関節状態で表示する。
            last = arm.sim
            xyz = arm.cube.copy()
            xyz[2] -= 1.015
            cube = marker(0, Marker.CUBE, xyz, [0.05] * 3, [1.0, 0.62, 0.12])
            (
                cube.pose.orientation.x,
                cube.pose.orientation.y,
                cube.pose.orientation.z,
                cube.pose.orientation.w,
            ) = map(float, arm.cube_quat)
            table = marker(
                1,
                Marker.CUBE,
                [0.1, 0.0, -0.012],
                [0.65, 0.55, 0.02],
                [0.31, 0.38, 0.44],
            )
            dest = marker(
                2,
                Marker.LINE_STRIP,
                [*goal, 0.002],
                [0.0025, 0.0, 0.0],
                [0.2, 1.0, 0.5],
            )
            dest.points = [
                Point(x=x, y=y, z=0.0)
                for x, y in [
                    (-0.035, -0.035),
                    (0.035, -0.035),
                    (0.035, 0.035),
                    (-0.035, 0.035),
                    (-0.035, -0.035),
                ]
            ]
            start = marker(
                3,
                Marker.LINE_STRIP,
                [0.0, -0.1, 0.002],
                [0.0015, 0.0, 0.0],
                [0.5, 0.7, 1.0],
            )
            start.points = dest.points
            training = marker(
                4,
                Marker.LINE_STRIP,
                [0.0, 0.0, 0.002],
                [0.001, 0.0, 0.0],
                [0.6, 0.65, 0.7],
            )
            training.points = [
                Point(x=x, y=y, z=0.0)
                for x, y in [
                    (0.20, -0.06),
                    (0.26, -0.06),
                    (0.26, 0.06),
                    (0.20, 0.06),
                    (0.20, -0.06),
                ]
            ]
            pub.publish(MarkerArray(markers=[cube, table, dest, start, training]))
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
