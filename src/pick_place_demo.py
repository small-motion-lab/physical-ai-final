#!/usr/bin/env python3
"""ROS 2との接続、関節指令の送信、初期化と搬送成功の判定。"""
import json
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState

JOINTS = [f"crane_plus_joint{i}" for i in range(1, 5)] + ["crane_plus_joint_hand"]
START = np.array([0.0, -0.1, 1.04])
GOAL = np.array([0.25, 0.0, 1.04])


def duration(seconds):
    """秒数secondsを、ROSのDurationメッセージへ変換して返す。"""
    ns = round(seconds * 1e9)
    return Duration(sec=ns // 10**9, nanosec=ns % 10**9)


def features(phase, count=161):
    """動作進行度を、正規化したガウス時間基底へ変換する。

    Args:
        phase: 0〜1の進行度を並べた配列。
        count: 基底の数。既定値は161。

    Returns:
        各行の和が1になる、時刻数×countの基底行列。"""
    p = np.asarray(phase).reshape(-1, 1)
    centers = np.linspace(0, 1, count)
    basis = np.exp(-0.5 * ((p - centers) / (1.2 / (count - 1))) ** 2)
    return basis / basis.sum(axis=1, keepdims=True)


class Arm(Node):
    """関節制御器・シミュレーション時刻・物体位置を扱うROS 2ノード。

    引数はない。生成前にrclpy.init()を呼ぶ。
    qは実測関節角、actionは制御器の指令角で、単位はrad。
    simはシミュレーション時刻（秒）、cubeは物体中心の世界座標（m）。
    使い終わったらclose()で購読用プロセスとノードを終了する。"""

    def __init__(self):
        """ROSの送受信とGazeboの物体位置の読み取りを開始する。"""
        super().__init__("learned_pick_place")
        self.q = self.sim = self.cube = None
        self.cube_quat = [0.0, 0.0, 0.0, 1.0]
        self.action = np.full(5, np.nan)
        self.action_wall = np.zeros(2)
        self.joint_wall = self.pose_wall = 0.0
        self.arm_pub = self.create_publisher(
            JointTrajectory, "/crane_plus_arm_controller/joint_trajectory", 10
        )
        self.hand_pub = self.create_publisher(
            JointTrajectory, "/crane_plus_gripper_controller/joint_trajectory", 10
        )
        self.create_subscription(
            JointState, "/joint_states", self.joints, qos_profile_sensor_data
        )
        self.create_subscription(Clock, "/clock", self.clock, qos_profile_sensor_data)
        for suffix in ["state", "controller_state"]:
            self.create_subscription(
                JointTrajectoryControllerState,
                "/crane_plus_arm_controller/" + suffix,
                lambda m: self.desired(m, slice(0, 4)),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointTrajectoryControllerState,
                "/crane_plus_gripper_controller/" + suffix,
                lambda m: self.desired(m, slice(4, 5)),
                qos_profile_sensor_data,
            )
        # 物体位置はGazeboの世界座標から取得する。TFの相対座標とは分ける。
        self.pose_reader = subprocess.Popen(
            [
                "ign",
                "topic",
                "-e",
                "-t",
                "/world/default/dynamic_pose/info",
                "--json-output",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self.read_poses, daemon=True).start()

    def joints(self, msg):
        """JointStateメッセージmsgから、対象5関節の実測角度を更新する。"""
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS])
            self.joint_wall = time.monotonic()

    def clock(self, msg):
        """Clockメッセージmsgから、シミュレーション時刻を秒で更新する。"""
        self.sim = msg.clock.sec + msg.clock.nanosec * 1e-9

    def desired(self, msg, columns):
        """制御器のメッセージから指令角を取り出す。

        Args:
            msg: JointTrajectoryControllerStateメッセージ。
            columns: action内の書き込み範囲。アームは0〜3、グリッパは4。

        Returns:
            なし。actionと指令を受信した時刻を更新する。"""
        positions = msg.reference.positions or msg.desired.positions
        if positions:
            self.action[columns] = positions
            self.action_wall[0 if columns.start == 0 else 1] = time.monotonic()

    def read_poses(self):
        """Gazeboの位置情報を別スレッドで読み、物体中心と姿勢を更新する。"""
        for line in self.pose_reader.stdout:
            try:
                message = json.loads(line)
                for pose in message.get("pose", []):
                    if pose.get("name") != "wood_cube_5cm":
                        continue
                    p, q = pose.get("position", {}), pose.get("orientation", {})
                    x, y, z, w = [q.get(k, 0.0) for k in ("x", "y", "z", "w")]
                    # モデル原点から25mm上にある、ブロック中心の位置へ換算する。
                    offset = 0.025 * np.array(
                        [
                            2 * (x * z + w * y),
                            2 * (y * z - w * x),
                            1 - 2 * (x * x + y * y),
                        ]
                    )
                    self.cube = (
                        np.array([p.get(k, 0.0) for k in ("x", "y", "z")]) + offset
                    )
                    self.cube_quat = [x, y, z, w]
                    self.pose_wall = time.monotonic()
            except (ValueError, KeyError):
                continue

    def spin(self):
        """ROSの受信コールバックを1回処理する。戻り値はない。"""
        rclpy.spin_once(self, timeout_sec=0.02)

    def ready(self):
        """制御器・関節状態・時刻・物体位置がそろうまで、最大60秒待つ。"""
        wall = time.monotonic()
        while time.monotonic() - wall < 60:
            self.spin()
            if (
                self.q is not None
                and self.sim is not None
                and self.cube is not None
                and np.isfinite(self.action).all()
                and self.arm_pub.get_subscription_count()
                and self.hand_pub.get_subscription_count()
            ):
                return
        raise RuntimeError(
            "Missing joint state, simulation clock, cube pose, or controllers"
        )

    def command(self, times, values):
        # アーム4関節とグリッパの指令を、それぞれの制御器へ送る。
        """アームとグリッパへ、時刻付きの関節指令を送る。

        Args:
            times: 送信時点から各目標までの所要時間。単位は秒。
            values: timesと同じ行数の、5関節の指令角度。単位はrad。

        Returns:
            なし。アーム4関節とグリッパへ別々に送信する。"""
        for pub, names, columns in [
            (self.arm_pub, JOINTS[:4], slice(0, 4)),
            (self.hand_pub, JOINTS[4:], slice(4, 5)),
        ]:
            msg = JointTrajectory(joint_names=names)
            msg.points = [
                JointTrajectoryPoint(
                    positions=[float(v) for v in q[columns]],
                    time_from_start=duration(t),
                )
                for t, q in zip(times, values)
            ]
            pub.publish(msg)

    def wait(self, seconds):
        """secondsで指定したシミュレーション時間だけ、受信処理を続けて待つ。"""
        start, wall = self.sim, time.monotonic()
        while self.sim - start < seconds:
            self.spin()
            if time.monotonic() - wall > max(60, seconds * 15):
                raise RuntimeError("Simulator stalled")

    def reset(self):
        # 試行の開始前だけ、アームとブロックを初期状態へ戻す。
        """試行前にアームとブロックを初期位置へ戻し、誤差が許容内か確認する。"""
        self.command([3.0], [np.zeros(5)])
        self.wait(4.0)
        request = 'name: "wood_cube_5cm", position: {x: 0, y: -0.1, z: 1.015}, orientation: {w: 1}'
        result = subprocess.run(
            [
                "ign",
                "service",
                "-s",
                "/world/default/set_pose",
                "--reqtype",
                "ignition.msgs.Pose",
                "--reptype",
                "ignition.msgs.Boolean",
                "--timeout",
                "5000",
                "--req",
                request,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode or "true" not in result.stdout:
            raise RuntimeError("Episode reset failed: " + result.stdout + result.stderr)
        self.wait(1.0)
        if np.linalg.norm(self.cube - START) > 0.005 or np.max(np.abs(self.q)) > 0.02:
            raise RuntimeError(f"Bad initial state: cube={self.cube}, joints={self.q}")

    def observe(self):
        """現在の時刻、関節角、指令角、物体中心を辞書で返す。古い受信値は拒否する。"""
        if (
            time.monotonic() - self.joint_wall > 5
            or time.monotonic() - self.pose_wall > 5
        ):
            raise RuntimeError("Stale simulator observations")
        if np.max(time.monotonic() - self.action_wall) > 5:
            raise RuntimeError("Stale controller command observations")
        return {
            "t": self.sim,
            "q": self.q.tolist(),
            "action": self.action.tolist(),
            "cube_center": self.cube.tolist(),
        }

    def close(self):
        """Gazeboの位置購読プロセスを終了し、ROSノードを破棄する。"""
        self.pose_reader.terminate()
        self.pose_reader.wait(timeout=5)
        self.destroy_node()


def evaluate(rows):
    """観測記録から、持ち上げ・配置誤差・高さ・安定性を判定する。

    Args:
        rows: t（秒）とcube_center（世界座標m）を含む観測記録のリスト。

    Returns:
        success、最終XY誤差m、持ち上げ量m、最終位置などを含む辞書。
        配置誤差はGOALに対して計算する。"""
    positions = np.array([r["cube_center"] for r in rows])
    t = np.array([r["t"] for r in rows])
    # 最後の1秒で、置き位置・高さ・安定性を確認する。
    final = positions[t >= t[-1] - 1.0]
    lift = float(positions[:, 2].max() - START[2])
    xy_error = float(np.linalg.norm(final[-1, :2] - GOAL[:2]))
    stable = float(np.max(np.linalg.norm(final - final[-1], axis=1))) < 0.005
    on_table = bool(np.max(np.abs(final[:, 2] - GOAL[2])) < 0.01)
    success = lift > 0.02 and xy_error < 0.03 and stable and on_table
    return {
        "success": bool(success),
        "lift_height_m": lift,
        "final_xy_error_m": xy_error,
        "stable_last_second": bool(stable),
        "on_table": on_table,
        "final_center_m": final[-1].tolist(),
        "duration_sim_seconds": float(t[-1] - t[0]),
        "criteria": "lift > 2cm; final XY within 3cm of (0.25,0); center height within 1cm of table-rest height; displacement <5mm during final 1sim-second",
    }
