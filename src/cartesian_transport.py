"""置き先XYを入力とする搬送動作の記録・学習・実行。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pick_place_demo as b

# 学習用の4点と、学習には使わない評価用の6点。座標の単位はm。
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local/outputs/cartesian_transport"
TRAIN = np.array([[0.20, -0.06], [0.26, -0.06], [0.20, 0.06], [0.26, 0.06]])
TEST = np.array(
    [
        [0.215, -0.03],
        [0.245, 0.03],
        [0.16, 0.03],
        [0.27, -0.04],
        [0.23, -0.10],
        [0.23, 0.10],
    ]
)
CENTER = np.array([0.23, 0.0])
SCALE = np.array([0.03, 0.06])


def score(rows, xy):
    """搬送の成否と、学習領域の外へ到達したかを判定する。

    Args:
        rows: 時刻tと物体中心cube_centerを含む観測記録のリスト。
        xy: 要求した置き先のXY座標。単位はm。

    Returns:
        搬送成否、配置誤差、範囲外への到達判定などを含む辞書。"""
    saved = b.GOAL
    b.GOAL = np.array([*xy, 1.04])
    try:
        result = b.evaluate(rows)
    finally:
        b.GOAL = saved
    result["requested_goal_m"] = [*map(float, xy), 1.04]
    result["criteria"] = (
        "lift >2cm; final XY error <3cm; table height within 1cm; final-second displacement <5mm"
    )
    xy = np.asarray(xy)
    final = np.asarray(result["final_center_m"][:2])
    lo = TRAIN.min(axis=0)
    hi = TRAIN.max(axis=0)
    below = xy < lo
    above = xy > hi
    outside = bool(np.any(below | above))
    reached = (
        bool(np.all(final[below] < lo[below]) and np.all(final[above] > hi[above]))
        if outside
        else None
    )
    result.update(
        requested_outside_training=outside,
        achieved_outside_training=reached,
        extrapolation_success=bool(result["success"] and reached) if outside else None,
        execution_status="executed",
    )
    return result


def record(arm, seconds):
    """指定したシミュレーション時間の観測を記録する。

    Args:
        arm: 接続済みのArmインスタンス。
        seconds: 記録するシミュレーション時間。単位は秒。

    Returns:
        約0.05秒間隔の関節状態・指令・物体位置のリスト。"""
    start = arm.sim
    wall = time.monotonic()
    rows = []
    last = -1.0
    while arm.sim - start < seconds:
        arm.spin()
        if arm.sim - last >= 0.05:
            rows.append(arm.observe())
            last = arm.sim
        if time.monotonic() - wall > max(300, seconds * 15):
            raise RuntimeError("Simulator timeout")
    return rows


def collect(
    arm, label, xy, teacher_launch="/tmp/pai-xy/teacher.launch.py", source_note=None
):
    """MoveItの手本を実行し、指令と観測をJSONに保存する。

    Args:
        arm: 接続済みのArmインスタンス。
        label: 保存ファイル名に使う、重複しない手本名。
        xy: 手本の置き先XY座標。単位はm。
        teacher_launch: 手本の動作計画を起動するlaunchファイル。
        source_note: 記録に残す手本の説明。省略時は既定の説明を使う。

    Returns:
        搬送成功かつ手本プロセスが正常終了した場合にTrue。"""
    path = OUT / (label + ".json")
    if path.exists():
        raise RuntimeError("Output exists")
    arm.reset()
    logpath = OUT / (label + ".log")
    with logpath.open("w") as log:
        proc = subprocess.Popen(
            [
                "ros2",
                "launch",
                teacher_launch,
                "example:=pick_and_place",
                "use_sim_time:=true",
                f"goal_x:={xy[0]}",
                f"goal_y:={xy[1]}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        rows = []
        last = -1.0
        wall = time.monotonic()
        try:
            while proc.poll() is None:
                arm.spin()
                if arm.sim - last >= 0.05:
                    rows.append(arm.observe())
                    last = arm.sim
                if time.monotonic() - wall > 300:
                    raise RuntimeError("Teacher timeout")
            rows.extend(record(arm, 2.0))
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
    stages = []
    for line in logpath.read_text().splitlines():
        if "XY_STAGE " in line:
            _, stage, stamp = line[line.index("XY_STAGE ") :].split()
            stages.append([int(stage), float(stamp)])
    result = score(rows, xy)
    doc = dict(
        label=label,
        goal_xy_m=list(map(float, xy)),
        source=source_note
        or "Independent stock MoveIt teacher, Cartesian destination; arm/gripper speed scale 0.5",
        process_returncode=proc.returncode,
        stages=stages,
        result=result,
        samples=rows,
    )
    path.write_text(json.dumps(doc, indent=2))
    print(json.dumps(dict(label=label, **result)), flush=True)
    return result["success"] and proc.returncode == 0


def features(phase, xy):
    # 動作の進行度を時間基底に変換し、置き先のX・Yと組み合わせる。
    """進行度と置き先を、回帰モデルの入力行列へ変換する。

    Args:
        phase: 0〜1の動作進行度を並べた配列。
        xy: 置き先XY座標。単位はm。

    Returns:
        各時刻に161基底×4項を対応させた、時刻数×644の行列。"""
    base = b.features(phase)
    x, y = (np.array(xy) - CENTER) / SCALE
    return np.hstack([base, base * x, base * y, base * x * y])


def train(labels, timeline_reference=None):
    """4本の手本を時刻合わせし、関節指令の回帰モデルを保存する。

    Args:
        labels: TRAINの順に並べた4本の手本名。
        timeline_reference: 共通の動作区切り時刻の配列。単位は秒。
            Noneなら4本の区切り時刻の平均を使う。

    Returns:
        なし。OUTにpolicy.npzとtraining.jsonを保存する。"""
    demos = [json.loads((OUT / (label + ".json")).read_text()) for label in labels]
    if len(demos) != 4 or not all(d["result"]["success"] for d in demos):
        raise ValueError("Four successful corner demonstrations required")
    if not np.allclose([d["goal_xy_m"] for d in demos], TRAIN):
        raise ValueError("Unexpected training locations/order")
    # 把持・持ち上げなどの区切りを使い、4本の手本の時刻をそろえる。
    anchors = []
    stage_lists = []
    for d in demos:
        t0 = d["samples"][0]["t"]
        t1 = d["samples"][-1]["t"]
        stage = [s for s in d["stages"] if s[0] > 2 and t0 < s[1] < t1]
        stage_lists.append([s[0] for s in stage])
        anchors.append(np.array([t0] + [s[1] for s in stage] + [t1]) - t0)
    if not stage_lists[0] or any(s != stage_lists[0] for s in stage_lists):
        raise ValueError("Inconsistent stage markers")
    if any(np.any(np.diff(a) <= 0) for a in anchors):
        raise ValueError("Invalid stage times")
    timeline = (
        np.mean(anchors, axis=0)
        if timeline_reference is None
        else np.asarray(timeline_reference, dtype=float)
    )
    if timeline.shape != anchors[0].shape or np.any(np.diff(timeline) <= 0):
        raise ValueError("Invalid reference timeline")
    grid = np.linspace(0, timeline[-1], 1001)
    phase = grid / grid[-1]
    xx = []
    yy = []
    aligned = []
    for d, a, xy in zip(demos, anchors, TRAIN):
        times = np.array([r["t"] - d["samples"][0]["t"] for r in d["samples"]])
        values = np.array([r["action"] for r in d["samples"]])
        # 共通の時間軸から、各手本で読み出す時刻を求める。
        query = np.interp(grid, timeline, a)
        y = np.column_stack([np.interp(query, times, values[:, j]) for j in range(5)])
        xx.append(features(phase, xy))
        yy.append(y)
        aligned.append(y)
    x = np.vstack(xx)
    y = np.vstack(yy)
    # 正則化を加えた最小二乗法で、5関節の指令角度を学習する。
    w = np.linalg.solve(x.T @ x + 1e-5 * np.eye(x.shape[1]), x.T @ y)
    np.savez(
        OUT / "policy.npz",
        weights=w,
        commands=np.array(aligned),
        duration=timeline[-1],
        training_xy=TRAIN,
    )
    meta = dict(
        training_labels=labels,
        training_xy_m=TRAIN.tolist(),
        held_out_xy_m=TEST.tolist(),
        model="161 normalized Gaussian time bases x [1, normalized x, normalized y, x*y], ridge 1e-5, five joint outputs",
        training_rmse_rad=float(np.sqrt(np.mean((x @ w - y) ** 2))),
        duration_sim_seconds=float(grid[-1]),
        source_hashes={
            label: hashlib.sha256((OUT / (label + ".json")).read_bytes()).hexdigest()
            for label in labels
        },
    )
    if timeline_reference is not None:
        meta["reference_timeline_seconds"] = timeline.tolist()
    (OUT / "training.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def predict(model, xy, mode, allow_extrapolation=False):
    """置き先に応じた動作全体の関節指令を生成する。

    Args:
        model: policy.npzから読み込んだ重み・手本指令・動作時間。
        xy: 置き先XY座標。単位はm。
        mode: learnedなら回帰モデル、directなら直接補間・外挿。
        allow_extrapolation: Trueなら、事前に決めた範囲外の評価点を許可する。

    Returns:
        1,001点の進行度配列と、1,001×5の関節指令行列。角度はrad。"""
    xy = np.array(xy)
    outside = np.any(xy < TRAIN.min(axis=0)) or np.any(xy > TRAIN.max(axis=0))
    # 学習範囲の外は、事前に決めた評価位置だけを許可する。
    if outside and (
        not allow_extrapolation
        or not any(np.allclose(xy, p, atol=1e-9, rtol=0) for p in TEST[2:])
    ):
        raise ValueError("Outside approved evaluation locations")
    phase = np.linspace(0, 1, 1001)
    if mode == "learned":
        return phase, features(phase, xy) @ model["weights"]
    if mode != "direct":
        raise ValueError("Unknown mode")
    # 比較用：4本の指令列を、置き先座標から求めた重みで混ぜる。
    u, v = (xy - TRAIN.min(axis=0)) / (TRAIN.max(axis=0) - TRAIN.min(axis=0))
    weights = np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
    return phase, np.einsum("i,ijk->jk", weights, model["commands"])


def deploy(arm, label, xy, mode, speed=0.75, allow_extrapolation=False):
    """生成指令を検査して実行し、観測と成否を保存する。

    Args:
        arm: 接続済みのArmインスタンス。
        label: 出力ファイルに使う、重複しない試行名。
        xy: 置き先XY座標。単位はm。
        mode: learnedまたはdirect。
        speed: 手本に対する速度比。0.75なら所要時間を1/0.75倍にする。
        allow_extrapolation: 事前に決めた範囲外の評価点を許可するか。

    Returns:
        なし。指令が関節制限を超えた場合も、未実行の記録を保存する。"""
    path = OUT / (label + ".json")
    if path.exists():
        raise RuntimeError("Output exists")
    m = np.load(OUT / "policy.npz")
    phase, q = predict(m, xy, mode, allow_extrapolation)
    # 全時刻の指令を先に確認し、1点でも関節制限を超えれば実行しない。
    limits = np.deg2rad(
        [
            [-150, 150],
            [-104.55, 102.20],
            [-146.48, 140.62],
            [-105.43, 101.32],
            [-40.62, 38.27],
        ]
    )
    if not np.isfinite(q).all() or np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        result = dict(
            label=label,
            mode=mode,
            speed_fraction=speed,
            success=False,
            execution_status="rejected_joint_limit",
            requested_goal_m=[*map(float, xy), 1.04],
            final_xy_error_m=None,
            lift_height_m=None,
            achieved_outside_training=False,
            extrapolation_success=False,
            command_min_rad=q.min(axis=0).tolist(),
            command_max_rad=q.max(axis=0).tolist(),
        )
        path.write_text(json.dumps(dict(result=result, samples=[]), indent=2))
        print(json.dumps(result), flush=True)
        return
    # ここからは計算済みの指令を送る。物体位置による補正は行わない。
    total = float(m["duration"]) / speed
    arm.reset()
    arm.command(0.2 + phase * total, q)
    rows = record(arm, total + 2.0)
    result = score(rows, xy)
    result.update(label=label, mode=mode, speed_fraction=speed)
    path.write_text(json.dumps(dict(result=result, samples=rows), indent=2))
    print(json.dumps(result), flush=True)


def main():
    """コマンドライン引数に応じて、記録・学習・搬送を実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "train", "run"])
    parser.add_argument("--label")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--extrapolate", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args.labels)
        return
    b.rclpy.init()
    arm = b.Arm()
    try:
        arm.ready()
        if args.mode == "collect":
            collect(arm, args.label, [args.x, args.y])
        else:
            deploy(
                arm,
                args.label,
                [args.x, args.y],
                "direct" if args.direct else "learned",
                allow_extrapolation=args.extrapolate,
            )
    finally:
        arm.close()
        b.rclpy.shutdown()


if __name__ == "__main__":
    main()
