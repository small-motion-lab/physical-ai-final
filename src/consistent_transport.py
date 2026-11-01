"""手本の接近目標をそろえ、変更前後を比較する実験の実行処理。"""

import argparse, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import cartesian_transport as c
from prepare_consistent import prepare

BASE = c.OUT
OUT = c.ROOT / "local/outputs/consistent_transport"
LABELS = [f"teacher_corner{i}_attempt1" for i in range(1, 5)]


def digest(path):
    """pathで指定したファイルのSHA-256を文字列で返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_timeline():
    """変更前4手本の区切り時刻を平均し、共通の時間軸を秒の配列で返す。"""
    anchors = []
    stage_lists = []
    for label in LABELS:
        d = json.loads((BASE / (label + ".json")).read_text())
        t0 = d["samples"][0]["t"]
        t1 = d["samples"][-1]["t"]
        stage = [s for s in d["stages"] if s[0] > 2 and t0 < s[1] < t1]
        anchors.append(np.array([t0] + [s[1] for s in stage] + [t1]) - t0)
        stage_lists.append([s[0] for s in stage])
    assert all(s == stage_lists[0] for s in stage_lists)
    return np.mean(anchors, axis=0)


def main():
    """手本収集、学習、モデル固定、36要求の評価を実行する。--resumeで中断から再開する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and not args.resume:
        raise RuntimeError(
            "Intervention exists; use --resume without overwriting records"
        )
    # 比較条件を固定するため、変更前モデルが実験時のものか確認する。
    baseline_hash = digest(BASE / "policy.npz")
    assert (
        baseline_hash
        == "20b20831312b5f3202f4b0d63c09a767404b43bbc75dbb51d606a14e92c4581f"
    )
    timeline = reference_timeline()
    order = [
        (m, i, r) for r in range(1, 4) for m in ("learned", "direct") for i in range(6)
    ]
    random.Random(1105).shuffle(order)
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        change = protocol["intervention"]
        assert protocol["baseline_policy_sha256"] == baseline_hash
        assert (
            protocol["training_xy_m"] == c.TRAIN.tolist()
            and protocol["evaluation_xy_m"] == c.TEST.tolist()
        )
        assert protocol["evaluation_order"] == [list(x) for x in order]
        assert np.allclose(
            protocol["reference_timeline_seconds"], timeline, atol=1e-12, rtol=0
        )
        assert (
            digest(Path("/tmp/pai-consistent/teacher_src/src/pick_and_place.cpp"))
            == change["teacher_cpp_sha256"]
        )
    else:
        change = prepare(BASE, OUT)
        protocol = dict(
            experiment_id="CONSISTENT_APPROACH_20261101",
            intervention=change,
            baseline_policy_sha256=baseline_hash,
            training_xy_m=c.TRAIN.tolist(),
            evaluation_xy_m=c.TEST.tolist(),
            source_teacher_hashes={
                label: digest(BASE / (label + ".json")) for label in LABELS
            },
            reference_timeline_seconds=timeline.tolist(),
            evaluation_order=order,
            speed_fraction=0.75,
            maximum_teacher_attempts_per_corner=2,
            selection="First successful teacher with approach target within 0.02rad",
            fixed_features=True,
            fixed_scoring=True,
            all_outcomes_retained=True,
            hypothesis="Making the common approach joint configuration consistent removes the E6 pre-grasp command-limit violation; check physical transport and all unchanged evaluation locations as well",
            limitation="New independent MoveIt teacher executions can also vary in path and contact; a before/after case comparison, not an isolated proof of every causal factor",
        )
        protocol_path.write_text(json.dumps(protocol, indent=2))
    with (OUT / "build.log").open("w") as log:
        subprocess.run(
            [
                "cmake",
                "-S",
                "/tmp/pai-consistent/teacher_src",
                "-B",
                "/tmp/pai-consistent/build",
                "-DBUILD_TESTING=OFF",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                "/tmp/pai-consistent/build",
                "--target",
                "pick_and_place",
                "-j2",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    c.OUT = OUT
    c.b.rclpy.init()
    arm = c.b.Arm()
    events_path = OUT / "infrastructure_events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []

    def initial_retry(fn, label):
        """初期化通信だけが失敗した場合、記録を残して1回再試行する。

        Args:
            fn: 手本収集または搬送を行う、引数なしの呼び出し。
            label: 障害記録に残す手本名または試行名。

        Returns:
            fnの戻り値。搬送を始めた後の失敗は再試行しない。"""
        for attempt in (1, 2):
            try:
                return fn()
            except RuntimeError as e:
                if "Episode reset failed" not in str(e) or attempt == 2:
                    raise
                events.append(
                    dict(
                        label=label,
                        phase="initialization",
                        error=str(e),
                        physical_task_started=False,
                    )
                )
                (OUT / "infrastructure_events.json").write_text(
                    json.dumps(events, indent=2)
                )
                print("Retry initialization only", flush=True)

    try:
        # 各置き先で最初に成功した手本を採用し、不成功の記録も残す。
        arm.ready()
        chosen = []
        for i, xy in enumerate(c.TRAIN, 1):
            for attempt in (1, 2):
                label = f"teacher_corner{i}_attempt{attempt}"
                print("COLLECT " + label, flush=True)
                path = OUT / (label + ".json")
                if not path.exists():
                    initial_retry(
                        lambda: c.collect(
                            arm,
                            label,
                            xy,
                            "/tmp/pai-consistent/teacher.launch.py",
                            "MoveIt teacher with common first picking-approach joint target; fixed speed/acceleration 0.5",
                        ),
                        label,
                    )
                demo = json.loads(path.read_text())
                ok = demo["result"]["success"] and demo["process_returncode"] == 0
                stamp = dict(demo["stages"]).get(5)
                target_ok = False
                if stamp is not None:
                    row = min(demo["samples"], key=lambda r: abs(r["t"] - stamp))
                    target_ok = bool(
                        np.max(
                            np.abs(
                                np.array(row["action"][:4])
                                - change["approach_joint_target_rad"]
                            )
                        )
                        < 0.02
                    )
                elif not any(e.get("label") == label for e in events):
                    events.append(
                        dict(
                            label=label,
                            phase="teacher_startup",
                            reason="Missing approach stage after MoveIt child failed to fetch current state; raw record and log retained",
                            physical_task_started=False,
                        )
                    )
                    events_path.write_text(json.dumps(events, indent=2))
                if ok and target_ok:
                    chosen.append(label)
                    break
            else:
                raise RuntimeError("No valid intervention teacher at corner " + str(i))
        (OUT / "teacher_selection.json").write_text(
            json.dumps(dict(labels=chosen, criterion=protocol["selection"]), indent=2)
        )
        if (OUT / "frozen_model.json").exists():
            policy_hash = json.loads((OUT / "frozen_model.json").read_text())[
                "policy_sha256"
            ]
            assert digest(OUT / "policy.npz") == policy_hash
            assert (
                json.loads((OUT / "training.json").read_text())["training_labels"]
                == chosen
            )
        else:
            if (OUT / "policy.npz").exists():
                raise RuntimeError("Unfrozen model exists; inspect before resuming")
            c.train(chosen, timeline_reference=timeline)
            policy_hash = digest(OUT / "policy.npz")
        model = np.load(OUT / "policy.npz")
        assert (
            abs(
                float(model["duration"])
                - float(np.load(BASE / "policy.npz")["duration"])
            )
            < 1e-9
        )
        (OUT / "frozen_model.json").write_text(
            json.dumps(
                dict(
                    policy_sha256=policy_hash,
                    duration_sim_seconds=float(model["duration"]),
                ),
                indent=2,
            )
        )
        # 固定したモデルを使い、保存済みの順序で両方式を評価する。
        for number, (mode, i, repeat) in enumerate(order, 1):
            label = f"eval_{mode}_goal{i+1}_r{repeat}"
            print(f"EVALUATE {number}/36 {label}", flush=True)
            assert (
                digest(OUT / "policy.npz") == policy_hash
                and digest(BASE / "policy.npz") == baseline_hash
            )
            if (OUT / (label + ".json")).exists():
                continue
            initial_retry(
                lambda: c.deploy(
                    arm, label, c.TEST[i], mode, 0.75, allow_extrapolation=i >= 2
                ),
                label,
            )
        (OUT / "completed.json").write_text(
            json.dumps(
                dict(
                    completed_outcomes=36,
                    policy_sha256=policy_hash,
                    baseline_policy_sha256=baseline_hash,
                ),
                indent=2,
            )
        )
        print("INTERVENTION_COMPLETE", flush=True)
    finally:
        arm.close()
        c.b.rclpy.shutdown()


if __name__ == "__main__":
    main()
