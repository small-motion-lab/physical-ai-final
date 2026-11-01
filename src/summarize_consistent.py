"""保存した物体位置から、変更前後の比較結果を再集計する。"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    """pathで指定したファイルのSHA-256を文字列で返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(root):
    """モデル・手本・評価条件の一致を確認し、集計に使う情報を読み込む。

    Args:
        root: local/outputsを含むリポジトリのパス。

    Returns:
        変更前後のデータフォルダ、評価条件、モデル情報、変更前の集計。
        記録のハッシュや学習条件が一致しなければ処理を止める。
    """
    before_dir = root / "local/outputs/cartesian_transport"
    after_dir = root / "local/outputs/consistent_transport"
    protocol = json.loads((after_dir / "protocol.json").read_text())
    frozen = json.loads((after_dir / "frozen_model.json").read_text())

    assert digest(before_dir / "policy.npz") == protocol["baseline_policy_sha256"]
    assert digest(after_dir / "policy.npz") == frozen["policy_sha256"]

    before = json.loads((before_dir / "summary.json").read_text())
    for trial in before["trials"]:
        assert digest(before_dir / (trial["label"] + ".json")) == trial["sha256"]
    for label, expected in protocol["source_teacher_hashes"].items():
        assert digest(before_dir / (label + ".json")) == expected

    training = json.loads((after_dir / "training.json").read_text())
    for label, expected in training["source_hashes"].items():
        assert digest(after_dir / (label + ".json")) == expected
    assert training["training_xy_m"] == protocol["training_xy_m"]
    assert training["held_out_xy_m"] == protocol["evaluation_xy_m"]
    assert (
        training["reference_timeline_seconds"] == protocol["reference_timeline_seconds"]
    )
    assert training["duration_sim_seconds"] == frozen["duration_sim_seconds"]

    return before_dir, after_dir, protocol, frozen, before


def score_trial(directory, mode, goal_index, repeat, order, xy, speed):
    """1要求の位置記録から成否を再計算し、集計用の辞書を返す。

    Args:
        directory: 変更後の評価JSONを保存したフォルダ。
        mode: learned（回帰モデル）またはdirect（直接補間・外挿）。
        goal_index: 評価位置の番号。0〜5がE1〜E6に対応する。
        repeat: 同じ位置・方式での繰り返し番号（1〜3）。
        order: 全要求の中で何番目かを示す番号（1始まり）。
        xy: 要求した置き先のXY座標。単位はm。
        speed: 手本の時間軸に対する実行速度の比率。

    Returns:
        実行の有無、搬送成否、配置誤差mm、失敗理由などの辞書。
    """
    label = f"eval_{mode}_goal{goal_index + 1}_r{repeat}"
    path = directory / (label + ".json")
    if not path.exists():
        raise RuntimeError("Pending outcome: " + label)

    record = json.loads(path.read_text())
    saved_result = record["result"]
    samples = record["samples"]
    assert saved_result["label"] == label
    assert saved_result["mode"] == mode
    assert saved_result["requested_goal_m"][:2] == xy
    assert saved_result["speed_fraction"] == speed

    row = dict(
        label=label,
        order=order,
        mode=mode,
        goal_id=goal_index + 1,
        repeat=repeat,
        xy_m=xy,
        executed=bool(samples),
        task_success=False,
        extrapolation_success=None,
        error_mm=None,
        failure_criteria=[],
        sha256=digest(path),
    )

    if not samples:
        # 指令検査で止めた要求は、物理実行とは区別する。
        assert saved_result["execution_status"] == "rejected_joint_limit"
        row["failure_criteria"] = ["joint_limit"]
        return row

    assert all(b["t"] > a["t"] for a, b in zip(samples, samples[1:]))
    tail = [
        sample["cube_center"]
        for sample in samples
        if sample["t"] >= samples[-1]["t"] - 1
    ]
    final = tail[-1]

    # 最終位置だけでなく、持ち上げ量と最後の1秒の安定性も調べる。
    error = math.dist(final[:2], xy)
    lift = max(sample["cube_center"][2] for sample in samples) - 1.04
    stable = max(math.dist(position, final) for position in tail) < 0.005
    on_table = max(abs(position[2] - 1.04) for position in tail) < 0.01
    success = lift > 0.02 and error < 0.03 and stable and on_table
    assert success == saved_result["success"]
    assert abs(error - saved_result["final_xy_error_m"]) < 1e-10

    row.update(
        task_success=success,
        error_mm=error * 1000,
        final_center_m=final,
        lift_mm=lift * 1000,
    )
    row["failure_criteria"] = [
        key
        for key, failed in [
            ("insufficient_lift", lift <= 0.02),
            ("placement_error", error >= 0.03),
            ("unstable", not stable),
            ("off_table", not on_table),
        ]
        if failed
    ]

    if goal_index >= 2:
        # 範囲外の目標では、実際の位置も学習領域の境界を越えたか調べる。
        outside = all(
            final[j] < [0.20, -0.06][j] for j in range(2) if xy[j] < [0.20, -0.06][j]
        ) and all(
            final[j] > [0.26, 0.06][j] for j in range(2) if xy[j] > [0.26, 0.06][j]
        )
        row["extrapolation_success"] = success and outside
        assert row["extrapolation_success"] == saved_result["extrapolation_success"]

    return row


def group_trials(rows):
    """各要求の辞書rowsを位置・方式別に集計し、平均誤差や成功数を返す。"""
    groups = []
    for goal_id in range(1, 7):
        for mode in ("learned", "direct"):
            trials = [
                row for row in rows if row["goal_id"] == goal_id and row["mode"] == mode
            ]
            errors = [row["error_mm"] for row in trials if row["executed"]]
            groups.append(
                dict(
                    goal_id=goal_id,
                    mode=mode,
                    requested=len(trials),
                    executed=sum(row["executed"] for row in trials),
                    successes=sum(row["task_success"] for row in trials),
                    rejected=sum(not row["executed"] for row in trials),
                    extrapolation_successes=sum(
                        row["extrapolation_success"] is True for row in trials
                    ),
                    mean_mm=statistics.mean(errors) if errors else None,
                    sample_sd_mm=statistics.stdev(errors) if len(errors) > 1 else None,
                    errors_mm=errors,
                    failure_counts={
                        key: sum(key in row["failure_criteria"] for row in trials)
                        for key in [
                            "joint_limit",
                            "insufficient_lift",
                            "placement_error",
                            "unstable",
                            "off_table",
                        ]
                    },
                )
            )
    return groups


def summarize_teachers(directory):
    """directory内の採用済み手本について、誤差と接近時の関節指令を返す。"""
    selection = json.loads((directory / "teacher_selection.json").read_text())
    teachers = []
    for label in selection["labels"]:
        path = directory / (label + ".json")
        record = json.loads(path.read_text())
        stamp = dict(record["stages"])[5]
        sample = min(record["samples"], key=lambda row: abs(row["t"] - stamp))
        teachers.append(
            dict(
                label=label,
                error_mm=record["result"]["final_xy_error_m"] * 1000,
                success=record["result"]["success"],
                approach_command_deg=[
                    math.degrees(value) for value in sample["action"][:4]
                ],
                sha256=digest(path),
            )
        )
    return teachers


def summarize(root):
    """リポジトリrootの記録を読み、全要求と位置別集計をまとめた辞書を返す。"""
    before_dir, after_dir, protocol, frozen, before = load_records(root)
    rows = []
    for order, (mode, goal_index, repeat) in enumerate(protocol["evaluation_order"], 1):
        xy = protocol["evaluation_xy_m"][goal_index]
        rows.append(
            score_trial(
                after_dir,
                mode,
                goal_index,
                repeat,
                order,
                xy,
                protocol["speed_fraction"],
            )
        )

    return dict(
        experiment_id=protocol["experiment_id"],
        protocol=protocol,
        policy_sha256=frozen["policy_sha256"],
        baseline_policy_sha256=protocol["baseline_policy_sha256"],
        baseline_groups=before["groups"],
        groups=group_trials(rows),
        trials=rows,
        teachers=summarize_teachers(after_dir),
        physical_rollouts=sum(row["executed"] for row in rows),
        rejected=sum(not row["executed"] for row in rows),
        successes=sum(row["task_success"] for row in rows),
        baseline_summary_sha256=digest(before_dir / "summary.json"),
        all_raw_scores_independently_verified=True,
    )


def write_summary(summary, output_dir):
    """集計辞書summaryをoutput_dirへJSONと試行一覧CSVで保存する。戻り値はない。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = summary["trials"]
    with (output_dir / "trials.csv").open("w") as file:
        writer = csv.DictWriter(
            file, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    """コマンドラインの--rootと--outを読み取り、再集計結果を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.root)
    write_summary(summary, args.out)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ["physical_rollouts", "successes", "rejected", "groups"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
