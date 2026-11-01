"""同梱データの展開と、再学習・搬送・評価の実行入口。"""

import argparse, hashlib, json, random, shutil, sys, zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONDITIONS = {"before": "cartesian_transport", "after": "consistent_transport"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    """同梱ZIPと全記録のハッシュを確認し、ZIPのパスとハッシュ一覧を返す。"""
    meta = json.loads((ROOT / "reference_data/bundle.json").read_text())
    archive = ROOT / "reference_data" / meta["file"]
    assert sha(archive) == meta["sha256"], "Reference archive changed"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("MANIFEST.json"))
        for name, h in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == h, name
    print(f"Verified reference archive: {len(manifest)} files", flush=True)
    return archive, manifest


def restore():
    # 展開前にハッシュを確認する。手元に異なる記録がある場合は上書きしない。
    """同梱記録をlocal/outputsへ展開する。異なる既存記録は上書きしない。"""
    archive, manifest = verify()
    with zipfile.ZipFile(archive) as z:
        for name in manifest:
            p = PurePosixPath(name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or p.parts[0] not in CONDITIONS.values()
            ):
                raise ValueError("Invalid archive path")
            dest = ROOT / "local/outputs" / name
            data = z.read(name)
            if dest.exists():
                if dest.read_bytes() != data:
                    raise RuntimeError(
                        "Existing local record differs; preserve it before restoring: "
                        + name
                    )
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
    print(
        "Reference records restored without overwriting different local files.",
        flush=True,
    )


def train(condition):
    # 元の記録を残したまま再学習し、生成指令が同梱モデルと一致するか調べる。
    """同梱手本から再学習し、元のモデルとの指令の一致を調べる。

    Args:
        condition: before（変更前）またはafter（接近目標の統一後）。

    Returns:
        なし。モデルと照合結果をlocal/retrained/<condition>に保存する。"""
    import numpy as np
    import cartesian_transport as c

    source = ROOT / "local/outputs" / CONDITIONS[condition]
    meta = json.loads((source / "training.json").read_text())
    out = ROOT / "local/retrained" / condition
    if (out / "policy.npz").exists():
        raise RuntimeError("Retrained model exists; preserve it before training again")
    out.mkdir(parents=True, exist_ok=True)
    for label in meta["training_labels"]:
        shutil.copy2(source / (label + ".json"), out / (label + ".json"))
    c.OUT = out
    timeline = meta.get("reference_timeline_seconds")
    c.train(meta["training_labels"], timeline_reference=timeline)
    original = np.load(source / "policy.npz")
    new = np.load(out / "policy.npz")
    peak = 0.0
    for xy in c.TEST:
        for mode in ("learned", "direct"):
            p, q = c.predict(original, xy, mode, allow_extrapolation=True)
            p2, q2 = c.predict(new, xy, mode, allow_extrapolation=True)
            peak = max(peak, float(np.max(np.abs(q - q2))))
    assert peak < 1e-6, peak
    result = dict(
        condition=condition,
        max_command_difference_rad=peak,
        reference_policy_sha256=sha(source / "policy.npz"),
        retrained_policy_sha256=sha(out / "policy.npz"),
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def simulate(args):
    # 新しい試行は別フォルダに保存し、レポートの評価記録を保つ。
    """指定されたモデルで、搬送または反復評価を行う。

    Args:
        args: CLI引数。commandはrunまたはevaluate、conditionはbefore/after、
            labelは出力名。runの場合はgoal（E1〜E6）とmethodも使う。

    Returns:
        なし。実行条件と結果をlocal/replayへ保存する。"""
    import cartesian_transport as c

    source = ROOT / "local/outputs" / CONDITIONS[args.condition]
    if not args.label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use letters, digits, underscore or hyphen for label")
    out = ROOT / "local/replay" / f"{args.condition}_{args.label}"
    if out.exists():
        raise RuntimeError("Output exists; use another label to keep all trials")
    out.mkdir(parents=True)
    shutil.copy2(source / "policy.npz", out / "policy.npz")
    c.OUT = out
    protocol = dict(
        condition=args.condition,
        source_policy_sha256=sha(source / "policy.npz"),
        speed_fraction=0.75,
    )
    if args.command == "evaluate":
        order = [
            (m, i, r)
            for r in range(1, 4)
            for m in ("learned", "direct")
            for i in range(6)
        ]
        random.Random(1105).shuffle(order)
    else:
        order = [(args.method, int(args.goal[1:]) - 1, 1)]
    protocol["order"] = order
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    c.b.rclpy.init()
    arm = c.b.Arm()
    try:
        arm.ready()
        for mode, i, r in order:
            c.deploy(
                arm,
                f"{args.label}_{mode}_E{i+1}_r{r}",
                c.TEST[i],
                mode,
                0.75,
                allow_extrapolation=i >= 2,
            )
        assert sha(out / "policy.npz") == protocol["source_policy_sha256"]
    finally:
        arm.close()
        if c.b.rclpy.ok():
            c.b.rclpy.shutdown()


def main():
    """コマンドラインのverify/restore/train/run/evaluateを読み、対応する処理を呼ぶ。"""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("restore")
    t = sub.add_parser("train")
    t.add_argument("--condition", choices=CONDITIONS, required=True)
    for command in ("run", "evaluate"):
        r = sub.add_parser(command)
        r.add_argument("--condition", choices=CONDITIONS, default="after")
        r.add_argument("--label", required=True)
        if command == "run":
            r.add_argument(
                "--goal", choices=[f"E{i}" for i in range(1, 7)], default="E6"
            )
            r.add_argument("--method", choices=["learned", "direct"], default="learned")
    a = p.parse_args()
    if a.command == "verify":
        verify()
    elif a.command == "restore":
        restore()
    elif a.command == "train":
        train(a.condition)
    else:
        simulate(a)


if __name__ == "__main__":
    main()
