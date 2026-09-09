"""HOI4D -> canonical frame Parquet cache.

  python -m modules.delay_nowcasting.data.build_hoi4d_cache \
      --config modules/delay_nowcasting/configs/data/hoi4d_v1.yaml

DexYCB/HOT3D cache 와 **같은 schema** 를 내보내므로 window builder, baseline, evaluator,
학습 코드를 그대로 쓴다.

앞선 둘과 다른 점은 `adapters/hoi4d.py` 머리말에 정리했다. 요약하면 3D 관절을 MANO
forward 로 만들고, 15 fps 고정이며, world frame 이 없어 `camera_to_world` 가 단위행렬이다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config, run_metadata
from ._shards import CHUNK, chunks, done, merge, save
from .adapters.hoi4d import (
    HANDEDNESS,
    SOURCE_DATASET,
    ManoJointSolver,
    SequenceInfo,
    discover_sequences,
    frame_timestamps_ns,
    read_intrinsics,
    read_joint_track,
)
from .canonical_schema import (
    NUM_JOINTS,
    assert_metric_units,
    canonical_frame_schema,
    validate_canonical_frames,
)

IDENTITY = np.eye(4, dtype=np.float64)


def build_sequence(info: SequenceInfo, solver: ManoJointSolver,
                   intrinsics: np.ndarray) -> tuple[pl.DataFrame, dict]:
    joints, gt_valid, in_frame = read_joint_track(info, solver, intrinsics)
    valid = gt_valid & in_frame
    if valid.any():
        assert_metric_units(joints[valid])

    n = len(info.frames)
    track_key = info.sequence_id
    # world frame 이 없다. joints_world 를 camera 와 같게 두고 변환을 단위행렬로 둔다.
    # 2.5D 경로는 joints_camera 만 읽으므로 학습·평가에 영향이 없다.
    table = pl.DataFrame({
        "sequence_id": np.full(n, track_key),
        "subject_id": np.full(n, info.camera),
        "frame_idx": info.frames.astype(np.int64),
        "timestamp_ns": frame_timestamps_ns(info.frames),
        "handedness": np.full(n, HANDEDNESS),
        "track_id": np.full(n, f"{track_key}:{HANDEDNESS}"),
        "valid": valid,
        "joints_world": joints.reshape(n, -1).astype(np.float32),
        "joints_camera": joints.reshape(n, -1).astype(np.float32),
        "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
        "camera_to_world": np.tile(IDENTITY.reshape(1, 16), (n, 1)).astype(np.float32),
        "source_dataset": np.full(n, SOURCE_DATASET),
        "source_frame_key": [f"{track_key}:{i}" for i in info.frames.tolist()],
    }, schema=canonical_frame_schema())

    return table, {"sequence_id": track_key, "camera": info.camera, "handedness": HANDEDNESS,
                   "num_frames": n, "n_valid": int(valid.sum()),
                   "n_gt_valid": int(gt_valid.sum()), "n_in_frame": int(in_frame.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description="HOI4D canonical frame cache 생성")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    root = Path(dataset["root"])
    solver = ManoJointSolver(REPO_ROOT / dataset["mano_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config     : {cfg.path} (hash {cfg.hash})")
    print(f"output dir : {out_dir}")
    summary = {"config_hash": cfg.hash, "run": run_metadata(cfg), "splits": {}}
    for split in args.splits:
        cameras = list(cfg["split"][split])
        sequences = discover_sequences(root, cameras)[: args.limit]
        if not sequences:
            print(f"  [{split}] 시퀀스가 없다: {cameras}")
            continue
        intrinsics = {c: read_intrinsics(root, c) for c in cameras}
        parts_dir = out_dir / f"canonical_frames_{split}_parts"
        groups = chunks(sequences, args.chunk)
        started, processed = time.time(), 0
        for index, group in enumerate(groups):
            ids = [s.sequence_id for s in group]
            if done(parts_dir, index, ids):
                continue
            tables, infos = [], []
            for info in group:
                table, meta = build_sequence(info, solver, intrinsics[info.camera])
                tables.append(table)
                infos.append(meta)
            save(parts_dir, index, pl.concat(tables), infos, ids)
            processed += len(group)
            rate = processed / max(time.time() - started, 1e-9)
            print(f"  [{split}] {index + 1}/{len(groups)} chunk ({rate * 60:.0f} seq/min)",
                  flush=True)
        combined, infos = merge(parts_dir, len(groups))
        validate_canonical_frames(combined)
        path = out_dir / f"canonical_frames_{split}.parquet"
        combined.write_parquet(path)
        summary["splits"][split] = {"path": str(path), "n_sequences": len(sequences),
                                    "n_rows": combined.height,
                                    "n_valid": int(combined["valid"].sum()),
                                    "sequences": infos}
        print(f"  [{split}] {len(sequences)} sequences, {combined.height} rows "
              f"({combined['valid'].sum()} valid) -> {path}")

    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest   : {out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
