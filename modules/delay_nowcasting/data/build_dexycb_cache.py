"""DexYCB -> canonical frame Parquet cache.

  python -m modules.delay_nowcasting.data.build_dexycb_cache \
      --config modules/delay_nowcasting/configs/data/dexycb_v1.yaml

HOT3D cache 와 **같은 schema** 를 내보내므로 window builder, baseline, evaluator,
학습 코드를 그대로 쓴다. 데이터셋을 바꿔도 비교 경로가 하나로 유지된다 (B 계획 §5).

HOT3D 와 다른 점:
  - joint 순서 변환이 없다. `joint_3d` 가 이미 canonical 21 이다.
  - timestamp 가 없어 frame index x 33.333 ms 로 만든다. 간격 지터가 없다.
  - 시퀀스가 약 74 frame(2.5 초)으로 짧고 대신 개수가 많다.
  - 실제 depth 가 있어 조건 C 에서 root lifting 을 GT 에 기대지 않아도 된다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, ResolvedConfig, load_config, run_metadata
from .adapters.dexycb import (
    SOURCE_DATASET,
    read_meta,
    choose_camera,
    frame_timestamps_ns,
    read_extrinsics,
    read_joint_track,
)
from .canonical_schema import (
    NUM_JOINTS,
    assert_metric_units,
    canonical_frame_schema,
    invert_transform,
    transform_points,
    validate_canonical_frames,
)

SEQUENCE_GLOB = "20*"


def discover_sequences(root: Path, subjects: list[str]) -> list[Path]:
    out = []
    for subject_dir in sorted(root.glob("*-subject-*")):
        tag = subject_dir.name.split("-subject-")[-1]
        if f"subject-{tag}" not in subjects and tag not in subjects:
            continue
        out += sorted(p for p in subject_dir.glob(SEQUENCE_GLOB) if p.is_dir())
    return out


def build_sequence(sequence_dir: Path, calibration_root: Path,
                   serial: str | None = None) -> tuple[pl.DataFrame, dict]:
    info = read_meta(sequence_dir)
    extrinsics = read_extrinsics(calibration_root, info.extrinsics_id)
    serial = serial or choose_camera(sequence_dir, info)

    joints_camera, gt_valid, in_frame = read_joint_track(sequence_dir, serial, info.num_frames)
    # 21 joint 가 모두 화면 안에 있는 프레임만 쓴다. 화면 밖은 이미지 기반 방법이
    # 애초에 볼 수 없으므로 학습에도 평가에도 넣지 않는다. 카메라마다 다르다.
    valid = gt_valid & in_frame
    camera_to_world = extrinsics[serial]
    joints_world = np.full_like(joints_camera, np.nan)
    rows = np.flatnonzero(valid)
    if len(rows):
        joints_world[rows] = transform_points(camera_to_world, joints_camera[rows])
        assert_metric_units(joints_world[rows])

    n = info.num_frames
    # 카메라마다 별도 track 이다. sequence_id 에 카메라를 넣어야 window 가 카메라
    # 경계를 넘지 않고, 통계에서도 카메라별로 따로 집계된다.
    track_key = f"{info.sequence_id}@{serial}"
    table = pl.DataFrame({
        "sequence_id": np.full(n, track_key),
        "subject_id": np.full(n, info.subject),
        "frame_idx": np.arange(n, dtype=np.int64),
        "timestamp_ns": frame_timestamps_ns(n),
        "handedness": np.full(n, info.handedness),
        "track_id": np.full(n, f"{track_key}:{info.handedness}"),
        "valid": valid,
        "joints_world": joints_world.reshape(n, -1).astype(np.float32),
        "joints_camera": joints_camera.reshape(n, -1).astype(np.float32),
        "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
        "camera_to_world": np.tile(camera_to_world.reshape(1, 16), (n, 1)).astype(np.float32),
        "source_dataset": np.full(n, SOURCE_DATASET),
        "source_frame_key": [f"{track_key}:{i}" for i in range(n)],
    }, schema=canonical_frame_schema())

    return table, {
        "sequence_id": track_key, "recording_id": info.sequence_id, "subject_id": info.subject,
        "camera": serial, "handedness": info.handedness,
        "num_frames": n, "n_valid": int(valid.sum()),
        "n_gt_valid": int(gt_valid.sum()), "n_in_frame": int(in_frame.sum()),
        "mano_calib": info.mano_calib, "extrinsics_id": info.extrinsics_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DexYCB canonical frame cache 생성")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-cameras", action="store_true",
                    help="8 대를 모두 track 으로 쓴다. GT 궤적은 카메라와 무관하게 같지만, "
                         "가시 프레임과 WiLoR 오차는 카메라마다 다르다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    root = Path(dataset["root"])
    calibration_root = Path(dataset["calibration_root"])
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config     : {cfg.path} (hash {cfg.hash})")
    print(f"output dir : {out_dir}")
    summary = {"config_hash": cfg.hash, "run": run_metadata(cfg), "splits": {}}
    for split in args.splits:
        subjects = list(cfg["split"][split])
        sequences = discover_sequences(root, subjects)[: args.limit]
        if not sequences:
            print(f"  [{split}] 아직 추출된 sequence 가 없다: {subjects}")
            continue
        tables, infos = [], []
        for i, sequence_dir in enumerate(sequences, 1):
            meta = read_meta(sequence_dir)
            serials = meta.serials if args.all_cameras else [None]
            for serial in serials:
                table, info = build_sequence(sequence_dir, calibration_root, serial)
                if info["n_valid"] == 0:      # 손이 한 프레임도 안 보이는 카메라는 버린다
                    continue
                tables.append(table)
                infos.append(info)
            if i % 20 == 0 or i == len(sequences):
                print(f"  [{split}] {i}/{len(sequences)} recordings, {len(tables)} tracks",
                      flush=True)
        combined = pl.concat(tables)
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
