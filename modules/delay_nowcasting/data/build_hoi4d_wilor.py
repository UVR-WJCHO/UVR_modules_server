"""HOI4D 이미지에 WiLoR 를 돌려 배포 조건 history 를 만든다.

  python -m modules.delay_nowcasting.data.build_hoi4d_wilor \
      --config modules/delay_nowcasting/configs/data/hoi4d_v1.yaml --split val

DexYCB 판(`build_dexycb_wilor.py`)과 다른 점은 둘이다.

  - RGB 가 mp4 라 순차 디코딩한다. 시퀀스당 파일 하나다.
  - depth 센서 영상을 쓰지 않고 GT 손목 깊이로 lifting 한다. **2.5D 표현에서는
    손목 절대 깊이가 약분되므로**(u_n = (u_px - cx)/fx, rel_z 는 차분) 이 선택이
    학습·평가에 영향을 주지 않는다.

행 순서는 `build_hoi4d_cache.py` 와 정확히 같아야 한다. 두 표를 row index 로 맞대기
때문이며, 그래서 검출이 0 인 시퀀스도 표에는 valid=False 로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config, run_metadata
from .adapters.aria_camera import PinholeCamera
from .adapters.hoi4d import (
    HANDEDNESS,
    IMAGE_SIZE,
    SOURCE_DATASET,
    ManoJointSolver,
    SequenceInfo,
    discover_sequences,
    frame_timestamps_ns,
    read_frames,
    read_intrinsics,
    read_joint_track,
)
from .adapters.wilor_infer import infer_frame, load_tracker, tracker_metadata
from ._shards import CHUNK, chunks, done, merge, save
from .build_hoi4d_cache import IDENTITY
from .canonical_schema import NUM_JOINTS, WRIST, canonical_frame_schema


def run_sequence(tracker, info: SequenceInfo, solver: ManoJointSolver,
                 intrinsics: np.ndarray) -> tuple[pl.DataFrame, dict]:
    gt, gt_valid, in_frame = read_joint_track(info, solver, intrinsics)
    gt_valid = gt_valid & in_frame
    pinhole = PinholeCamera(intrinsics[0, 0], intrinsics[1, 1],
                            intrinsics[0, 2], intrinsics[1, 2], *IMAGE_SIZE)

    n = len(info.frames)
    predicted = np.full((n, NUM_JOINTS, 3), np.nan)
    found = np.zeros(n, dtype=bool)
    position = {int(f): i for i, f in enumerate(info.frames)}

    for frame_index, image in read_frames(info.video, info.frames):
        row = position[frame_index]
        if not gt_valid[row]:
            continue
        detections = infer_frame(tracker, image, pinhole, {HANDEDNESS: gt[row][WRIST]})
        if not detections:
            continue
        det = detections[0]
        z = float(gt[row][WRIST, 2]) + (det.root_relative[:, 2] - det.root_relative[WRIST, 2])
        camera = np.empty((NUM_JOINTS, 3))
        camera[:, 2] = z
        camera[:, 0] = (det.joints_2d[:, 0] - pinhole.cx) / pinhole.fx * z
        camera[:, 1] = (det.joints_2d[:, 1] - pinhole.cy) / pinhole.fy * z
        predicted[row] = camera
        found[row] = True

    valid = found & gt_valid
    track_key = info.sequence_id
    table = pl.DataFrame({
        "sequence_id": np.full(n, track_key),
        "subject_id": np.full(n, info.camera),
        "frame_idx": info.frames.astype(np.int64),
        "timestamp_ns": frame_timestamps_ns(info.frames),
        "handedness": np.full(n, HANDEDNESS),
        "track_id": np.full(n, f"{track_key}:{HANDEDNESS}"),
        "valid": valid,
        "joints_world": predicted.reshape(n, -1).astype(np.float32),
        "joints_camera": predicted.reshape(n, -1).astype(np.float32),
        "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
        "camera_to_world": np.tile(IDENTITY.reshape(1, 16), (n, 1)).astype(np.float32),
        "source_dataset": np.full(n, f"{SOURCE_DATASET}+WiLoR"),
        "source_frame_key": [f"{track_key}:{i}" for i in info.frames.tolist()],
    }, schema=canonical_frame_schema())

    return table, {"sequence_id": track_key, "camera": info.camera, "num_frames": n,
                   "n_valid": int(valid.sum()),
                   "detection_rate": float(valid.sum() / max(gt_valid.sum(), 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description="HOI4D WiLoR history cache")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="이만큼씩 저장한다. 중단해도 구간 단위로 재시작된다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    root = Path(dataset["root"])
    solver = ManoJointSolver(REPO_ROOT / dataset["mano_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = list(cfg["split"][args.split])
    sequences = discover_sequences(root, cameras)[: args.limit]
    intrinsics = {c: read_intrinsics(root, c) for c in cameras}

    tracker = load_tracker(REPO_ROOT)
    metadata = tracker_metadata(tracker, REPO_ROOT)
    metadata["root_lifting"] = "GT wrist depth (2.5D 에서 약분되므로 무관)"
    print(f"config   : {cfg.path} (hash {cfg.hash})")
    print(f"wilor    : {json.dumps(metadata, ensure_ascii=False)}")
    print(f"sequences: {len(sequences)} ({args.split}: {cameras})", flush=True)

    # 구간마다 저장한다. 다시 시작하면 이미 끝난 구간은 건너뛴다 (_shards.py).
    stem = args.out_name or f"wilor_frames_{args.split}"
    parts_dir = out_dir / f"{stem}_parts"
    groups = chunks(sequences, args.chunk)
    resumed = sum(done(parts_dir, i, [s.sequence_id for s in g])
                  for i, g in enumerate(groups))
    if resumed:
        print(f"  이미 끝난 구간 {resumed}/{len(groups)} 건너뜀", flush=True)

    started, processed = time.time(), 0
    for index, group in enumerate(groups):
        ids = [s.sequence_id for s in group]
        if done(parts_dir, index, ids):
            continue
        tables, infos = [], []
        for info in group:
            table, meta = run_sequence(tracker, info, solver, intrinsics[info.camera])
            tables.append(table)
            infos.append(meta)
        save(parts_dir, index, pl.concat(tables), infos, ids)
        processed += len(group)
        elapsed = time.time() - started
        remaining = sum(len(g) for i, g in enumerate(groups) if i > index)
        print(f"  [{index + 1}/{len(groups)}] {processed} seq, {elapsed / 60:.1f} min, "
              f"{processed / max(elapsed, 1e-9) * 60:.1f} seq/min, "
              f"eta {remaining / max(processed / elapsed, 1e-9) / 60:.0f} min", flush=True)

    combined, infos = merge(parts_dir, len(groups))
    combined.write_parquet(out_dir / f"{stem}.parquet")
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(
        {"wilor": metadata, "run": run_metadata(cfg), "sequences": infos},
        indent=2, ensure_ascii=False))
    print(f"\nframes   : {combined.height} rows ({combined['valid'].sum()} valid) "
          f"-> {out_dir / f'{stem}.parquet'}")


if __name__ == "__main__":
    main()
