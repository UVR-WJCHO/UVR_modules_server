"""DexYCB 이미지에 WiLoR 를 돌려 배포 조건(조건 C) history 를 만든다.

  python -m modules.delay_nowcasting.data.build_dexycb_wilor \
      --config modules/delay_nowcasting/configs/data/dexycb_v1.yaml --split val

HOT3D 판(`build_wilor_cache.py`)과 결정적으로 다른 점: **실제 depth 를 쓴다.**
HOT3D cache 에는 depth 가 없어 root lifting 에 GT wrist depth 를 썼고, 그래서 조건 C 가
실제보다 낙관적이었다. DexYCB 에는 `aligned_depth_to_color_*.png` 가 있어
`main_handtrack.py: lift_pose_cam3d` 와 같은 경로를 그대로 재현한다.

  wrist 픽셀 주변의 유효 depth 최근접값 -> wrist 깊이
  나머지 관절은 WiLoR 의 root-relative z 로 올린다

이미지가 원근 카메라(RealSense)라 fisheye rectification 도 필요 없다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config, run_metadata
from .adapters.aria_camera import PinholeCamera
from .adapters.dexycb import (
    SOURCE_DATASET,
    frame_timestamps_ns,
    read_extrinsics,
    read_intrinsics,
    read_joint_track,
    read_meta,
)
from .adapters.wilor_infer import infer_frame, load_tracker, tracker_metadata
from .build_dexycb_cache import discover_sequences
from .canonical_schema import NUM_JOINTS, WRIST, canonical_frame_schema, transform_points

DEPTH_SCALE_M = 0.001          # DexYCB aligned depth 는 mm 단위 uint16
WRIST_DEPTH_WINDOW = 3
# 서버와 같은 보정: aligned depth 는 손목 '표면'을 찍지만 관절은 그보다 안쪽이다
WRIST_DEPTH_OFFSET_M = 0.013


def sample_wrist_depth(depth_m: np.ndarray, u: float, v: float,
                       window: int = WRIST_DEPTH_WINDOW) -> float | None:
    """(u, v) 주변에서 유효한 최근접 depth. 서버 `_sample_wrist_depth_mm` 와 같은 규칙."""
    h, w = depth_m.shape[:2]
    u, v = int(round(u)), int(round(v))
    patch = depth_m[max(v - window, 0):v + window + 1, max(u - window, 0):u + window + 1]
    valid = patch[patch > 0]
    return float(valid.min()) if valid.size else None


def run_sequence(tracker, sequence_dir: Path, calibration_root: Path,
                 use_gt_depth: bool = False, serial: str | None = None
                 ) -> tuple[pl.DataFrame, dict]:
    info = read_meta(sequence_dir)
    from .build_dexycb_cache import build_sequence

    _, gt_info = build_sequence(sequence_dir, calibration_root, serial)
    serial = gt_info["camera"]
    intrinsics = read_intrinsics(calibration_root, serial)
    pinhole = PinholeCamera(intrinsics["fx"], intrinsics["fy"],
                            intrinsics["cx"], intrinsics["cy"], 640, 480)
    camera_to_world = read_extrinsics(calibration_root, info.extrinsics_id)[serial]

    gt_camera, gt_valid, in_frame = read_joint_track(sequence_dir, serial, info.num_frames)
    gt_valid = gt_valid & in_frame
    predicted = np.full((info.num_frames, NUM_JOINTS, 3), np.nan)
    found = np.zeros(info.num_frames, dtype=bool)
    depth_failed = 0

    for frame in range(info.num_frames):
        color = cv2.imread(str(sequence_dir / serial / f"color_{frame:06d}.jpg"))
        if color is None:
            continue
        depth = cv2.imread(str(sequence_dir / serial / f"aligned_depth_to_color_{frame:06d}.png"),
                           cv2.IMREAD_UNCHANGED)
        # infer_frame 은 wrist 깊이를 인자로 받는다. 실제 depth 로 채우기 위해
        # 먼저 2D 를 얻어야 하므로, GT wrist 를 placeholder 로 넣고 2D 만 쓴다.
        placeholder = {info.handedness: gt_camera[frame][WRIST]} if gt_valid[frame] else {}
        if not placeholder:
            continue
        detections = infer_frame(tracker, color, pinhole, placeholder)
        if not detections:
            continue
        det = detections[0]

        if use_gt_depth or depth is None:
            wrist_depth = float(gt_camera[frame][WRIST, 2])
        else:
            sampled = sample_wrist_depth(depth.astype(np.float32) * DEPTH_SCALE_M,
                                         det.joints_2d[WRIST, 0], det.joints_2d[WRIST, 1])
            if sampled is None:
                depth_failed += 1
                continue
            wrist_depth = sampled + WRIST_DEPTH_OFFSET_M

        z = wrist_depth + (det.root_relative[:, 2] - det.root_relative[WRIST, 2])
        camera = np.empty((NUM_JOINTS, 3))
        camera[:, 2] = z
        camera[:, 0] = (det.joints_2d[:, 0] - pinhole.cx) / pinhole.fx * z
        camera[:, 1] = (det.joints_2d[:, 1] - pinhole.cy) / pinhole.fy * z
        predicted[frame] = camera
        found[frame] = True

    valid = found & gt_valid
    world = np.full_like(predicted, np.nan)
    rows = np.flatnonzero(valid)
    if len(rows):
        world[rows] = transform_points(camera_to_world, predicted[rows])

    n = info.num_frames
    # GT cache 와 track key 를 정확히 맞춰야 두 표의 행이 1:1 로 대응한다.
    track_key = f"{info.sequence_id}@{serial}"
    table = pl.DataFrame({
        "sequence_id": np.full(n, track_key),
        "subject_id": np.full(n, info.subject),
        "frame_idx": np.arange(n, dtype=np.int64),
        "timestamp_ns": frame_timestamps_ns(n),
        "handedness": np.full(n, info.handedness),
        "track_id": np.full(n, f"{track_key}:{info.handedness}"),
        "valid": valid,
        "joints_world": world.reshape(n, -1).astype(np.float32),
        "joints_camera": predicted.reshape(n, -1).astype(np.float32),
        "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
        "camera_to_world": np.tile(camera_to_world.reshape(1, 16), (n, 1)).astype(np.float32),
        "source_dataset": np.full(n, f"{SOURCE_DATASET}+WiLoR"),
        "source_frame_key": [f"{track_key}:{i}" for i in range(n)],
    }, schema=canonical_frame_schema())

    return table, {"sequence_id": track_key, "recording_id": info.sequence_id, "camera": serial,
                   "handedness": info.handedness, "num_frames": n,
                   "n_valid": int(valid.sum()),
                   "detection_rate": float(valid.sum() / max(gt_valid.sum(), 1)),
                   "depth_sample_failures": depth_failed}


def main() -> None:
    ap = argparse.ArgumentParser(description="DexYCB WiLoR history cache (조건 C)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--all-cameras", action="store_true",
                    help="8 대를 모두 처리한다. 카메라마다 WiLoR 오차가 달라 조건 C 표본이 늘어난다")
    ap.add_argument("--use-gt-depth", action="store_true",
                    help="depth 센서 대신 GT wrist depth 를 쓴다(HOT3D 조건과 맞춰 비교할 때)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    calibration_root = Path(dataset["calibration_root"])
    sequences = discover_sequences(Path(dataset["root"]),
                                   list(cfg["split"][args.split]))[: args.limit]

    tracker = load_tracker(REPO_ROOT)
    metadata = tracker_metadata(tracker, REPO_ROOT)
    metadata["root_lifting"] = ("GT wrist depth" if args.use_gt_depth
                                else "실제 aligned depth (서버 lift_pose_cam3d 와 동일 규칙)")
    print(f"config   : {cfg.path} (hash {cfg.hash})")
    print(f"wilor    : {json.dumps(metadata, ensure_ascii=False)}")

    tables, infos = [], []
    started = time.time()
    for i, sequence_dir in enumerate(sequences, 1):
        meta = read_meta(sequence_dir)
        for serial in (meta.serials if args.all_cameras else [None]):
            try:
                table, info = run_sequence(tracker, sequence_dir, calibration_root,
                                           args.use_gt_depth, serial)
            except ValueError:              # 손이 한 프레임도 안 보이는 카메라
                continue
            # 검출이 0 이어도 표에는 남긴다(valid=False). GT cache 와 track 집합이
            # 어긋나면 행 index 가 밀려 target 을 엉뚱한 행에서 읽게 된다.
            tables.append(table)
            infos.append(info)
        if i % 10 == 0 or i == len(sequences):
            rate = i / (time.time() - started)
            print(f"  [{i}/{len(sequences)}] {len(tables)} tracks  "
                  f"({rate * 60:.1f} recording/min)", flush=True)

    combined = pl.concat(tables)
    stem = args.out_name or f"wilor_frames_{args.split}"
    combined.write_parquet(out_dir / f"{stem}.parquet")
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(
        {"wilor": metadata, "run": run_metadata(cfg), "sequences": infos},
        indent=2, ensure_ascii=False))
    print(f"\nframes   : {combined.height} rows ({combined['valid'].sum()} valid) "
          f"-> {out_dir / f'{stem}.parquet'}")


if __name__ == "__main__":
    main()
