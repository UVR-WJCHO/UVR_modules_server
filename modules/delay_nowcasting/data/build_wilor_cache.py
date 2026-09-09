"""HOT3D preview 영상에 WiLoR 를 돌려 배포 조건(조건 C)의 history 를 만든다.

  python -m modules.delay_nowcasting.data.build_wilor_cache \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml --split val

출력은 canonical frame 표와 **같은 schema** 라, window builder 와 evaluator 를 그대로
재사용해 GT history 와 정확히 같은 방식으로 비교할 수 있다 (B 계획 §5).

한계(결과 보고 시 반드시 함께 밝힐 것):
  - root depth 는 GT wrist 를 쓴다. HOT3D cache 에 depth 가 없어서다. 실제 배포는
    depth 센서 오차가 더해지므로 이 조건도 여전히 낙관적이다.
  - 이미지가 preview mp4 라 원본 VRS 보다 압축 열화가 있다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from ..config import REPO_ROOT, ResolvedConfig, load_config, run_metadata
from .adapters.aria_camera import Fisheye624, rectification_map
from .adapters.wilor_infer import (
    MP4_TO_ANNOTATION_ROTATION, infer_frame, load_tracker, tracker_metadata)
from .build_windows import load_canonical_frames
from .canonical_schema import NUM_JOINTS, WRIST, canonical_frame_schema
from .splits import build_split_manifest

SOURCE_DATASET = "HOT3D+WiLoR"


def run_sequence(tracker, cfg: ResolvedConfig, gt_frames: pl.DataFrame, sequence_id: str,
                 rectify_size: int, fov_deg: float) -> tuple[pl.DataFrame, dict]:
    raw_dir = Path(cfg["dataset"]["raw_root"]) / sequence_id
    fisheye = Fisheye624.from_json(raw_dir / "camera_models.json")
    map_x, map_y, pinhole, _ = rectification_map(fisheye, rectify_size, fov_deg)

    sub = gt_frames.filter(pl.col("sequence_id") == sequence_id)
    gt = {}
    for hand in ("LEFT", "RIGHT"):
        g = sub.filter(pl.col("handedness") == hand).sort("frame_idx")
        gt[hand] = {
            "frame_idx": g["frame_idx"].to_numpy(),
            "timestamp_ns": g["timestamp_ns"].to_numpy(),
            "valid": g["valid"].to_numpy(),
            "joints_camera": g["joints_camera"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64),
            "camera_to_world": g["camera_to_world"].to_numpy().reshape(-1, 4, 4).astype(np.float64),
        }

    n_frames = len(gt["RIGHT"]["frame_idx"])
    predicted = {hand: np.full((n_frames, NUM_JOINTS, 3), np.nan) for hand in gt}
    found = {hand: np.zeros(n_frames, dtype=bool) for hand in gt}

    cap = cv2.VideoCapture(str(next(raw_dir.glob("*preview_rgb.mp4"))))
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rectified = cv2.remap(cv2.rotate(frame, MP4_TO_ANNOTATION_ROTATION),
                              map_x, map_y, cv2.INTER_LINEAR)
        wrists = {}
        for hand, data in gt.items():
            pos = np.where(data["frame_idx"] == k)[0]
            if len(pos) and data["valid"][pos[0]]:
                wrists[hand] = data["joints_camera"][pos[0]][WRIST]
        if wrists:
            for det in infer_frame(tracker, rectified, pinhole, wrists):
                pos = np.where(gt[det.handedness]["frame_idx"] == k)[0]
                predicted[det.handedness][pos[0]] = det.joints_camera
                found[det.handedness][pos[0]] = True
        k += 1
    cap.release()

    parts = []
    for hand, data in gt.items():
        valid = found[hand] & data["valid"]
        camera = predicted[hand]
        world = np.full_like(camera, np.nan)
        rows = np.flatnonzero(valid)
        if len(rows):
            T = data["camera_to_world"][rows]
            world[rows] = (np.einsum("nij,nkj->nki", T[:, :3, :3], camera[rows])
                           + T[:, None, :3, 3])
        n = len(valid)
        parts.append(pl.DataFrame({
            "sequence_id": np.full(n, sequence_id),
            "subject_id": np.full(n, sequence_id.split("_")[0]),
            "frame_idx": data["frame_idx"],
            "timestamp_ns": data["timestamp_ns"],
            "handedness": np.full(n, hand),
            "track_id": np.full(n, f"{sequence_id}:{hand}"),
            "valid": valid,
            "joints_world": world.reshape(n, -1).astype(np.float32),
            "joints_camera": camera.reshape(n, -1).astype(np.float32),
            "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
            "camera_to_world": data["camera_to_world"].reshape(n, 16).astype(np.float32),
            "source_dataset": np.full(n, SOURCE_DATASET),
            "source_frame_key": [f"{sequence_id}:{ts}" for ts in data["timestamp_ns"].tolist()],
        }, schema=canonical_frame_schema()))

    info = {
        "sequence_id": sequence_id,
        "n_mp4_frames": k,
        "detection_rate": {hand: float((found[hand] & gt[hand]["valid"]).sum()
                                       / max(gt[hand]["valid"].sum(), 1)) for hand in gt},
    }
    return pl.concat(parts), info


def main() -> None:
    ap = argparse.ArgumentParser(description="WiLoR history cache 생성 (조건 C)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="특정 sequence 만 처리. noise 통계를 train 에서 뽑을 때 쓴다")
    ap.add_argument("--out-name", default=None, help="출력 parquet 이름(확장자 제외)")
    ap.add_argument("--rectify-size", type=int, default=1024)
    ap.add_argument("--fov-deg", type=float, default=110.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    gt_frames = load_canonical_frames(out_dir / f"canonical_frames_{args.split}.parquet")
    available = build_split_manifest(cfg)["sequences"][args.split]
    if args.sequences:
        missing = [s for s in args.sequences if s not in available]
        if missing:
            raise ValueError(f"{args.split} split 에 없는 sequence: {missing}")
        sequences = list(args.sequences)
    else:
        sequences = available[: args.limit]

    tracker = load_tracker(REPO_ROOT)
    metadata = tracker_metadata(tracker, REPO_ROOT)
    print(f"config   : {cfg.path} (hash {cfg.hash})")
    print(f"wilor    : {json.dumps(metadata, ensure_ascii=False)}")

    tables, infos = [], []
    for i, sequence_id in enumerate(sequences, 1):
        table, info = run_sequence(tracker, cfg, gt_frames, sequence_id,
                                   args.rectify_size, args.fov_deg)
        tables.append(table)
        infos.append(info)
        rate = {k: round(v, 3) for k, v in info["detection_rate"].items()}
        print(f"  [{i}/{len(sequences)}] {sequence_id} rows={table.height} detect={rate}",
              flush=True)

    combined = pl.concat(tables)
    stem = args.out_name or f"wilor_frames_{args.split}"
    path = out_dir / f"{stem}.parquet"
    combined.write_parquet(path)
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(
        {"wilor": metadata, "run": run_metadata(cfg),
         "rectify_size": args.rectify_size, "fov_deg": args.fov_deg,
         "sequences": infos}, indent=2, ensure_ascii=False))
    print(f"\nframes   : {combined.height} rows ({combined['valid'].sum()} valid) -> {path}")


if __name__ == "__main__":
    main()
