"""HOT3D -> canonical frame Parquet cache.

  python -m modules.delay_nowcasting.data.build_cache \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml

canonical pose 는 raw 의 MANO parameter 에서 만들고(adapters/hot3d.py 참조),
timestamp 는 cache `frames.parquet` 의 RGB frame timestamp 에 맞춘다. MANO trajectory 는
RGB/SLAM 두 stream 의 timestamp 합집합이라 RGB 것만 남겨야 배포(RGB 구동)와 rate 가 같다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, ResolvedConfig, load_config, run_metadata
from .adapters.aria_camera import Fisheye624, rectification_map
from .adapters.hot3d import (
    SOURCE_DATASET,
    HandPoseRecords,
    ManoJointSolver,
    read_hand_pose_trajectory,
    verify_against_landmarks,
)
from .canonical_schema import (
    NUM_JOINTS,
    assert_metric_units,
    assert_timestamps_ns,
    canonical_frame_schema,
    invert_transform,
    make_transform,
    quat_wxyz_to_matrix,
    transform_points,
    validate_canonical_frames,
)
from .splits import SPLITS, build_split_manifest, subject_of


def _camera_to_world(frames: pl.DataFrame, sequence_dir: Path) -> np.ndarray:
    """(T, 4, 4) world <- RGB camera. headset pose(world <- device) 에 device <- camera 를 붙인다."""
    quat = frames.select(["rot_qw", "rot_qx", "rot_qy", "rot_qz"]).to_numpy()
    trans = frames.select(["t_x", "t_y", "t_z"]).to_numpy()
    world_from_device = make_transform(quat_wxyz_to_matrix(quat), trans)

    models = json.loads((sequence_dir / "camera_models.json").read_text())
    rgb = next(m for m in models if m["label"] == "camera-rgb")
    device_from_camera = make_transform(
        quat_wxyz_to_matrix(np.asarray(rgb["T_Device_Camera"]["quaternion_wxyz"])),
        np.asarray(rgb["T_Device_Camera"]["translation_xyz"]),
    )
    return world_from_device @ device_from_camera


def build_sequence(cfg: ResolvedConfig, solver: ManoJointSolver, sequence_id: str,
                   verify: bool = False, rectify_size: int = 1024,
                   fov_deg: float = 110.0) -> tuple[pl.DataFrame, dict]:
    dataset = cfg["dataset"]
    raw_dir = Path(dataset["raw_root"]) / sequence_id
    cache_dir = Path(dataset["cache_root"]) / "sequences" / f"{sequence_id}__dev{dataset['device_index']}"

    frames = pl.read_parquet(cache_dir / "frames.parquet").sort("frame_idx")
    frame_ts = frames["timestamp_ns"].to_numpy()
    # DexYCB 와 같은 규칙: 21 joint 가 모두 rectified 화면 안에 들어오는 프레임만 쓴다.
    # 이미지 기반 방법이 볼 수 없는 프레임을 학습/평가에 넣지 않기 위해서다.
    fisheye = Fisheye624.from_json(raw_dir / "camera_models.json")
    _, _, pinhole, _ = rectification_map(fisheye, rectify_size, fov_deg)
    assert_timestamps_ns(frame_ts)
    cam_to_world = _camera_to_world(frames, raw_dir)
    world_to_cam = invert_transform(cam_to_world)
    ts_position = {int(ts): i for i, ts in enumerate(frame_ts.tolist())}

    records = read_hand_pose_trajectory(raw_dir, dataset["mano_trajectory_file"])
    cache_hands = pl.read_parquet(cache_dir / "hands.parquet") if verify else None

    n_frames = frames.height
    parts, checks, invalid_counts, in_frame_counts = [], [], {}, {}
    for handedness, hand_records in sorted(records.items()):
        # RGB frame timestamp 에 있는 record 만 남긴다(중복 timestamp 는 첫 것만).
        keep, positions, seen = [], [], set()
        for i, ts in enumerate(hand_records.timestamp_ns.tolist()):
            pos = ts_position.get(ts)
            if pos is not None and ts not in seen:
                seen.add(ts)
                keep.append(i)
                positions.append(pos)
        if not keep:
            continue
        keep = np.asarray(keep)
        positions = np.asarray(positions)

        subset = HandPoseRecords(
            handedness=handedness,
            timestamp_ns=hand_records.timestamp_ns[keep],
            pose=hand_records.pose[keep],
            betas=hand_records.betas[keep],
            rotation=hand_records.rotation[keep],
            translation=hand_records.translation[keep],
        )
        solved = solver.solve(subset)
        assert_metric_units(solved)
        camera_points = transform_points(world_to_cam[positions], solved)
        pixels = pinhole.project(camera_points)
        in_frame = (np.isfinite(pixels).all(axis=(1, 2))
                    & (pixels[..., 0] >= 0).all(axis=1) & (pixels[..., 0] < pinhole.width).all(axis=1)
                    & (pixels[..., 1] >= 0).all(axis=1) & (pixels[..., 1] < pinhole.height).all(axis=1))
        if verify:
            checks.append(verify_against_landmarks(
                solved, subset.timestamp_ns, handedness, cache_hands))

        # pose 가 없는 frame 은 버리지 않고 valid=false 로 남긴다 (B 계획 §4.1).
        # zero pose 로 채우면 정지한 손처럼 보이므로 NaN 을 쓴다.
        valid = np.zeros(n_frames, dtype=bool)
        valid[positions] = in_frame
        joints_world = np.full((n_frames, NUM_JOINTS, 3), np.nan)
        joints_world[positions] = solved
        joints_camera = np.full_like(joints_world, np.nan)
        joints_camera[positions] = camera_points
        invalid_counts[handedness] = int(n_frames - valid.sum())
        in_frame_counts[handedness] = int(in_frame.sum())

        parts.append(pl.DataFrame({
            "sequence_id": np.full(n_frames, sequence_id),
            "subject_id": np.full(n_frames, subject_of(sequence_id)),
            "frame_idx": frames["frame_idx"].to_numpy(),
            "timestamp_ns": frame_ts,
            "handedness": np.full(n_frames, handedness),
            "track_id": np.full(n_frames, f"{sequence_id}:{handedness}"),
            "valid": valid,
            "joints_world": joints_world.reshape(n_frames, -1).astype(np.float32),
            "joints_camera": joints_camera.reshape(n_frames, -1).astype(np.float32),
            "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
            "camera_to_world": cam_to_world.reshape(n_frames, 16).astype(np.float32),
            "source_dataset": np.full(n_frames, SOURCE_DATASET),
            "source_frame_key": [f"{sequence_id}:{ts}" for ts in frame_ts.tolist()],
        }, schema=canonical_frame_schema()))

    df = pl.concat(parts) if parts else pl.DataFrame(schema=canonical_frame_schema())
    info = {
        "sequence_id": sequence_id,
        "subject_id": subject_of(sequence_id),
        "n_rgb_frames": n_frames,
        "n_rows": df.height,
        "hands": sorted(records),
        "timestamp_ns_range": [int(frame_ts[0]), int(frame_ts[-1])],
        "n_invalid_per_hand": invalid_counts, "n_in_frame_per_hand": in_frame_counts,
    }
    if verify:
        info["landmark_check"] = checks
    return df, info


def main() -> None:
    ap = argparse.ArgumentParser(description="HOT3D canonical frame cache 생성")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="split 당 sequence 수 제한(스모크용)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true",
                    help="UmeTrack landmark 대조 검증을 건너뛴다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = cfg["skeleton"]
    solver = ManoJointSolver(
        REPO_ROOT / skeleton["mano_model_path"],
        num_pca_comps=skeleton["num_pca_comps"],
        flat_hand_mean=skeleton["flat_hand_mean"],
        joint_map=skeleton["joint_map"],
        device=args.device,
    )
    tolerance = float(skeleton["landmark_agreement_tolerance_mm"])
    manifest = build_split_manifest(cfg)

    print(f"config     : {cfg.path} (hash {cfg.hash})")
    print(f"output dir : {out_dir}")
    summary = {"config_hash": cfg.hash, "run": run_metadata(cfg), "splits": {}}
    for split in args.splits:
        sequences = manifest["sequences"][split][: args.limit]
        frames_out, infos, worst = [], [], 0.0
        for i, sequence_id in enumerate(sequences, 1):
            df, info = build_sequence(cfg, solver, sequence_id, verify=not args.no_verify)
            frames_out.append(df)
            infos.append(info)
            for check in info.get("landmark_check", []):
                worst = max(worst, check["mean_mm"])
                if check["mean_mm"] > tolerance:
                    raise ValueError(
                        f"{sequence_id} {check['handedness']}: UmeTrack 대조 평균 "
                        f"{check['mean_mm']:.2f} mm > 허용 {tolerance} mm")
            print(f"  [{split}] {i}/{len(sequences)} {sequence_id} rows={df.height}", flush=True)

        table = pl.concat(frames_out)
        validate_canonical_frames(table)
        path = out_dir / f"canonical_frames_{split}.parquet"
        table.write_parquet(path)
        summary["splits"][split] = {
            "path": str(path),
            "n_sequences": len(sequences),
            "n_rows": table.height,
            "worst_landmark_mean_mm": worst,
            "sequences": infos,
        }
        print(f"  [{split}] rows={table.height} worst landmark mean={worst:.2f} mm -> {path}")

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest   : {manifest_path}")


if __name__ == "__main__":
    main()
