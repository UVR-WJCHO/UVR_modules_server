"""HOT3D raw MANO trajectory -> canonical 21-joint world pose.

왜 raw MANO 인가: `HOT3D_cache/sequences/*/hands.parquet` 의 21 landmark 는 UmeTrack
규약이라 canonical 21 (= WiLoR/DexYCB) 과 1:1 대응되지 않는다. thumb MCP 가 없고
PALM_CENTER 가 남는다. raw 의 `mano_hand_pose_trajectory.jsonl` 을 MANO 로 풀면
배포 경로(WiLoR)와 정확히 같은 skeleton 이 나온다.

실측으로 확정한 HOT3D MANO 규약 (2026-08-17, `verify_against_landmarks` 로 재현 가능):

  hand_poses key "0" = LEFT, "1" = RIGHT
  pose(15)  = MANO PCA coefficient, flat_hand_mean=False
  betas(10) = MANO shape
  wrist_xform = smplx 의 (global_orient, transl) 그 자체. 즉
      world = R @ (J - J0) + J0 + t
  좌수는 MANO_LEFT.pkl 이 없어 MANO_RIGHT 출력의 x 를 반전해 만든다(WiLoR 와 동일).

UmeTrack landmark 와의 잔차는 joint 당 2~14 mm 이고, 이는 두 skeleton 규약의 정의
차이다(UmeTrack WRIST_JOINT 는 tracked frame origin, MANO wrist 는 LBS root).
차선 규약이 39 mm 이므로 판별은 명확하다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..canonical_schema import NUM_JOINTS, load_joint_map, quat_wxyz_to_matrix

SOURCE_DATASET = "HOT3D"
HAND_KEY_TO_HANDEDNESS = {"0": "LEFT", "1": "RIGHT"}
_MIRROR = np.array([-1.0, 1.0, 1.0])

# UmeTrack landmark -> canonical index. thumb 의 non-tip 은 두 규약의 대응이 불확실해
# 검증에서 제외한다(매핑이 아니라 검증용 대조표다).
UMETRACK_CHECK_PAIRS: tuple[tuple[str, int], ...] = (
    ("WRIST_JOINT", 0), ("THUMB_FINGERTIP", 4),
    ("INDEX_PROXIMAL_FRAME", 5), ("INDEX_INTERMEDIATE_FRAME", 6),
    ("INDEX_DISTAL_FRAME", 7), ("INDEX_FINGER_FINGERTIP", 8),
    ("MIDDLE_PROXIMAL_FRAME", 9), ("MIDDLE_INTERMEDIATE_FRAME", 10),
    ("MIDDLE_DISTAL_FRAME", 11), ("MIDDLE_FINGER_FINGERTIP", 12),
    ("RING_PROXIMAL_FRAME", 13), ("RING_INTERMEDIATE_FRAME", 14),
    ("RING_DISTAL_FRAME", 15), ("RING_FINGER_FINGERTIP", 16),
    ("PINKY_PROXIMAL_FRAME", 17), ("PINKY_INTERMEDIATE_FRAME", 18),
    ("PINKY_DISTAL_FRAME", 19), ("PINKY_FINGER_FINGERTIP", 20),
)


@dataclass
class HandPoseRecords:
    """한 sequence, 한 hand 의 시간축 MANO parameter."""
    handedness: str
    timestamp_ns: np.ndarray      # (T,) int64
    pose: np.ndarray              # (T, 15) float32
    betas: np.ndarray             # (T, 10) float32
    rotation: np.ndarray          # (T, 3, 3) float64, world <- wrist
    translation: np.ndarray       # (T, 3) float64


class ManoJointSolver:
    """MANO parameter -> canonical 21 world joint. 좌우 모두 MANO_RIGHT 로 푼다."""

    def __init__(self, model_path: str | Path, num_pca_comps: int = 15,
                 flat_hand_mean: bool = False, joint_map: str = "mano_to_canonical",
                 device: str = "cpu"):
        import smplx

        spec = load_joint_map(joint_map)
        self.index_map = spec["index_map"]
        tips = spec["tip_vertex_ids"]
        self.tip_vertex_ids = np.array(
            [tips[k] for k in ("thumb", "index", "middle", "ring", "pinky")], dtype=np.int64)
        self.device = torch.device(device)
        self.model = smplx.MANO(
            model_path=str(model_path), is_rhand=True, use_pca=True,
            num_pca_comps=num_pca_comps, flat_hand_mean=flat_hand_mean,
            create_transl=False, batch_size=1,
        ).to(self.device).eval()

    @torch.no_grad()
    def solve(self, records: HandPoseRecords, batch_size: int = 4096) -> np.ndarray:
        """(T, 21, 3) world joints, meter."""
        n = len(records.timestamp_ns)
        out = np.empty((n, NUM_JOINTS, 3), dtype=np.float64)
        mirror = records.handedness == "LEFT"
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            joints = self._model_frame_joints(
                records.pose[start:stop], records.betas[start:stop], mirror)
            R = records.rotation[start:stop]
            t = records.translation[start:stop]
            root = joints[:, :1, :]
            # HOT3D wrist_xform == smplx (global_orient, transl): root 를 중심으로 회전
            out[start:stop] = np.einsum(
                "nij,nkj->nki", R, joints - root) + root + t[:, None, :]
        return out

    @torch.no_grad()
    def _model_frame_joints(self, pose: np.ndarray, betas: np.ndarray,
                            mirror: bool) -> np.ndarray:
        b = len(pose)
        result = self.model(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=self.device),
            hand_pose=torch.as_tensor(pose, dtype=torch.float32, device=self.device),
            global_orient=torch.zeros(b, 3, dtype=torch.float32, device=self.device),
        )
        joints = result.joints.cpu().numpy().astype(np.float64)          # (b, 16, 3)
        tips = result.vertices.cpu().numpy()[:, self.tip_vertex_ids].astype(np.float64)
        j21 = np.concatenate([joints, tips], axis=1)[:, self.index_map]  # (b, 21, 3)
        return j21 * _MIRROR if mirror else j21


def read_hand_pose_trajectory(sequence_dir: Path,
                              filename: str = "mano_hand_pose_trajectory.jsonl",
                              ) -> dict[str, HandPoseRecords]:
    """handedness -> HandPoseRecords. 손이 없는 timestamp 는 그 손에서 빠진다."""
    buckets: dict[str, dict[str, list]] = {
        hand: {"ts": [], "pose": [], "betas": [], "quat": [], "t": []}
        for hand in HAND_KEY_TO_HANDEDNESS.values()
    }
    with (sequence_dir / filename).open() as handle:
        for line in handle:
            record = json.loads(line)
            ts = record["timestamp_ns"]
            for key, payload in record["hand_poses"].items():
                hand = HAND_KEY_TO_HANDEDNESS.get(key)
                if hand is None:
                    raise ValueError(f"{sequence_dir.name}: 알 수 없는 hand key {key!r}")
                bucket = buckets[hand]
                bucket["ts"].append(ts)
                bucket["pose"].append(payload["pose"])
                bucket["betas"].append(payload["betas"])
                bucket["quat"].append(payload["wrist_xform"]["q_wxyz"])
                bucket["t"].append(payload["wrist_xform"]["t_xyz"])

    out = {}
    for hand, bucket in buckets.items():
        if not bucket["ts"]:
            continue
        ts = np.asarray(bucket["ts"], dtype=np.int64)
        order = np.argsort(ts, kind="stable")
        out[hand] = HandPoseRecords(
            handedness=hand,
            timestamp_ns=ts[order],
            pose=np.asarray(bucket["pose"], dtype=np.float32)[order],
            betas=np.asarray(bucket["betas"], dtype=np.float32)[order],
            rotation=quat_wxyz_to_matrix(np.asarray(bucket["quat"], dtype=np.float64))[order],
            translation=np.asarray(bucket["t"], dtype=np.float64)[order],
        )
    return out


def verify_against_landmarks(joints: np.ndarray, timestamps: np.ndarray, handedness: str,
                             cache_hands, ) -> dict:
    """canonical joint 를 cache 의 UmeTrack landmark 와 대조해 매핑을 검증한다.

    두 skeleton 은 규약이 달라 완전 일치하지 않는다. 잘못된 매핑/좌우 반전은 수십 mm
    이상으로 벌어지므로 판별에는 충분하다.
    """
    import polars as pl

    prefix = "L" if handedness == "LEFT" else "R"
    wanted = set(timestamps.tolist())
    sub = cache_hands.filter(pl.col("timestamp_ns").is_in(list(wanted)))
    index_of = {ts: i for i, ts in enumerate(timestamps.tolist())}

    per_joint, used = [], 0
    for label, canon_idx in UMETRACK_CHECK_PAIRS:
        rows = (sub.filter(pl.col("joint_label") == f"{prefix}_{label}")
                   .sort("timestamp_ns"))
        if not rows.height:
            continue
        ref = rows.select(["x", "y", "z"]).to_numpy()
        idx = np.array([index_of[ts] for ts in rows["timestamp_ns"].to_list()])
        per_joint.append((label, canon_idx,
                          float(np.linalg.norm(joints[idx, canon_idx] - ref, axis=-1).mean())))
        used = len(idx)

    values = np.array([v for _, _, v in per_joint])
    return {
        "handedness": handedness,
        "n_frames": used,
        "mean_mm": float(values.mean() * 1000),
        "max_joint_mm": float(values.max() * 1000),
        "per_joint_mm": {label: v * 1000 for label, _, v in per_joint},
    }
