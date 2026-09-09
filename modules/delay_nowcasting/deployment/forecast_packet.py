"""HandForecast 직렬화 계약 (B 계획 §5, §11 Phase 6).

horizon grid 를 평탄화해 보내므로 flatten/unflatten round-trip 이 계약의 핵심이다.
`horizons_ms[i]` 의 pose 는 `joints_25d[(i*63):((i+1)*63)]` 이다.

자세는 2.5D 다 — 관절마다 (u_n, v_n, rel_z). 정규화 image 좌표(x/z, y/z, 무차원)와
손목 기준 상대 depth(m) 이고 손목의 rel_z 는 0 이다. 손목 절대 depth 는 보내지 않는다.
기기가 자기 최신 depth 로 되올린다 — z_j = d_wrist + rel_z_j, x_j = u_n_j * z_j.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..data.canonical_schema import NUM_JOINTS

PROTOCOL_VERSION = 2      # 1 은 world 절대 3D 였다
_COMM = Path(__file__).resolve().parents[3] / "_comm"


def _proto():
    if str(_COMM) not in sys.path:
        sys.path.insert(0, str(_COMM))
    import hl2_forecast_pb2 as proto      # noqa: E402

    return proto


def flatten(grid: np.ndarray) -> list[float]:
    """(H, 21, 3) -> H*63 평탄화."""
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(f"forecast grid shape 이 (H, {NUM_JOINTS}, 3) 이 아니다: {array.shape}")
    return array.reshape(-1).tolist()


def unflatten(values, n_horizons: int) -> np.ndarray:
    """H*63 -> (H, 21, 3)."""
    array = np.asarray(values, dtype=np.float32)
    expected = n_horizons * NUM_JOINTS * 3
    if array.size != expected:
        raise ValueError(f"joints_world 길이가 {expected} 가 아니라 {array.size} 다")
    return array.reshape(n_horizons, NUM_JOINTS, 3)


def build(*, source_frame_id: int, source_capture_timestamp_ns: int,
          legacy_timestamp: float, handedness: str, track_id: str,
          horizons_ms, grid: np.ndarray | None, anchor_25d: np.ndarray | None,
          model_version: str, history_length: int, valid: bool,
          invalid_reason: str = "NONE", timings: dict | None = None) -> bytes:
    proto = _proto()
    message = proto.HandForecast()
    message.protocol_version = PROTOCOL_VERSION
    message.source_frame_id = int(source_frame_id)
    message.source_capture_timestamp_ns = int(source_capture_timestamp_ns)
    message.legacy_timestamp = float(legacy_timestamp)
    for name, value in (timings or {}).items():
        setattr(message, name, int(value))

    message.valid = bool(valid)
    message.track_id = track_id
    message.handedness = proto.RIGHT if handedness == "RIGHT" else proto.LEFT
    message.coordinate_space = proto.IMAGE_25D
    message.model_version = model_version
    message.history_length = int(history_length)
    message.invalid_reason = getattr(proto, invalid_reason, proto.NONE)

    message.horizons_ms.extend([float(h) for h in horizons_ms])
    if grid is not None:
        message.joints_25d.extend(flatten(grid))
    if anchor_25d is not None:
        message.source_anchor_25d.extend(
            np.asarray(anchor_25d, np.float32).reshape(-1).tolist())
    return message.SerializeToString()


def parse(payload: bytes) -> dict:
    proto = _proto()
    message = proto.HandForecast()
    message.ParseFromString(payload)
    horizons = list(message.horizons_ms)
    return {
        "protocol_version": message.protocol_version,
        "source_frame_id": message.source_frame_id,
        "source_capture_timestamp_ns": message.source_capture_timestamp_ns,
        "valid": message.valid,
        "track_id": message.track_id,
        "handedness": proto.Handedness.Name(message.handedness),
        "invalid_reason": proto.InvalidReason.Name(message.invalid_reason),
        "model_version": message.model_version,
        "history_length": message.history_length,
        "horizons_ms": horizons,
        "joints_25d": (unflatten(message.joints_25d, len(horizons))
                       if message.joints_25d else None),
        "source_anchor_25d": (np.asarray(message.source_anchor_25d, np.float32)
                              .reshape(NUM_JOINTS, 3)
                              if message.source_anchor_25d else None),
    }


def interpolate(horizons_ms, grid: np.ndarray, pose_age_ms: float) -> np.ndarray:
    """HMD 가 하는 일. 인접한 두 forecast 를 선형 보간한다 (B 계획 §0.1).

    가장 가까운 것을 고르는 게 아니다. grid 범위 밖은 clamp 한다 — 무한 외삽하지 않는다.
    """
    horizons = np.asarray(horizons_ms, dtype=np.float64)
    order = np.argsort(horizons)
    horizons, grid = horizons[order], np.asarray(grid)[order]
    age = float(np.clip(pose_age_ms, horizons[0], horizons[-1]))
    upper = int(np.searchsorted(horizons, age, side="left"))
    if upper == 0:
        return grid[0].copy()
    lower = upper - 1
    span = horizons[upper] - horizons[lower]
    if span <= 0:
        return grid[upper].copy()
    weight = (age - horizons[lower]) / span
    return (1.0 - weight) * grid[lower] + weight * grid[upper]
