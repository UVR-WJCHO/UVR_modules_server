"""모든 방법이 공유하는 정확도 metric (B 계획 §9.1).

입력은 (B, 21, 3) meter, 출력은 mm. metric 을 방법 쪽에서 따로 계산하지 않는다.
"""
from __future__ import annotations

import numpy as np

from ..data.canonical_schema import FINGERTIPS, WRIST

MM_PER_M = 1000.0


def per_joint_error_mm(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(B, 21, 3) x2 -> (B, 21) joint 별 유클리드 거리, mm."""
    return np.linalg.norm(prediction - target, axis=-1) * MM_PER_M


def root_relative(joints: np.ndarray) -> np.ndarray:
    return joints - joints[..., WRIST:WRIST + 1, :]


def sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """sample 당 스칼라 metric 묶음. 이후 sequence 단위로 집계한다 (§9.5)."""
    error = per_joint_error_mm(prediction, target)
    root_rel_error = per_joint_error_mm(root_relative(prediction), root_relative(target))
    return {
        "abs_mpjpe_mm": error.mean(axis=-1),
        "root_rel_mpjpe_mm": root_rel_error.mean(axis=-1),
        "wrist_err_mm": error[..., WRIST],
        "fingertip_mpjpe_mm": error[..., list(FINGERTIPS)].mean(axis=-1),
    }


def wrist_speed_mps(history_joints: np.ndarray, history_time_ms: np.ndarray) -> np.ndarray:
    """anchor 시점의 wrist 속도 (m/s). motion tercile 분류에 쓴다 (§9.4)."""
    dt_s = (history_time_ms[:, -1] - history_time_ms[:, -2]) / 1000.0
    delta = history_joints[:, -1, WRIST] - history_joints[:, -2, WRIST]
    return np.linalg.norm(delta, axis=-1) / np.maximum(dt_s, 1e-6)


def articulation_speed_mps(history_joints: np.ndarray, history_time_ms: np.ndarray) -> np.ndarray:
    """wrist 를 뺀 관절들의 평균 속도 (m/s)."""
    dt_s = (history_time_ms[:, -1] - history_time_ms[:, -2]) / 1000.0
    delta = root_relative(history_joints[:, -1]) - root_relative(history_joints[:, -2])
    return np.linalg.norm(delta, axis=-1).mean(axis=-1) / np.maximum(dt_s, 1e-6)


def tercile_labels(values: np.ndarray) -> np.ndarray:
    """값을 low/mid/high 3분위로 나눈다. 경계는 이 표본에서만 정한다."""
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.full(len(values), "mid", dtype=object)
    labels[values <= low] = "low"
    labels[values > high] = "high"
    return labels.astype(str)
