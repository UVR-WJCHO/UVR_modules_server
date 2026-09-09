"""실제 WiLoR history 의 오차 특성을 측정해 augmentation parameter 로 바꾼다.

B 계획 §4.5: "Noise scale 은 임의로 논문 test 에 맞추지 않는다. 실제 WiLoR cache 에서
joint 별 residual 분포를 측정한 후 train configuration 을 갱신한다."

측정은 반드시 **train subject** 에서 한다. val 에서 뽑은 통계로 augmentation 을 맞추면
model selection 이 오염된다.

오차를 세 성분으로 나눈다.
  bias   : (sequence, hand) 별 평균 offset. WiLoR 가 추정한 MANO shape 이 피험자와
           다른 데서 오는 정적 성분이라, 움직임 예측에는 거의 영향이 없다.
  drift  : bias 를 뺀 뒤 남는 느리게 변하는 성분 (시간 상관이 큼).
  jitter : 이웃 프레임 3차 보간으로부터의 이탈. 프레임별 흰 잡음에 가깝고,
           속도/가속도 추정을 직접 망가뜨리는 성분이다.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .canonical_schema import NUM_JOINTS

NEIGHBOUR_OFFSETS = (-2, -1, 1, 2)


def _smooth_reference(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 21, 3) 궤적에서 이웃 4 프레임 3차 보간으로 각 프레임을 예측한다.

    반환 (예측값, 사용 가능한 프레임 mask). 자기 자신은 fit 에 넣지 않으므로,
    예측과의 차이가 곧 그 프레임의 독립적인 흔들림이다.
    """
    T = len(track)
    design = np.vander(np.asarray(NEIGHBOUR_OFFSETS, dtype=np.float64), 4)
    pinv = np.linalg.pinv(design)                       # (4, 4)
    usable = np.zeros(T, dtype=bool)
    usable[2:T - 2] = True
    predicted = np.full_like(track, np.nan)
    if T > 4:
        idx = np.arange(2, T - 2)
        neighbours = np.stack([track[idx + o] for o in NEIGHBOUR_OFFSETS], axis=1)
        coef = np.einsum("dn,tnjk->tdjk", pinv, neighbours)
        predicted[idx] = coef[:, -1]                    # offset 0 에서의 값
    return predicted, usable


def measure(wilor: pl.DataFrame, gt: pl.DataFrame) -> dict:
    """두 표(행 순서 동일)에서 joint 별 bias / drift / jitter 통계를 낸다."""
    if wilor.height != gt.height:
        raise ValueError("WiLoR 표와 GT 표의 행 수가 다르다")

    wilor_joints = wilor["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    gt_joints = gt["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    both_valid = wilor["valid"].to_numpy() & gt["valid"].to_numpy()
    keys = list(zip(wilor["sequence_id"].to_list(), wilor["handedness"].to_list()))

    bias_norms, drift, jitter, gaps, run_lengths = [], [], [], [], []
    start = 0
    for i in range(1, len(keys) + 1):
        if i < len(keys) and keys[i] == keys[start]:
            continue
        segment = slice(start, i)
        valid = both_valid[segment]
        start = i
        if valid.sum() < 32:
            continue

        error = wilor_joints[segment] - gt_joints[segment]
        bias = np.nanmean(np.where(valid[:, None, None], error, np.nan), axis=0)
        bias_norms.append(np.linalg.norm(bias, axis=-1))              # (21,)
        centred = error - bias

        # jitter: WiLoR 궤적 자체의 자기일관성 이탈. 유효 구간이 연속인 곳만 본다.
        predicted, usable = _smooth_reference(np.where(valid[:, None, None],
                                                       wilor_joints[segment], np.nan))
        ok = usable & valid & np.isfinite(predicted).all(axis=(1, 2))
        if ok.any():
            jitter.append(np.linalg.norm(wilor_joints[segment][ok] - predicted[ok], axis=-1))
        drift.append(np.linalg.norm(centred[valid], axis=-1))

        # 검출 누락 통계
        gaps.append(1.0 - valid.mean())
        missing, run = valid == False, 0                              # noqa: E712
        for m in missing:
            run = run + 1 if m else 0
            if run:
                run_lengths.append(run)

    per_joint_jitter = np.concatenate(jitter) if jitter else np.zeros((1, NUM_JOINTS))
    per_joint_drift = np.concatenate(drift) if drift else np.zeros((1, NUM_JOINTS))
    bias_norms = np.stack(bias_norms) if bias_norms else np.zeros((1, NUM_JOINTS))
    runs = np.asarray(run_lengths) if run_lengths else np.zeros(1)

    return {
        "n_tracks": len(bias_norms),
        "per_joint_jitter_mm": (np.median(per_joint_jitter, axis=0) * 1000).tolist(),
        "per_joint_drift_mm": (np.median(per_joint_drift, axis=0) * 1000).tolist(),
        "per_joint_bias_mm": (np.median(bias_norms, axis=0) * 1000).tolist(),
        "jitter_mm_median": float(np.median(per_joint_jitter) * 1000),
        "jitter_mm_p95": float(np.percentile(per_joint_jitter, 95) * 1000),
        "drift_mm_median": float(np.median(per_joint_drift) * 1000),
        "bias_mm_median": float(np.median(bias_norms) * 1000),
        "drop_rate": float(np.mean(gaps)),
        "burst_drop_median": float(np.median(runs)),
        "burst_drop_p95": float(np.percentile(runs, 95)),
    }
