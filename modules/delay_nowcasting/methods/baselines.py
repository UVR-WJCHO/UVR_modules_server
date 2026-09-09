"""학습 없는 baseline (B 계획 §6).

공통 계약: 모든 방법은 같은 window 를 받고 frame index 가 아니라 **실제 timestamp** 로
시간을 다룬다.

    predict(history_joints, history_time_ms, horizon_ms) -> (B, 21, 3)

    history_joints  : (B, N, 21, 3) world meter, 마지막이 anchor
    history_time_ms : (B, N) anchor 를 0 으로 한 상대 시각(과거가 음수)
    horizon_ms      : (B,) anchor 로부터의 예측 시간
"""
from __future__ import annotations

import numpy as np


def _fit_polynomial(joints: np.ndarray, time_ms: np.ndarray, degree: int,
                    n_fit: int) -> np.ndarray:
    """최근 n_fit frame 에 대한 최소제곱 다항 fit 계수 (B, degree+1, 21, 3).

    시간축은 초 단위로 두어 계수가 m, m/s, m/s^2 이 된다. 계수 순서는 낮은 차수부터라
    coef[0] 이 anchor 시각(t=0)의 위치다.
    """
    j = joints[:, -n_fit:]
    t = time_ms[:, -n_fit:] / 1000.0
    B, N = t.shape
    # 시간축을 history 길이로 정규화해 Vandermonde 의 조건수를 낮춘다. 정규방정식은
    # 조건수를 제곱하므로, 정규화 없이는 degree=2 에서 유효숫자가 눈에 띄게 깎인다.
    scale = np.maximum(np.abs(t).max(axis=1, keepdims=True), 1e-9)
    ts = t / scale
    design = np.stack([ts ** d for d in range(degree + 1)], axis=-1)    # (B, N, D)
    y = j.reshape(B, N, -1)                                             # (B, N, 63)
    # np.linalg.lstsq 는 batch 를 지원하지 않으므로 정규방정식을 batch 로 푼다.
    gram = np.einsum("bnd,bne->bde", design, design)
    rhs = np.einsum("bnd,bnk->bdk", design, y)
    coef = np.linalg.solve(gram, rhs)                                   # (B, D, 63)
    powers = scale[..., None] ** -np.arange(degree + 1)[None, :, None]  # (B, D, 1)
    return (coef * powers).reshape(B, degree + 1, *joints.shape[2:])


def _clip_magnitude(vec: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None:
        return vec
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norm, 1e-12))
    return vec * scale


def hold(history_joints: np.ndarray, history_time_ms: np.ndarray,
         horizon_ms: np.ndarray) -> np.ndarray:
    """J_hat(t+h) = J(t)."""
    return history_joints[:, -1].copy()


def constant_velocity(history_joints: np.ndarray, history_time_ms: np.ndarray,
                      horizon_ms: np.ndarray, n_fit: int = 2,
                      clip_speed_mps: float | None = None) -> np.ndarray:
    """n_fit=2 는 최근 2 frame 차분, n_fit>=3 은 최소제곱 선형 fit."""
    anchor = history_joints[:, -1]
    if n_fit == 2:
        dt_s = (history_time_ms[:, -1] - history_time_ms[:, -2]) / 1000.0
        velocity = (anchor - history_joints[:, -2]) / dt_s[:, None, None]
    else:
        coef = _fit_polynomial(history_joints, history_time_ms, degree=1, n_fit=n_fit)
        velocity = coef[:, 1]
    velocity = _clip_magnitude(velocity, clip_speed_mps)
    h = horizon_ms[:, None, None] / 1000.0
    return anchor + velocity * h


def constant_acceleration(history_joints: np.ndarray, history_time_ms: np.ndarray,
                          horizon_ms: np.ndarray, n_fit: int = 4,
                          clip_speed_mps: float | None = None,
                          clip_accel_mps2: float | None = None) -> np.ndarray:
    """최소 3 timestamp 로 velocity 와 acceleration 을 추정한다."""
    if n_fit < 3:
        raise ValueError("constant acceleration 은 최소 3 frame 이 필요하다")
    coef = _fit_polynomial(history_joints, history_time_ms, degree=2, n_fit=n_fit)
    velocity = _clip_magnitude(coef[:, 1], clip_speed_mps)
    accel = _clip_magnitude(2.0 * coef[:, 2], clip_accel_mps2)
    anchor = history_joints[:, -1]
    h = horizon_ms[:, None, None] / 1000.0
    return anchor + velocity * h + 0.5 * accel * h ** 2


def kalman_constant_velocity(history_joints: np.ndarray, history_time_ms: np.ndarray,
                             horizon_ms: np.ndarray, process_noise: float = 1000.0,
                             measurement_noise: float = 1e-2) -> np.ndarray:
    """관절·축별로 독립인 constant-velocity Kalman filter (B 계획 §6.4).

    process_noise 는 가속도 white noise 의 분산, measurement_noise 는 관측 분산이다.
    둘 다 validation 에서만 고른다. 결과를 정하는 것은 두 값의 **비**뿐이고, 2.5D
    표현에서 다시 훑어 보니 q/r 이 1e5(DexYCB) 와 5e4(HOT3D) 부근에서 평평했다.
    비가 이렇게 크다는 것은 필터가 관측을 거의 그대로 믿는다는 뜻이고, 실제로 8 프레임
    history 에서는 2 프레임 차분과 거의 같은 값을 낸다.
    """
    B, N = history_time_ms.shape
    shape = history_joints.shape[2:]                       # (21, 3)
    flat = history_joints.reshape(B, N, -1)                # (B, N, 63)
    k = flat.shape[-1]

    # 상태 [p, v], 공분산 2x2. 축마다 독립이라 (B, 63) 스칼라 다발로 굴린다.
    p = flat[:, 0]
    v = np.zeros_like(p)
    p_var = np.full_like(p, 1.0)
    pv_cov = np.zeros_like(p)
    v_var = np.full_like(p, 1.0)

    def update(dt, measurement, p, v, p_var, pv_cov, v_var):
        # predict
        p = p + v * dt
        p_var = p_var + 2 * dt * pv_cov + dt ** 2 * v_var + process_noise * dt ** 3 / 3
        pv_cov = pv_cov + dt * v_var + process_noise * dt ** 2 / 2
        v_var = v_var + process_noise * dt
        # correct (위치만 관측)
        innovation_var = p_var + measurement_noise
        gain_p = p_var / innovation_var
        gain_v = pv_cov / innovation_var
        residual = measurement - p
        p = p + gain_p * residual
        v = v + gain_v * residual
        p_var_new = (1 - gain_p) * p_var
        pv_cov_new = (1 - gain_p) * pv_cov
        v_var_new = v_var - gain_v * pv_cov
        return p, v, p_var_new, pv_cov_new, v_var_new

    dts = np.diff(history_time_ms, axis=1) / 1000.0
    for step in range(1, N):
        dt = np.maximum(dts[:, step - 1], 1e-6)[:, None]
        p, v, p_var, pv_cov, v_var = update(dt, flat[:, step], p, v, p_var, pv_cov, v_var)

    h = (horizon_ms / 1000.0)[:, None]
    return (p + v * h).reshape(B, *shape)


# evaluate.py 가 도는 baseline 목록. 이름은 결과 표의 method column 이 된다.
BASELINES: dict[str, callable] = {
    "hold": hold,
    "cv_2frame": lambda j, t, h: constant_velocity(j, t, h, n_fit=2),
    "cv_robust": lambda j, t, h: constant_velocity(j, t, h, n_fit=4),
    "cv_robust_clip": lambda j, t, h: constant_velocity(j, t, h, n_fit=4, clip_speed_mps=2.0),
    "const_accel": lambda j, t, h: constant_acceleration(j, t, h, n_fit=4),
    "kalman_cv": kalman_constant_velocity,
}
