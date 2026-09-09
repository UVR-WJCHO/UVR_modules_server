"""학습 모델이 쓰는 입력 feature 와 constant-velocity base (torch).

B 계획 §7.1 의 M0 구성:

    base    = constant_velocity(history, horizon)
    feature = [normalized history, velocity, acceleration, dt, horizon embedding]
    출력    = base + residual

baselines.py 의 numpy 구현과 수치가 일치해야 한다(tests/test_features.py 가 확인).
"""
from __future__ import annotations

import torch

# 실기기 세션 두 건에서 렌더 pose age p50 이 202.8~208.2 ms, p90 이 245.9~248.8 ms,
# p99 가 280 ms 였다. 199 ms grid 로는 프레임의 55~63% 가 상한에 클램프된다. 그래서
# grid 를 300 ms 까지 넓혔다 (2026-08-27).
# 이 값이 encoding 의 정규화 기준이므로, grid 상한보다 작으면 그 위의 horizon 들이
# 같은 encoding 으로 뭉개진다. 바뀌면 예전 checkpoint 는 호환되지 않는다.
HORIZON_CLAMP_MS = 300.0
HORIZON_ENCODING_DIM = 6


def fit_polynomial(joints: torch.Tensor, time_ms: torch.Tensor, degree: int,
                   n_fit: int) -> torch.Tensor:
    """최근 n_fit frame 의 최소제곱 다항 fit 계수 (B, degree+1, 21, 3), 시간 단위는 초.

    baselines._fit_polynomial 과 같은 방식이다. 시간축을 정규화해 정규방정식의
    조건수를 낮춘다.
    """
    j = joints[:, -n_fit:]
    t = time_ms[:, -n_fit:] / 1000.0
    B, N = t.shape
    scale = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    ts = t / scale
    design = torch.stack([ts ** d for d in range(degree + 1)], dim=-1)      # (B, N, D)
    y = j.reshape(B, N, -1)                                                # (B, N, 63)
    gram = torch.einsum("bnd,bne->bde", design, design)
    rhs = torch.einsum("bnd,bnk->bdk", design, y)
    if degree == 1:
        # 2x2 는 닫힌 형식으로 푼다. torch.linalg.solve 는 ONNX opset 17 로 안 나가고,
        # CV base 는 항상 degree 1 이라 이 경로만 배포에 실린다.
        a, b = gram[:, 0, 0], gram[:, 0, 1]
        c, d = gram[:, 1, 0], gram[:, 1, 1]
        det = (a * d - b * c).unsqueeze(-1)
        coef = torch.stack([(d.unsqueeze(-1) * rhs[:, 0] - b.unsqueeze(-1) * rhs[:, 1]) / det,
                            (a.unsqueeze(-1) * rhs[:, 1] - c.unsqueeze(-1) * rhs[:, 0]) / det],
                           dim=1)                                          # (B, 2, 63)
    else:
        coef = torch.linalg.solve(gram, rhs)                               # (B, D, 63)
    powers = scale.unsqueeze(-1) ** -torch.arange(
        degree + 1, device=t.device, dtype=t.dtype).view(1, -1, 1)
    return (coef * powers).reshape(B, degree + 1, *joints.shape[2:])


# 추정기가 튄 프레임에서는 2-frame 차분 속도가 물리적으로 불가능한 값이 된다(실측 최대
# 117 m/s). GT 기준 실제 손 속도는 p99.9 가 1.7 m/s, 최대 3.6 m/s 이므로 2.0 m/s 로 자르면
# 실제 움직임은 통과시키면서 잡음만 걷어낸다. baseline cv_robust_clip 과 같은 값이다.
CLIP_SPEED_MPS = 2.0

# 2.5D 표현에서 쓰는 상한. 채널 단위가 다르므로 (광선 /s, rel_z m/s) 두 값이다.
# 거리를 곱해 m/s 로 환산하지 않는다 — 그 거리가 이 표현에서 버린 값이다. 대신 GT
# 를 같은 표현으로 옮겨 직접 쟀다: 광선 p99 2.767, p99.9 5.160, 최대 27.13 /s,
# rel_z p99 0.376, p99.9 0.744, 최대 3.84 m/s (프레임쌍 787,415).
# 3D 에서 p99.9 1.7 을 2.0 으로 잘랐던 것과 같은 기준으로 p99.9 바로 위에 둔다.
#
# 필요한 이유: base_n_fit 이 작으면 마지막 프레임쌍의 순간 튐이 그대로 horizon 에
# 곱해진다. HL2/웹캠 기록에서 관절 화면속도가 p50 117 px/s 인데 p99 에서 마지막
# 차분이 중앙값의 7.06 배로 뛰었고, base 가 anchor 에서 709 px 까지 날아갔다.
CLIP_SPEED_25D = (6.0, 0.9)



def constant_velocity_base(history: torch.Tensor, history_time_ms: torch.Tensor,
                           horizon_ms: torch.Tensor, n_fit: int = 2,
                           clip_speed_mps: float | None = CLIP_SPEED_MPS) -> torch.Tensor:
    """residual 이 올라탈 base. Phase 1 에서 GT history 최강이었던 2-frame 차분이 기본."""
    anchor = history[:, -1]
    if n_fit == 2:
        dt_s = (history_time_ms[:, -1] - history_time_ms[:, -2]) / 1000.0
        velocity = (anchor - history[:, -2]) / dt_s.view(-1, 1, 1)
    else:
        velocity = fit_polynomial(history, history_time_ms, degree=1, n_fit=n_fit)[:, 1]
    if isinstance(clip_speed_mps, (tuple, list)):
        # 2.5D: 앞 두 채널(광선)은 크기로, rel_z 는 따로 자른다
        ray_limit, depth_limit = clip_speed_mps
        ray, depth = velocity[..., :2], velocity[..., 2:]
        norm = torch.linalg.norm(ray, dim=-1, keepdim=True)
        ray = ray * torch.clamp(ray_limit / norm.clamp_min(1e-12), max=1.0)
        velocity = torch.cat([ray, depth.clamp(-depth_limit, depth_limit)], dim=-1)
    elif clip_speed_mps is not None:
        norm = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        velocity = velocity * torch.clamp(clip_speed_mps / norm.clamp_min(1e-12), max=1.0)
    return anchor + velocity * (horizon_ms / 1000.0).view(-1, 1, 1)


def horizon_encoding(horizon_ms: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, 6). B 계획 §7.4."""
    h = (horizon_ms.clamp(0.0, HORIZON_CLAMP_MS) / HORIZON_CLAMP_MS).unsqueeze(-1)
    pi = torch.pi
    return torch.cat([h, h ** 2, torch.sin(pi * h), torch.cos(pi * h),
                      torch.sin(2 * pi * h), torch.cos(2 * pi * h)], dim=-1)


def build_features(history: torch.Tensor, history_time_ms: torch.Tensor,
                   horizon_ms: torch.Tensor, accel_n_fit: int = 4) -> torch.Tensor:
    """(B, N, 21, 3) history -> (B, F) feature.

    좌표는 anchor frame 의 wrist 를 원점으로 옮겨 절대 위치 의존을 없앤다. 시간은
    실제 timestamp 차이를 그대로 쓴다(frame index 를 시간으로 가정하지 않는다).
    """
    B, N = history_time_ms.shape
    wrist = history[:, -1, 0]                                   # (B, 3)
    relative = history - wrist.view(B, 1, 1, 3)                 # (B, N, 21, 3)

    dt_s = (history_time_ms[:, -1] - history_time_ms[:, -2]) / 1000.0
    velocity = (history[:, -1] - history[:, -2]) / dt_s.view(-1, 1, 1)
    accel = 2.0 * fit_polynomial(history, history_time_ms, degree=2, n_fit=accel_n_fit)[:, 2]

    return torch.cat([
        relative.reshape(B, -1),                                # N * 63
        velocity.reshape(B, -1),                                # 63
        accel.reshape(B, -1),                                   # 63
        history_time_ms / HORIZON_CLAMP_MS,                     # N
        horizon_encoding(horizon_ms),                           # 6
    ], dim=-1)


def feature_dim(history_length: int, num_joints: int = 21) -> int:
    return (history_length + 2) * num_joints * 3 + history_length + HORIZON_ENCODING_DIM
