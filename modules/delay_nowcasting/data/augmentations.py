"""WiLoR 같은 history 열화를 GT 위에 재현한다 (B 계획 §4.5).

조건 B(corrupted GT) 학습용이다. scale 은 임의로 정하지 않고 `noise_model.measure` 가
**train subject** 에서 뽑은 값으로 채운다.

각 항목은 독립적으로 on/off 할 수 있어야 한다(§4.5, §10.2 ablation 9). 그래서 하나의
dataclass 로 묶고 weight 0 이면 그 항목이 사라지도록 했다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .canonical_schema import FINGERTIPS, NUM_JOINTS, WRIST


@dataclass
class NoiseConfig:
    """모든 크기는 meter, 확률은 0~1."""
    jitter_m: float = 0.0                 # 프레임별 흰 잡음 (joint 별 등방)
    fingertip_jitter_scale: float = 1.0   # fingertip 은 보통 더 크다 (§4.5)
    drift_m: float = 0.0                  # 느리게 변하는 성분의 정상상태 크기
    drift_halflife_frames: float = 8.0    # drift 의 시간 상관
    wrist_translation_m: float = 0.0      # global wrist 만 흔드는 성분
    bias_m: float = 0.0                   # window 당 고정 offset (shape 불일치)
    clean_ratio: float = 0.0              # 이 비율만큼은 잡음을 넣지 않는다 (깨끗+잡음 혼합 학습)
    # 손목 depth 가 한 프레임 튀는 것. HL2 기록에서 잰 값이다 — 프레임간 손목 depth
    # 변화가 중앙값 1.0 mm 인데 1.7% 의 프레임이 40 mm 를, 0.8% 가 100 mm 를 넘고
    # 최대 355 mm 였다. 그 한 프레임이 history 에 8 프레임 남아 예측을 발산시킨다.
    depth_spike_rate: float = 0.0         # 프레임당 튐 발생 확률
    depth_spike_min_m: float = 0.03       # 튐 크기 (log-uniform 으로 뽑는다)
    depth_spike_max_m: float = 0.35
    depth_spike_distance_m: float = 0.35  # 손까지의 공칭 거리. 가로세로 배율을 정한다
    drop_rate: float = 0.0                # 프레임 단위 무작위 누락
    burst_drop_rate: float = 0.0          # burst 시작 확률
    burst_drop_max: int = 4
    enabled: tuple[str, ...] = field(default_factory=lambda: (
        "jitter", "drift", "wrist", "bias", "drop", "burst", "depth_spike"))

    @classmethod
    def from_measurement(cls, stats: dict, **overrides) -> "NoiseConfig":
        """`noise_model.measure` 결과를 그대로 parameter 로 옮긴다."""
        per_joint = torch.tensor(stats["per_joint_jitter_mm"])
        tip = per_joint[list(FINGERTIPS)].median()
        other = per_joint[[j for j in range(NUM_JOINTS) if j not in FINGERTIPS]].median()
        base = dict(
            jitter_m=stats["jitter_mm_median"] / 1000.0,
            fingertip_jitter_scale=float(tip / other.clamp_min(1e-6)),
            drift_m=stats["drift_mm_median"] / 1000.0,
            bias_m=stats["bias_mm_median"] / 1000.0,
            drop_rate=stats["drop_rate"],
            burst_drop_rate=stats["drop_rate"] / max(stats["burst_drop_median"], 1.0),
            burst_drop_max=int(max(stats["burst_drop_p95"], 2)),
        )
        base.update(overrides)
        return cls(**base)


def _joint_scale(config: NoiseConfig, device, dtype) -> torch.Tensor:
    scale = torch.ones(NUM_JOINTS, device=device, dtype=dtype)
    scale[list(FINGERTIPS)] = config.fingertip_jitter_scale
    return scale.view(1, 1, NUM_JOINTS, 1)


def corrupt_history(history: torch.Tensor, visibility: torch.Tensor,
                    config: NoiseConfig, generator: torch.Generator | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N, 21, 3) GT history -> WiLoR 같은 history 와 갱신된 visibility.

    누락된 프레임은 **직전 유효 프레임으로 유지(hold)** 한다. 배포에서 anchor 가 오지
    않으면 마지막 anchor 가 그대로 남기 때문이다. 0 으로 채우면 정지한 손이 되어버린다.
    """
    B, N = history.shape[:2]
    device, dtype = history.device, history.dtype
    out = history.clone()
    on = set(config.enabled)

    # 깨끗한 입력과 잡음 입력을 한 batch 안에 섞는다. 잡음만으로 학습하면 정밀도를
    # 잃고, 깨끗한 입력만으로 학습하면 실제 입력에서 무너진다(조건 A/C 표 참조).
    keep_clean = (torch.rand(B, device=device, generator=generator) < config.clean_ratio
                  if config.clean_ratio > 0 else
                  torch.zeros(B, device=device, dtype=torch.bool))

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    def rand(*shape):
        return torch.rand(*shape, device=device, dtype=dtype, generator=generator)

    if "jitter" in on and config.jitter_m > 0:
        out = out + randn(B, N, NUM_JOINTS, 3) * config.jitter_m * _joint_scale(
            config, device, dtype)

    if "drift" in on and config.drift_m > 0:
        # AR(1) 로 시간 상관을 준다. halflife 에서 상관이 절반이 되도록 계수를 잡는다.
        alpha = 0.5 ** (1.0 / max(config.drift_halflife_frames, 1e-3))
        step = randn(B, N, NUM_JOINTS, 3) * config.drift_m * (1 - alpha ** 2) ** 0.5
        drift = torch.empty_like(step)
        drift[:, 0] = randn(B, NUM_JOINTS, 3) * config.drift_m
        for t in range(1, N):
            drift[:, t] = alpha * drift[:, t - 1] + step[:, t]
        out = out + drift

    if "wrist" in on and config.wrist_translation_m > 0:
        out = out + (randn(B, N, 1, 3) * config.wrist_translation_m)

    if "bias" in on and config.bias_m > 0:
        out = out + randn(B, 1, NUM_JOINTS, 3) * config.bias_m

    if "depth_spike" in on and config.depth_spike_rate > 0:
        # 손목 depth 가 delta 만큼 틀리면 손 전체가 시선 방향으로 delta 만큼 밀리고,
        # 가로세로는 (거리+delta)/거리 배가 된다 (x = (u-cx)/f * z 이므로). 학습 좌표계에는
        # 카메라가 없으니 시선 방향은 window 마다 무작위로 잡는다 — 배포에서 머리 방향이
        # world 기준으로 임의인 것과 같다.
        hit = rand(B, N) < config.depth_spike_rate
        ray = randn(B, 1, 1, 3)
        ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        low, high = config.depth_spike_min_m, config.depth_spike_max_m
        size = low * (high / low) ** rand(B, N)               # log-uniform
        sign = torch.where(rand(B, N) < 0.5, -1.0, 1.0)
        delta = (size * sign * hit).view(B, N, 1, 1)
        centre = out.mean(dim=2, keepdim=True)
        radial = out - centre
        perpendicular = radial - (radial * ray).sum(-1, keepdim=True) * ray
        out = out + delta * ray + perpendicular * (delta / config.depth_spike_distance_m)

    missing = torch.zeros(B, N, device=device, dtype=torch.bool)
    if "drop" in on and config.drop_rate > 0:
        missing |= rand(B, N) < config.drop_rate
    if "burst" in on and config.burst_drop_rate > 0:
        starts = rand(B, N) < config.burst_drop_rate
        length = torch.randint(1, config.burst_drop_max + 1, (B, N), device=device,
                               generator=generator)
        for t in range(N):
            for offset in range(config.burst_drop_max):
                if t + offset < N:
                    missing[:, t + offset] |= starts[:, t] & (length[:, t] > offset)
    # anchor 는 항상 있어야 forecast 를 만들 수 있다
    missing[:, -1] = False

    if missing.any():
        for t in range(1, N):
            hold = missing[:, t]
            out[hold, t] = out[hold, t - 1]
        # 첫 프레임이 없으면 뒤에서 채운다(그 앞이 없으므로)
        first = missing[:, 0]
        if first.any():
            out[first, 0] = out[first, 1]

    new_visibility = visibility.clone()
    new_visibility[missing] = 0
    if keep_clean.any():
        out[keep_clean] = history[keep_clean]
        new_visibility[keep_clean] = visibility[keep_clean]
    return out, new_visibility


# ---------------------------------------------------------------------------
# 기하 증강 (2.5D 전용). window 단위로 하나의 변환을 뽑아 history 와 target 에
# **똑같이** 적용한다. 프레임마다 다른 변환을 주면 속도·가속도가 망가진다.
# ---------------------------------------------------------------------------
@dataclass
class GeomConfig:
    """모두 window(클립) 단위. 각도는 도, scale 은 배율."""
    rotate_deg: float = 0.0        # ±이 값 사이 균등. 광축 둘레 회전
    scale_log2: float = 0.0        # 배율을 2^U(-a, a) 로 뽑는다. 0.5 면 0.71~1.41 배
    mirror_prob: float = 0.0       # 좌우 반전 확률


def sample_geom(batch: int, config: GeomConfig, device, dtype,
                generator: torch.Tensor | None = None) -> dict:
    """window 단위 변환 파라미터를 뽑는다. 짝 window 를 함께 추론하는 시간적 일관성
    loss 에서는 **두 window 가 같은 파라미터를 써야** 비교가 성립하므로 분리해 둔다."""
    rand = lambda: torch.rand(batch, device=device, dtype=dtype, generator=generator)
    return {
        "scale": torch.pow(2.0, (rand() * 2 - 1) * config.scale_log2)
                 if config.scale_log2 > 0 else None,
        "flip": (rand() < config.mirror_prob) if config.mirror_prob > 0 else None,
        "angle": ((rand() * 2 - 1) * (config.rotate_deg * math.pi / 180.0))
                 if config.rotate_deg > 0 else None,
    }


def augment_window(history: torch.Tensor, target: torch.Tensor,
                   handedness: torch.Tensor, geom, generator=None):
    """(B,N,21,3), (B,21,3), (B,) -> 같은 모양. 2.5D (u_n, v_n, rel_z) 를 가정한다.

    geom 은 GeomConfig 이거나 sample_geom 이 만든 파라미터 dict 다.

    회전 : 광축 둘레 카메라 roll 과 **정확히 같다**. (u_n, v_n) 을 원점 둘레로 돌리고
           rel_z 는 건드리지 않는다 — (x,y,z) -> (Rx, Ry, z) 이므로 u_n=x/z 가 그대로 돈다.
    반전 : u_n 부호를 뒤집고 handedness 를 바꾼다. 역시 정확하다.
    확대 : 손목을 중심으로 (u_n, v_n) 과 rel_z 를 함께 s 배 한다. 같은 거리에서 손이
           s 배 큰 것과 정확히 같고, 거리 변화로 읽으면 rel_z/z_wrist 만큼(약 10%)
           어긋난다. 등방이라 뼈 구조가 자기 모순을 일으키지 않는다.
    """
    params = (geom if isinstance(geom, dict)
              else sample_geom(history.shape[0], geom, history.device, history.dtype,
                               generator))
    history, target, handedness = history.clone(), target.clone(), handedness.clone()

    if params["scale"] is not None:
        s = params["scale"]
        wrist = history[:, :, WRIST:WRIST + 1, :2]
        history[..., :2] = wrist + (history[..., :2] - wrist) * s[:, None, None, None]
        history[..., 2] = history[..., 2] * s[:, None, None]
        wrist = target[:, WRIST:WRIST + 1, :2]
        target[..., :2] = wrist + (target[..., :2] - wrist) * s[:, None, None]
        target[..., 2] = target[..., 2] * s[:, None]

    if params["flip"] is not None:
        sign = torch.where(params["flip"], -1.0, 1.0).to(history.dtype)
        history[..., 0] = history[..., 0] * sign[:, None, None]
        target[..., 0] = target[..., 0] * sign[:, None]
        handedness = torch.where(params["flip"], 1 - handedness, handedness)

    if params["angle"] is not None:
        cos, sin = torch.cos(params["angle"]), torch.sin(params["angle"])
        def rotate(points, shape):
            u, v = points[..., 0], points[..., 1]
            c, s_ = cos.reshape(shape), sin.reshape(shape)
            return torch.stack([c * u - s_ * v, s_ * u + c * v], dim=-1)
        history[..., :2] = rotate(history, (-1, 1, 1))
        target[..., :2] = rotate(target, (-1, 1))

    return history, target, handedness
