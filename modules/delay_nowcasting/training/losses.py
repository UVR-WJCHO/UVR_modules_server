"""학습 loss (B 계획 §7.6).

    L = 1.0 * L_abs + 1.0 * L_root_rel + 2.0 * L_wrist + 0.1 * L_bone + 0.1 * L_displacement

L_bone 은 tip bone 을 제외한다. MANO 의 fingertip 은 joint regressor 가 아니라 mesh
vertex 에서 나와 굽힘에 따라 길이가 변하므로(Phase 1 확인), 고정 길이로 강제하면
틀린 제약이 된다.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data.canonical_schema import BONES, FINGERTIPS, WRIST

# tip 을 뺀 rigid bone 만 (Phase 1: joint-to-joint bone 은 상대 표준편차 ~1e-7)
RIGID_BONES = tuple((a, b) for a, b in BONES if b not in FINGERTIPS)
MIN_HORIZON_MS = 16.0          # displacement rate 의 0 나눗셈 방지 (§7.6)

DEFAULT_WEIGHTS = {
    "abs": 1.0,
    "root_rel": 1.0,
    "wrist": 2.0,
    "bone": 0.1,
    "bone_2d": 0.0,       # 2.5D 표현 전용. 3D 학습에서는 0 이다
    "displacement": 0.1,
    # 인접한 두 anchor 에서 나온 예측의 시간적 일관성. 실측에서 GT 는 프레임간
    # 7.06 mm 움직이는데 예측은 13.72 mm 움직였다 — 정확하면서도 떤다.
    #   smooth : |Δ예측|            변화 자체를 누른다. abs 와 상충해 실제 거래가 생기지만
    #                               과하면 예측이 굳어 지연이 다시 생긴다.
    #   track  : |Δ예측 - ΔGT|      GT 가 움직인 만큼만 움직이게 한다. abs 와 겹쳐
    #                               효과가 작을 수 있다.
    "temporal_smooth": 0.0,
    "temporal_track": 0.0,
}


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, beta: float) -> torch.Tensor:
    """(B, J, 3) 에 대해 joint 단위 visibility mask 를 적용한 Smooth L1."""
    per_joint = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").sum(-1)
    weight = mask.to(per_joint.dtype)
    return (per_joint * weight).sum() / weight.sum().clamp_min(1.0)


def _bone_lengths(joints: torch.Tensor, dims: int | None = None) -> torch.Tensor:
    """관절 간 거리. `dims` 를 주면 앞 몇 채널만 쓴다.

    2.5D 표현에서는 세 채널의 단위가 달라 3D 거리가 물리적 길이가 아니다. 대신 앞 두
    채널(정규화 image 좌표)만 쓰면 화면상 뼈 길이가 되고, 이건 잘 정의된 값이다.
    손 모양이 프레임마다 늘었다 줄었다 하는 것을 막는 용도다.
    """
    a = joints[:, [x[0] for x in RIGID_BONES]]
    b = joints[:, [x[1] for x in RIGID_BONES]]
    if dims is not None:
        a, b = a[..., :dims], b[..., :dims]
    return torch.linalg.norm(a - b, dim=-1)


def nowcast_loss(prediction: torch.Tensor, target: torch.Tensor,
                 anchor: torch.Tensor, horizon_ms: torch.Tensor,
                 visibility: torch.Tensor | None = None,
                 weights: dict[str, float] | None = None,
                 beta: float = 0.01,
                 previous: tuple[torch.Tensor, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    """총 loss 와 항목별 값을 돌려준다. beta 는 meter 단위 기준 1 cm."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if visibility is None:
        visibility = torch.ones(prediction.shape[:2], device=prediction.device)

    pred_rel = prediction - prediction[:, WRIST:WRIST + 1]
    target_rel = target - target[:, WRIST:WRIST + 1]
    horizon_s = (horizon_ms.clamp_min(MIN_HORIZON_MS) / 1000.0).view(-1, 1, 1)

    terms = {
        "abs": _masked_smooth_l1(prediction, target, visibility, beta),
        "root_rel": _masked_smooth_l1(pred_rel, target_rel, visibility, beta),
        "wrist": F.smooth_l1_loss(prediction[:, WRIST], target[:, WRIST], beta=beta),
        "bone": (_bone_lengths(prediction) - _bone_lengths(target)).abs().mean(),
        "bone_2d": (_bone_lengths(prediction, 2) - _bone_lengths(target, 2)).abs().mean(),
        "displacement": F.smooth_l1_loss(
            (prediction - anchor) / horizon_s, (target - anchor) / horizon_s, beta=beta),
    }
    zero = prediction.new_zeros(())
    if previous is None:
        terms["temporal_smooth"] = terms["temporal_track"] = zero
    else:
        prev_pred, prev_target = previous
        d_pred = prediction - prev_pred
        d_target = target - prev_target
        # 짝이 없는 표본은 자기 자신을 가리켜 차이가 0 이다. 그런 표본은 제외한다.
        has_pair = (prev_target - target).abs().amax(dim=(1, 2)) > 0
        w = has_pair.to(prediction.dtype).view(-1, 1, 1)
        n = w.sum().clamp_min(1.0)
        terms["temporal_smooth"] = (d_pred.abs() * w).sum() / (n * prediction.shape[1] * 3)
        terms["temporal_track"] = ((d_pred - d_target).abs() * w).sum() / (
            n * prediction.shape[1] * 3)
    total = sum(weights[name] * value for name, value in terms.items())
    return total, {name: float(value.detach()) for name, value in terms.items()}
