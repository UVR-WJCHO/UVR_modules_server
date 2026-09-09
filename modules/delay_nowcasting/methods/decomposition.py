"""Global(wrist) / local(root-relative) 분해와 복원 (B 계획 §3.3, §7.3).

    w_t     = J_t[0]              # world-space wrist
    q_t[j]  = J_t[j] - w_t        # wrist-relative articulation, world 축 유지

palm-aligned local 좌표와 wrist orientation 6D 표현은 direct 분해가 유효한 뒤
2차 ablation 으로 추가한다(§3.3). 지금은 그 전 단계다.
"""
from __future__ import annotations

import torch

from ..data.canonical_schema import WRIST


def decompose(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 21, 3) -> (wrist (..., 3), root_relative (..., 21, 3))."""
    wrist = joints[..., WRIST, :]
    return wrist, joints - wrist.unsqueeze(-2)


def compose(wrist: torch.Tensor, root_relative: torch.Tensor) -> torch.Tensor:
    """decompose 의 역. wrist 는 (..., 3), root_relative 는 (..., 21, 3)."""
    joints = root_relative + wrist.unsqueeze(-2)
    # root_relative[WRIST] 가 정확히 0 이 아닐 수 있으므로 wrist 를 강제로 맞춘다.
    return torch.cat([wrist.unsqueeze(-2), joints[..., 1:, :]], dim=-2)
