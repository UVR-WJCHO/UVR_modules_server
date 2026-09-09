"""Stage M0 — constant-velocity residual MLP (B 계획 §7.1).

연구 최종안이 아니라 B 방식의 feasibility gate 다. TCN 을 만들기 전에
"delay-conditioned 학습이 CV 를 이길 수 있는가"만 본다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..data.canonical_schema import NUM_JOINTS
from .features import build_features, constant_velocity_base, feature_dim


class ResidualMLP(nn.Module):
    def __init__(self, history_length: int = 8, hidden_width: int = 256,
                 hidden_layers: int = 3, dropout: float = 0.1,
                 base_n_fit: int = 2, accel_n_fit: int = 4):
        super().__init__()
        self.history_length = history_length
        self.base_n_fit = base_n_fit
        self.accel_n_fit = accel_n_fit

        in_dim = feature_dim(history_length, NUM_JOINTS)
        layers: list[nn.Module] = []
        for i in range(hidden_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden_width, hidden_width),
                       nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_width, NUM_JOINTS * 3))
        self.net = nn.Sequential(*layers)
        # residual 은 0 에서 출발해야 학습 초기에 CV base 를 망가뜨리지 않는다
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.register_buffer("feature_mean", torch.zeros(in_dim))
        self.register_buffer("feature_std", torch.ones(in_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-6))

    def forward(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                horizon_ms: torch.Tensor, handedness: torch.Tensor | None = None,
                visibility: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, 21, 3) -> (B, 21, 3) world 예측.

        handedness/visibility 는 M0 명세(§7.1)에 없어 받기만 하고 쓰지 않는다.
        이 둘을 쓰는 M1/M2 와의 차이가 곧 그 입력의 가치가 된다.
        """
        base = constant_velocity_base(history, history_time_ms, horizon_ms, self.base_n_fit)
        features = build_features(history, history_time_ms, horizon_ms, self.accel_n_fit)
        normalized = (features - self.feature_mean) / self.feature_std
        residual = self.net(normalized).view(-1, NUM_JOINTS, 3)
        return base + residual

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
