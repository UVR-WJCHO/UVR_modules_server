"""관절 그래프를 명시한 delay-conditioned nowcaster.

`tcn_lite` 는 21 관절을 63D 평탄 벡터로 다루므로 손의 운동학적 연결을 데이터에서
배워야 한다. 여기서는 관절을 **노드로 두고 뼈대로 이웃을 정의**해, 각 관절이 자기
이웃하고만 섞이도록 한다. 선행 연구가 손 자세에 graph convolution 을 쓰는 근거와 같다.

  per-joint 입력 : wrist 기준 좌표 3 + 관절 속도 3 + visibility 1 = 7
  graph block    : 자기 자신 / 이웃 / 손목 세 갈래를 각각 선형 변환해 합친다
  time           : 관절별로 독립인 causal TCN (tcn_lite 와 같은 규약, left padding)
  fusion         : 마지막 시점 상태 + horizon + handedness
  출력           : CV base 위의 residual (tcn_lite 와 동일)

용량은 tcn_lite 와 맞춘다. 이득이 구조에서 오는지 파라미터에서 오는지 갈라야 한다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..data.canonical_schema import BONES, NUM_JOINTS, WRIST
from .decomposition import compose
from .features import HORIZON_ENCODING_DIM, constant_velocity_base, horizon_encoding

PER_JOINT_DIM = 3 + 3 + 1        # wrist 기준 좌표, 속도, visibility


def _adjacency() -> torch.Tensor:
    """뼈대로 정의한 대칭 인접행렬. 행 정규화해 이웃 평균이 되게 한다."""
    a = torch.zeros(NUM_JOINTS, NUM_JOINTS)
    for i, j in BONES:
        a[i, j] = a[j, i] = 1.0
    return a / a.sum(dim=1, keepdim=True).clamp_min(1.0)


class GraphBlock(nn.Module):
    """자기 자신 / 이웃 / 손목 세 갈래. 손목을 따로 두는 이유는 모든 관절이 손목
    기준으로 표현되어 있어 전역 기준점 역할을 하기 때문이다."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.self_fc = nn.Linear(channels, channels)
        self.neigh_fc = nn.Linear(channels, channels)
        self.wrist_fc = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.register_buffer("adj", _adjacency())

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, J, C)
        neigh = torch.einsum("jk,btkc->btjc", self.adj, x)
        wrist = x[:, :, WRIST:WRIST + 1].expand_as(x)
        y = self.self_fc(x) + self.neigh_fc(neigh) + self.wrist_fc(wrist)
        return x + self.drop(self.act(self.norm(y)))


class TemporalBlock(nn.Module):
    """관절별로 독립인 causal convolution. 정규화는 timestep 별 channel LayerNorm
    이어야 미래가 과거로 새지 않는다 (tcn_lite 와 같은 이유)."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, J, C)
        B, T, J, C = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B * J, C, T)       # 관절을 batch 로 접는다
        y = self.conv(nn.functional.pad(y, (self.pad, 0)))
        y = y.reshape(B, J, C, T).permute(0, 3, 1, 2)
        return x + self.drop(self.act(self.norm(y)))


class TCNGraph(nn.Module):
    def __init__(self, history_length: int = 8, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4), dropout: float = 0.1,
                 horizon_dim: int = 32, n_graph: int = 2,
                 use_horizon_encoding: bool = True, use_cv_base: bool = True,
                 decomposed: bool = False, base_n_fit: int = 2):
        super().__init__()
        self.history_length = history_length
        self.use_horizon_encoding = use_horizon_encoding
        self.use_cv_base = use_cv_base
        self.decomposed = decomposed
        self.base_n_fit = base_n_fit

        self.embed = nn.Linear(PER_JOINT_DIM, channels)
        self.graph = nn.ModuleList(GraphBlock(channels, dropout) for _ in range(n_graph))
        self.time = nn.ModuleList(TemporalBlock(channels, d, dropout) for d in dilations)
        self.handedness = nn.Embedding(2, 16)
        if use_horizon_encoding:
            self.horizon_encoder = nn.Sequential(
                nn.Linear(HORIZON_ENCODING_DIM, horizon_dim), nn.GELU(),
                nn.Linear(horizon_dim, horizon_dim))

        fused = channels + 16 + (horizon_dim if use_horizon_encoding else 0)
        self.trunk = nn.Sequential(nn.Linear(fused, channels), nn.GELU())
        # 관절마다 자기 상태로 자기 residual 을 낸다. 평탄 벡터 head 와 달리 관절 간
        # 파라미터를 공유하므로 용량이 작다.
        self.head = nn.Linear(channels, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # 관절 차원을 유지하므로 중간 텐서가 tcn_lite 의 21 배다. 평가 chunk 를 줄인다.
        self.eval_chunk = 20_000

        self.register_buffer("feature_mean", torch.zeros(PER_JOINT_DIM))
        self.register_buffer("feature_std", torch.ones(PER_JOINT_DIM))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-6))

    def per_frame_features(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                           visibility: torch.Tensor | None) -> torch.Tensor:
        """(B, N, 21, 3) -> (B, N, 21, 7). 관절마다 하나의 특징 벡터를 만든다."""
        B, N = history_time_ms.shape
        wrist = history[:, -1, WRIST]
        relative = history - wrist.view(B, 1, 1, 3)
        dt_ms = torch.cat([history_time_ms[:, :1] * 0,
                           history_time_ms[:, 1:] - history_time_ms[:, :-1]], dim=1)
        delta = torch.cat([history[:, :1] * 0, history[:, 1:] - history[:, :-1]], dim=1)
        velocity = delta / (dt_ms.view(B, N, 1, 1) / 1000.0).clamp(min=1e-3)
        if visibility is None:
            visibility = torch.ones(B, N, NUM_JOINTS, device=history.device)
        return torch.cat([relative, velocity, visibility.to(history.dtype).unsqueeze(-1)], -1)

    def forward(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                horizon_ms: torch.Tensor, handedness: torch.Tensor | None = None,
                visibility: torch.Tensor | None = None) -> torch.Tensor:
        f = self.per_frame_features(history, history_time_ms, visibility)
        x = self.embed((f - self.feature_mean) / self.feature_std)   # (B, N, J, C)
        for g in self.graph:
            x = g(x)
        for t in self.time:
            x = t(x)
        state = x[:, -1]                                             # (B, J, C)

        if handedness is None:
            handedness = torch.ones(len(history), dtype=torch.long, device=history.device)
        parts = [state, self.handedness(handedness).unsqueeze(1).expand(-1, NUM_JOINTS, -1)]
        if self.use_horizon_encoding:
            h = self.horizon_encoder(horizon_encoding(horizon_ms))
            parts.append(h.unsqueeze(1).expand(-1, NUM_JOINTS, -1))
        hidden = self.trunk(torch.cat(parts, dim=-1))

        anchor = history[:, -1]
        base = (constant_velocity_base(history, history_time_ms, horizon_ms, self.base_n_fit)
                if self.use_cv_base else anchor)
        residual = self.head(hidden)
        if self.decomposed:
            wrist = base[:, WRIST] + residual[:, WRIST]
            local = base[:, 1:] - base[:, WRIST:WRIST + 1] + residual[:, 1:]
            return compose(wrist, torch.cat([torch.zeros_like(local[:, :1]), local], 1))
        return base + residual

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
