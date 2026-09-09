"""Stage M1/M2 — delay-conditioned causal TCN (B 계획 §7.2, §7.3).

    per-frame input : wrist 기준 joint, wrist 속도, visibility, delta time
    frame encoder   : Linear -> GELU -> Linear, 128 channel
    causal TCN      : kernel 3, dilation 1/2/4, 128 channel, **left padding 만**
    horizon encoder : scalar + sinusoidal -> 32D
    fusion          : 마지막 causal state + horizon embedding + handedness embedding
    output head     : constant-velocity base 위의 63D residual

같은 클래스로 §10.1/§10.2 의 대조군을 모두 만든다.
  - `use_horizon_encoding=False` : fixed-horizon TCN (horizon 별로 따로 학습)
  - `use_cv_base=False`          : CV residual base 제거 ablation
  - `decomposed=True`            : M2, wrist 3D 와 root-relative 60D 를 따로 예측

모든 convolution 은 왼쪽만 padding 한다. 미래 입력이 현재 출력에 영향을 주지 않는지는
tests/test_tcn_causality.py 가 확인한다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..data.canonical_schema import NUM_JOINTS, WRIST
from .decomposition import compose
from .features import (CLIP_SPEED_MPS, HORIZON_ENCODING_DIM, constant_velocity_base,
                       horizon_encoding)

PER_FRAME_DIM = NUM_JOINTS * 3 + 3 + NUM_JOINTS + 1     # rel joints, wrist 속도, visibility, dt


class CausalBlock(nn.Module):
    """kernel 3 dilated causal convolution + residual.

    정규화는 **timestep 별 channel LayerNorm** 이어야 한다. GroupNorm/BatchNorm 은
    시간축까지 묶어 통계를 내므로 미래 프레임이 과거 출력에 새어 들어간다
    (tests/test_tcn.py 의 causality 테스트가 이걸 잡는다).
    """

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation                          # 왼쪽만
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))       # (B, C, N)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)         # channel 축만 정규화
        return x + self.drop(self.act(y))


_UNSET = object()          # "키를 아예 안 줬다" 와 "null 을 줬다" 를 가른다


class TCNLite(nn.Module):
    def __init__(self, history_length: int = 8, channels: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4), dropout: float = 0.1,
                 horizon_dim: int = 32, use_horizon_encoding: bool = True,
                 use_cv_base: bool = True, decomposed: bool = False,
                 base_n_fit: int = 2, clip_speed=_UNSET,
                 learn_base_gain: bool = True, shared_base_gain: bool = False):
        super().__init__()
        self.history_length = history_length
        self.use_horizon_encoding = use_horizon_encoding
        self.use_cv_base = use_cv_base
        self.decomposed = decomposed
        self.base_n_fit = base_n_fit
        # 좌표계마다 단위가 다르다. 키를 안 주면 기존 기본값, null 이면 상한 없음,
        # 두 값을 주면 (광선, rel_z) 로 따로 자른다.
        self.clip_speed = (CLIP_SPEED_MPS if clip_speed is _UNSET
                           else None if clip_speed is None
                           else tuple(clip_speed) if isinstance(clip_speed, (list, tuple))
                           else float(clip_speed))

        # 관절별 CV base 이득. 1.0 에서 출발해 학습이 관절마다 조정한다.
        # 손끝은 굽혔다 펴는 왕복이라 선형 외삽이 지나치는데(실측: 손끝에서 cv_2frame 이
        # hold 보다 나빴다, 137 vs 103), 이 값이 내려가면 그만큼 base 를 덜 믿게 된다.
        # 켜고 끄는 경계를 만들지 않으므로 관절 사이가 어긋나지 않는다.
        # shared_base_gain 을 켜면 21 개 대신 하나를 학습한다. 관절별로 나누는 것이
        # 실제로 이득인지 재는 ablation 용이고, broadcasting 은 그대로 동작한다.
        self.base_gain = nn.Parameter(torch.ones(1 if shared_base_gain else NUM_JOINTS, 1),
                                      requires_grad=learn_base_gain)

        self.frame_encoder = nn.Sequential(
            nn.Linear(PER_FRAME_DIM, channels), nn.GELU(), nn.Linear(channels, channels))
        self.blocks = nn.ModuleList(
            [CausalBlock(channels, d, dropout) for d in dilations])
        self.handedness = nn.Embedding(2, 16)

        fused = channels + 16 + (horizon_dim if use_horizon_encoding else 0)
        if use_horizon_encoding:
            self.horizon_encoder = nn.Sequential(
                nn.Linear(HORIZON_ENCODING_DIM, horizon_dim), nn.GELU(),
                nn.Linear(horizon_dim, horizon_dim))

        # trunk 는 공유하고 마지막 linear 만 나눈다. 그래야 M1(63D 하나)과
        # M2(3D + 60D)의 parameter 수가 같아져, 이득이 용량 차이에서 오지 않는다 (§6.5).
        self.trunk = nn.Sequential(nn.Linear(fused, channels), nn.GELU())
        if decomposed:
            self.global_head = nn.Linear(channels, 3)
            self.local_head = nn.Linear(channels, (NUM_JOINTS - 1) * 3)
            heads = (self.global_head, self.local_head)
        else:
            self.head = nn.Linear(channels, NUM_JOINTS * 3)
            heads = (self.head,)
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.register_buffer("feature_mean", torch.zeros(PER_FRAME_DIM))
        self.register_buffer("feature_std", torch.ones(PER_FRAME_DIM))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-6))

    def per_frame_features(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                           visibility: torch.Tensor | None) -> torch.Tensor:
        """(B, N, 21, 3) -> (B, N, PER_FRAME_DIM). anchor 의 wrist 를 원점으로 둔다."""
        B, N = history_time_ms.shape
        wrist = history[:, -1, WRIST]                                   # (B, 3)
        relative = (history - wrist.view(B, 1, 1, 3)).reshape(B, N, -1)  # (B, N, 63)

        # torch.diff 는 ONNX export 가 안 되므로 slicing 뺄셈으로 쓴다.
        dt_ms = torch.cat([history_time_ms[:, :1] * 0,
                           history_time_ms[:, 1:] - history_time_ms[:, :-1]], dim=1)
        wrist_track = history[:, :, WRIST]                              # (B, N, 3)
        wrist_delta = torch.cat([wrist_track[:, :1] * 0,
                                 wrist_track[:, 1:] - wrist_track[:, :-1]], dim=1)
        velocity = wrist_delta / (dt_ms.unsqueeze(-1) / 1000.0).clamp(min=1e-3)
        if visibility is None:
            visibility = torch.ones(B, N, NUM_JOINTS, device=history.device)
        return torch.cat([relative, velocity, visibility.to(history.dtype),
                          dt_ms.unsqueeze(-1) / 100.0], dim=-1)

    def forward(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                horizon_ms: torch.Tensor, handedness: torch.Tensor | None = None,
                visibility: torch.Tensor | None = None) -> torch.Tensor:
        features = self.per_frame_features(history, history_time_ms, visibility)
        normalized = (features - self.feature_mean) / self.feature_std

        x = self.frame_encoder(normalized).transpose(1, 2)              # (B, C, N)
        for block in self.blocks:
            x = block(x)
        state = x[:, :, -1]                                             # anchor 시점 state

        if handedness is None:
            handedness = torch.ones(len(history), dtype=torch.long, device=history.device)
        parts = [state, self.handedness(handedness)]
        if self.use_horizon_encoding:
            parts.append(self.horizon_encoder(horizon_encoding(horizon_ms)))
        hidden = self.trunk(torch.cat(parts, dim=-1))

        anchor = history[:, -1]
        base = anchor
        if self.use_cv_base:
            raw = constant_velocity_base(history, history_time_ms, horizon_ms,
                                         self.base_n_fit, self.clip_speed)
            base = anchor + (raw - anchor) * self.base_gain

        if self.decomposed:
            wrist = base[:, WRIST] + self.global_head(hidden)
            local = base[:, 1:] - base[:, WRIST:WRIST + 1] \
                + self.local_head(hidden).view(-1, NUM_JOINTS - 1, 3)
            return compose(wrist, torch.cat(
                [torch.zeros_like(local[:, :1]), local], dim=1))
        return base + self.head(hidden).view(-1, NUM_JOINTS, 3)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
