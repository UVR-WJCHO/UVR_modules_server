"""Window index 를 GPU 텐서로 올려두고 batch 를 gather 한다.

frame 표는 window 보다 훨씬 작고(train 기준 약 150 MB) 한 frame 이 여러 window 의
history 에 재사용되므로, window 를 통째로 materialize 하지 않고 index 로만 들고 있다가
batch 마다 gather 하는 편이 메모리·속도 모두 낫다. DataLoader 없이 GPU 안에서 끝난다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..data.build_windows import load_canonical_frames
from ..data.canonical_schema import NUM_JOINTS

NS_PER_MS = 1_000_000.0


@dataclass
class WindowTensors:
    joints: torch.Tensor          # (R, 21, 3) float32, history 를 읽는 frame 표
    timestamp_ns: torch.Tensor    # (R,) int64
    history_rows: torch.Tensor    # (M, N) int64
    target_row: torch.Tensor      # (M,) int64
    horizon_ms: torch.Tensor      # (M,) float32
    requested_horizon_ms: np.ndarray
    sequence_id: np.ndarray
    handedness: np.ndarray
    handedness_index: torch.Tensor       # (M,) LEFT=0, RIGHT=1
    target_joints: torch.Tensor | None = None   # 주면 target 만 다른 표에서 읽는다
    # target 보간(15 fps 데이터셋에서 33 ms 배수 격자를 쓰기 위한 것). weight 0 이면
    # target_row 하나만 쓴다. history 는 절대 보간하지 않는다 — 배포 추정기의 잡음
    # 구조가 매끈해지면 이 논문의 학습 조건이 깨진다.
    target_row2: torch.Tensor | None = None
    target_weight: torch.Tensor | None = None
    # 시간적 일관성 loss 용. prev_index[i] 는 같은 track·같은 horizon 에서 anchor 가
    # 한 프레임 앞선 window 의 index 다. 없으면 자기 자신을 가리킨다.
    prev_index: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.target_row)

    @property
    def history_length(self) -> int:
        return self.history_rows.shape[1]

    def batch(self, index: torch.Tensor, noise=None,
              generator: torch.Generator | None = None,
              geom=None) -> dict[str, torch.Tensor]:
        """noise 를 주면 history 만 열화시킨다. target 은 항상 깨끗한 GT 다.

        조건 B(corrupted GT) 학습이 이 경로다. 열화는 batch 마다 새로 뽑아 같은 window
        가 매번 다른 잡음을 보게 한다.
        """
        rows = self.history_rows[index]                                  # (B, N)
        history = self.joints[rows]                                      # (B, N, 21, 3)
        ts = self.timestamp_ns[rows]
        history_time_ms = (ts - ts[:, -1:]).to(torch.float32) / NS_PER_MS
        visibility = torch.ones(history.shape[:3], device=history.device,
                                dtype=history.dtype)
        if noise is not None:
            from ..data.augmentations import corrupt_history

            history, visibility = corrupt_history(history, visibility, noise, generator)
        source = self.joints if self.target_joints is None else self.target_joints
        target = source[self.target_row[index]]
        if self.target_weight is not None:
            w = self.target_weight[index]
            if torch.any(w > 0):
                target = torch.lerp(target, source[self.target_row2[index]],
                                    w[:, None, None])
        handedness = self.handedness_index[index]
        # 기하 증강은 잡음 뒤에 온다. history 와 target 에 **같은** 변환을 걸어야
        # 클립 안의 속도·가속도가 보존된다 (augment_window 주석 참고).
        if geom is not None:
            from ..data.augmentations import augment_window

            history, target, handedness = augment_window(
                history, target, handedness, geom, generator)
        return {
            "history": history,
            "history_time_ms": history_time_ms,
            "horizon_ms": self.horizon_ms[index],
            "target": target,
            "anchor": history[:, -1],
            "visibility": visibility,
            "handedness": handedness,
        }


def _aligned_targets(history: pl.DataFrame, target_path: Path) -> np.ndarray:
    """target 표가 history 표와 행 단위로 정확히 대응하는지 확인하고 joint 를 돌려준다.

    두 표의 track 집합이 다르면 행 index 가 밀려 엉뚱한 프레임을 target 으로 읽는다.
    조용히 넘어가면 NaN 이 섞여 loss 가 inf 가 되므로 여기서 잡는다.
    """
    target = load_canonical_frames(target_path)
    if target.height != history.height:
        raise ValueError(
            f"{target_path.name}: 행 수가 history({history.height})와 다르다({target.height}). "
            "두 cache 의 track 집합이 어긋났다")
    for column in ("sequence_id", "handedness", "timestamp_ns"):
        if not (target[column].to_numpy() == history[column].to_numpy()).all():
            raise ValueError(f"{target_path.name}: {column} 이 history 와 정렬되지 않는다")
    return target["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3)


def _previous_window(windows: pl.DataFrame) -> np.ndarray:
    """anchor 가 한 프레임 앞선 같은 track·같은 horizon window 의 행 index.

    시간적 일관성은 **연속한 두 anchor 의 예측**을 비교해야 잴 수 있다. 학습은 window
    를 무작위로 섞으므로 그 짝을 미리 찾아 둔다. 짝이 없으면 자기 자신을 가리켜
    차이가 0 이 되고, loss 에 기여하지 않는다.
    """
    key = (windows["track_id"].to_numpy().astype(str) + "|"
           + windows["requested_horizon_ms"].to_numpy().astype(str))
    anchor = windows["anchor_row"].to_numpy().astype(np.int64)
    order = np.arange(len(anchor))
    lookup = {(k, int(a)): i for k, a, i in zip(key, anchor, order)}
    prev = np.array([lookup.get((k, int(a) - 1), i)
                     for k, a, i in zip(key, anchor, order)], dtype=np.int64)
    return prev


def load_split(out_dir: Path, split: str, device: str = "cuda",
               horizons_ms: list[float] | None = None,
               history_file: str | None = None, target_file: str | None = None,
               windows_file: str | None = None) -> WindowTensors:
    """history/target/window 를 각각 다른 파일에서 읽을 수 있다.

    배포 조건으로 검증하려면 history=WiLoR, target=GT 로 나눠야 한다.
    """
    frames = load_canonical_frames(
        out_dir / (history_file or f"canonical_frames_{split}.parquet"))
    windows = pl.read_parquet(out_dir / (windows_file or f"windows_{split}.parquet"))
    if horizons_ms is not None:
        windows = windows.filter(pl.col("requested_horizon_ms").is_in(horizons_ms))

    joints = frames["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3)
    # 무효 frame 은 NaN 이다. window 는 유효 frame 만 참조하므로 학습에는 들어오지
    # 않지만, gather 실수를 조용히 넘기지 않도록 NaN 을 그대로 둔다.
    return WindowTensors(
        joints=torch.as_tensor(joints, dtype=torch.float32, device=device),
        timestamp_ns=torch.as_tensor(frames["timestamp_ns"].to_numpy(), device=device),
        history_rows=torch.as_tensor(
            windows["history_rows"].to_numpy().astype(np.int64), device=device),
        target_row=torch.as_tensor(
            windows["target_row"].to_numpy().astype(np.int64), device=device),
        horizon_ms=torch.as_tensor(
            windows["horizon_ms"].to_numpy(), dtype=torch.float32, device=device),
        requested_horizon_ms=windows["requested_horizon_ms"].to_numpy(),
        sequence_id=windows["sequence_id"].to_numpy(),
        handedness=windows["handedness"].to_numpy(),
        handedness_index=torch.as_tensor(
            (windows["handedness"].to_numpy() == "RIGHT").astype(np.int64), device=device),
        target_joints=(None if target_file is None else torch.as_tensor(
            _aligned_targets(frames, out_dir / target_file), dtype=torch.float32, device=device)),
        prev_index=torch.as_tensor(_previous_window(windows), device=device),
        # 보간 컬럼은 새 window 표에만 있다. 옛 표는 그대로 읽히고 보간 없이 동작한다.
        target_row2=(torch.as_tensor(windows["target_row2"].to_numpy().astype(np.int64),
                                     device=device)
                     if "target_row2" in windows.columns else None),
        target_weight=(torch.as_tensor(windows["target_weight"].to_numpy(),
                                       dtype=torch.float32, device=device)
                       if "target_weight" in windows.columns else None),
    )
