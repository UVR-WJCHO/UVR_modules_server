"""서버가 유지하는 (device, track, hand) 별 anchor ring buffer (B 계획 §2.4, §5).

nowcaster 는 과거 N 프레임을 본다. 그런데 `main_handtrack.py` 는 프레임마다 독립
처리라 history 가 없다. 이 모듈이 그 상태를 담당한다.

계획서 §2.4 의 규칙을 그대로 구현한다.
  - all-ones dummy pose 로 validity 를 표현하지 않는다. 무효 프레임은 아예 넣지 않는다
  - 같은 hand history 에서 source timestamp 가 역행하면 그 packet 을 버린다
  - 두 유효 anchor 간 간격이 gap 한계를 넘거나 track 이 바뀌면 history 를 reset 한다
  - history key 는 최소 (device_id, track_id, handedness)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..data.canonical_schema import NUM_JOINTS

NS_PER_MS = 1_000_000.0
DEFAULT_HISTORY = 8
DEFAULT_GAP_MS = 250.0          # 이 이상 벌어지면 연속된 움직임으로 볼 수 없다


@dataclass
class Anchor:
    timestamp_ns: int
    joints_world: np.ndarray     # (21, 3)
    frame_id: int = 0


@dataclass
class TrackHistory:
    history_length: int = DEFAULT_HISTORY
    max_gap_ms: float = DEFAULT_GAP_MS
    anchors: deque = field(default_factory=deque)
    n_reset: int = 0
    n_rejected_regression: int = 0

    def push(self, anchor: Anchor) -> str:
        """anchor 를 넣고 무슨 일이 있었는지 돌려준다: ok / regressed / reset."""
        if self.anchors:
            previous = self.anchors[-1]
            if anchor.timestamp_ns <= previous.timestamp_ns:
                self.n_rejected_regression += 1
                return "regressed"
            gap_ms = (anchor.timestamp_ns - previous.timestamp_ns) / NS_PER_MS
            if gap_ms > self.max_gap_ms:
                self.anchors.clear()
                self.n_reset += 1
                self.anchors.append(anchor)
                return "reset"
        self.anchors.append(anchor)
        while len(self.anchors) > self.history_length:
            self.anchors.popleft()
        return "ok"

    @property
    def ready(self) -> bool:
        return len(self.anchors) == self.history_length

    def tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """(N, 21, 3) joints 와 (N,) anchor 기준 상대 시각(ms, 과거가 음수)."""
        joints = np.stack([a.joints_world for a in self.anchors])
        ts = np.array([a.timestamp_ns for a in self.anchors], dtype=np.int64)
        return joints, (ts - ts[-1]).astype(np.float64) / NS_PER_MS


class HistoryStore:
    """여러 device/track/hand 의 history 를 담는다. key 가 다르면 서로 격리된다."""

    def __init__(self, history_length: int = DEFAULT_HISTORY,
                 max_gap_ms: float = DEFAULT_GAP_MS):
        self.history_length = history_length
        self.max_gap_ms = max_gap_ms
        self._tracks: dict[tuple[str, str, str], TrackHistory] = {}

    def key(self, device_id: str, track_id: str, handedness: str) -> tuple[str, str, str]:
        return (str(device_id), str(track_id), str(handedness))

    def push(self, device_id: str, track_id: str, handedness: str,
             timestamp_ns: int, joints_world: np.ndarray, frame_id: int = 0) -> str:
        joints = np.asarray(joints_world, dtype=np.float64)
        if joints.shape != (NUM_JOINTS, 3):
            raise ValueError(f"joints shape 이 ({NUM_JOINTS}, 3) 이 아니다: {joints.shape}")
        if not np.isfinite(joints).all():
            return "invalid"            # 무효 pose 는 history 에 넣지 않는다
        key = self.key(device_id, track_id, handedness)
        track = self._tracks.get(key)
        if track is None:
            track = TrackHistory(self.history_length, self.max_gap_ms)
            self._tracks[key] = track
        return track.push(Anchor(int(timestamp_ns), joints, int(frame_id)))

    def get(self, device_id: str, track_id: str, handedness: str) -> TrackHistory | None:
        return self._tracks.get(self.key(device_id, track_id, handedness))

    def ready(self, device_id: str, track_id: str, handedness: str) -> bool:
        track = self.get(device_id, track_id, handedness)
        return bool(track and track.ready)

    def drop(self, device_id: str, track_id: str, handedness: str) -> None:
        self._tracks.pop(self.key(device_id, track_id, handedness), None)

    def stats(self) -> dict:
        return {
            "n_tracks": len(self._tracks),
            "n_ready": sum(1 for t in self._tracks.values() if t.ready),
            "n_reset": sum(t.n_reset for t in self._tracks.values()),
            "n_rejected_regression": sum(t.n_rejected_regression
                                         for t in self._tracks.values()),
        }
