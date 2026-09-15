"""세 가지 시간 단위를 하나로 맞춘다.

지표 코드는 전부 QPC(100 ns 정수)를 전제한다(`common.QPC = 1e7`). 2 초 창이
`2 * QPC` ticks 로 계산되므로, 단위가 다른 값을 그대로 넣으면 창 길이가 엉뚱해지고
recency 가중이 걸리지 않는다. 오류는 나지 않고 값만 뭉개진다.

입력마다 단위가 다르다.

    hl2ss 스트림              QPC 100 ns 정수      그대로
    HL2SensorPacket.timestamp Unity Time.time 초   float32, 앱 기동이 원점
    HL2Audio.start_timestamp  나노초 정수          같은 원점

여기서 전부 QPC 로 올린다. 원점은 어느 경로든 세션 안에서 일관되므로, 창 길이와
recency 가중만 맞으면 지표는 정상 동작한다.

주의: `Time.time` 은 float32 라 유효숫자가 7 자리다. 기동 후 1000 초면 해상도가
약 0.1 ms, 10000 초면 약 1 ms 로 떨어진다. 창이 2 초인 지표에는 무해하지만, 더 짧은
창을 쓰게 되면 다시 따져야 한다.
"""
from __future__ import annotations

QPC_PER_SEC = 10_000_000        # 100 ns ticks
QPC_PER_NS = 0.01               # 1 ns = 0.01 tick


def from_seconds(t: float) -> int:
    """Unity `Time.time` 등 초 단위 실수 -> QPC."""
    return int(round(float(t) * QPC_PER_SEC))


def from_nanoseconds(t: int) -> int:
    """`HL2Audio.start_timestamp` 등 나노초 정수 -> QPC."""
    return int(round(int(t) * QPC_PER_NS))


def from_qpc(t: int) -> int:
    """hl2ss 패킷. 이미 QPC 다."""
    return int(t)


def to_seconds(qpc: int) -> float:
    return float(qpc) / QPC_PER_SEC
