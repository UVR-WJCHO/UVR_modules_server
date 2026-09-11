"""Spatial Input 패킷을 지표 계산이 기대하는 배열 형태로 편다.

이 저장소의 `_hl2ss/` 는 기기에 올라간 hl2ss 앱과 같은 버전이어야 하므로 그대로 둔다.
그 버전의 SI API 는 관절 하나씩 꺼내는 형태다.

    si = hl2ss.unpack_si(packet.payload)
    si.is_valid_hand_left()          # bool
    si.get_hand_left().get_joint_pose(j)   # j = 0..25, 하나씩

최신 hl2ss 는 `si.hand_left.position` 이 (26, 3) 배열로 바로 나오는데, Adaptive-MM-UI
쪽 지표 코드가 그 형태를 전제한다. 여기서 한 번 펴 주면 지표 코드를 고치지 않아도 된다.

반환은 모두 numpy 배열이며, 유효하지 않은 항목은 None 이다. 호출자는 valid 플래그가
아니라 None 여부로 분기하면 된다.
"""
from __future__ import annotations

import numpy as np

# 손 관절 수. hl2ss.SI_HandJointKind.TOTAL 과 같아야 한다.
NUM_HAND_JOINTS = 26


class SIFrame:
    """SI 한 패킷을 편 결과.

    head_position / head_forward / head_up : (3,)  또는 None
    eye_origin / eye_direction             : (3,)  또는 None
    hand_left / hand_right                 : dict  또는 None
        {'position': (26,3), 'orientation': (26,4), 'radius': (26,), 'accuracy': (26,)}
    """

    __slots__ = ("timestamp", "head_position", "head_forward", "head_up",
                 "eye_origin", "eye_direction", "hand_left", "hand_right")

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.head_position = self.head_forward = self.head_up = None
        self.eye_origin = self.eye_direction = None
        self.hand_left = self.hand_right = None

    @property
    def has_head(self) -> bool:
        return self.head_position is not None

    @property
    def has_eye(self) -> bool:
        return self.eye_direction is not None

    def hand(self, side: str):
        """side: 'L' 또는 'R'."""
        return self.hand_left if side == "L" else self.hand_right


def _flatten_hand(hand) -> dict:
    """`_SI_Hand` 를 관절별로 훑어 (26, ...) 배열로 쌓는다."""
    orientation = np.empty((NUM_HAND_JOINTS, 4), np.float32)
    position = np.empty((NUM_HAND_JOINTS, 3), np.float32)
    radius = np.empty(NUM_HAND_JOINTS, np.float32)
    accuracy = np.empty(NUM_HAND_JOINTS, np.int32)
    for j in range(NUM_HAND_JOINTS):
        p = hand.get_joint_pose(j)
        orientation[j] = p.orientation
        position[j] = p.position
        radius[j] = p.radius
        accuracy[j] = p.accuracy
    return {"orientation": orientation, "position": position,
            "radius": radius, "accuracy": accuracy}


def unpack(hl2ss_module, packet) -> SIFrame:
    """SI 패킷 하나를 SIFrame 으로.

    hl2ss_module 을 인자로 받는 이유는 이 파일이 `_hl2ss` 를 sys.path 에 올리는 책임을
    지지 않기 위해서다. 진입점이 이미 올려 둔 모듈을 그대로 넘긴다.
    """
    out = SIFrame(packet.timestamp)
    si = hl2ss_module.unpack_si(packet.payload)

    if si.is_valid_head_pose():
        hp = si.get_head_pose()
        out.head_position = np.asarray(hp.position, np.float32)
        out.head_forward = np.asarray(hp.forward, np.float32)
        out.head_up = np.asarray(hp.up, np.float32)

    if si.is_valid_eye_ray():
        er = si.get_eye_ray()
        out.eye_origin = np.asarray(er.origin, np.float32)
        out.eye_direction = np.asarray(er.direction, np.float32)

    if si.is_valid_hand_left():
        out.hand_left = _flatten_hand(si.get_hand_left())
    if si.is_valid_hand_right():
        out.hand_right = _flatten_hand(si.get_hand_right())

    return out
