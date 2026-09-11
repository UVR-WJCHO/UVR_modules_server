"""HL2 센서 스트림에서 사용자 상태를 정량화한다.

Adaptive-MM-UI 의 Server/modules 를 이 저장소로 옮긴 것이다. 지표 계산은 그대로이고,
달라진 것은 두 가지다.

  - SI(head/eye/hands) 접근은 `si_adapter` 를 거친다. 이 저장소의 `_hl2ss/` 는 기기 앱과
    같은 버전이어야 하므로 최신판의 배열 API 대신 관절별 `get_joint_pose()` 를 쓴다.
  - 녹화는 `recorder` 가 맡는다. 이미지와 이벤트는 스트리밍 append, 고정폭 배열은 NPZ.
"""
