"""bridge 가 자기 와치·폰이 어느 HL2 와 짝인지 DEVICE_PAIR 로 알린다.

HQS(main_all_hl2_receiver.py)는 HL2 를 기기(clientIdentity)별로 따로 계산한다. 와치·폰 메시지에는
HL2 기기 ID 가 없으므로, receiver 는 이 알림을 보고 짝인 기기의 HQS 에만 넣는다.
나중에 뜬 receiver 도 알 수 있도록 주기적으로 다시 보낸다.
"""
from __future__ import annotations

import asyncio
import json

from hub_client import HubClient, KW_DEVICE_PAIR

PAIR_PERIOD_S = 2.0                  # 다시 알리는 주기
PAIR_TIMEOUT_S = 3 * PAIR_PERIOD_S   # 받는 쪽은 이만큼 알림이 없으면(bridge 꺼짐) 짝이 없다고 본다


async def announce_pair(hub: HubClient, kind: str, device: str) -> None:
    """kind: "watch" / "phone". device: 짝인 HL2 의 clientIdentity."""
    msg = json.dumps({"kind": kind, "device": device}, separators=(",", ":")).encode()
    while True:
        hub.send(msg, KW_DEVICE_PAIR)
        await asyncio.sleep(PAIR_PERIOD_S)
