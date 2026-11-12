"""폰(TCP)을 comm_hub 에 붙이는 bridge.

폰 앱(RoverHandoff)은 ZMQ 를 못 쓰고 앱도 바꾸지 않기로 했다. 폰이 TCP 로 보내는 메시지를
comm_hub 로 올리고, HL2 가 폰에 보내는 메시지를 받아 폰에 내려보낸다. 와치는 main_watchbridge.py 가
따로 맡는다.

    comm_hub.py            broker. 내용을 모른다
    main_watchbridge.py    와치(WebSocket) -> comm_hub
    main_phonebridge.py    이 파일. 폰(TCP) <-> comm_hub
    main_*.py              처리 모듈

실행
    python comm_hub.py
    python main_phonebridge.py --device HoloLens2_A   # 폰 앱 화면에서 서버 IP 를 이 PC 로 바꾼다

폰: PHONE_TO_HL2 로 올라가고, HL2_TO_PHONE 중 --device 기기가 보낸 것을 받아 폰에 내려보낸다
(modules/devicebridge/phone.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "_comm"))
sys.path.insert(0, _ROOT)

from hub_client import (HubClient, add_broker_args,                      # noqa: E402
                        KW_PHONE_TO_HL2, KW_HL2_TO_PHONE)
from modules.devicebridge.phone import PhoneAdapter                     # noqa: E402

IDENTITY = b"DEVICEBRIDGE_PHONE"    # 와치·폰 bridge 를 나누기 전과 같은 identity
STATUS_PERIOD_S = 5.0


async def _status_loop(adapter) -> None:
    last = time.time()
    while True:
        await asyncio.sleep(STATUS_PERIOD_S)
        now = time.time()
        print(adapter.status(now - last), flush=True)
        last = now


async def _main(args) -> None:
    hub = HubClient(args.host, args.port, recv_kw=KW_HL2_TO_PHONE, result_kw=KW_PHONE_TO_HL2,
                    identity=IDENTITY, queue_size=64,               # 이벤트라 빠지면 안 된다
                    from_source=args.device.encode() if args.device else None)
    phone = PhoneAdapter(hub)
    try:
        await asyncio.gather(phone.serve(args.tcp_host, args.tcp_port), _status_loop(phone))
    finally:
        hub.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="폰을 comm_hub 에 붙인다")
    add_broker_args(ap)
    ap.add_argument("--device", default=None,
                    help="폰과 연동할 HL2 의 clientIdentity. HL2_TO_PHONE 은 이 기기가 보낸 것만 폰에 넘긴다. "
                         "주지 않으면 모든 기기")
    ap.add_argument("--tcp-host", default="0.0.0.0", help="폰 TCP 수신 주소")
    ap.add_argument("--tcp-port", type=int, default=8777, help="폰 TCP 포트 (폰 앱 기본값 8777)")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("종료", flush=True)


if __name__ == "__main__":
    main()
