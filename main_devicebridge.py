"""ZMQ 를 말하지 못하는 기기들을 comm_hub 에 붙이는 bridge.

HL2 앱은 comm_hub 에 직접 붙는다. 와치(WebSocket)와 폰(TCP)은 그러지 못하고 앱도 바꾸지
않기로 했다. 그래서 기기별 프로토콜을 받아 comm_hub 로 올리는 어댑터를 이 프로세스 하나에
모은다. 기기가 늘어도 프로세스는 늘지 않는다.

    comm_hub.py            broker. 내용을 모른다
    main_devicebridge.py   이 파일. 와치(WebSocket), 폰(TCP)
    main_*.py              처리 모듈

실행
    python comm_hub.py
    python main_devicebridge.py          # 와치·폰 앱 화면에서 서버 IP 를 이 PC 로 바꾼다

와치: WATCH_HR / WATCH_ACTIVITY / WATCH_AUDIO / WATCH_VAD 로 올라간다 (watch.py).
폰:   PHONE_TO_HL2 로 올라가고 HL2_TO_PHONE 을 받아 폰에 내려보낸다 (phone.py).
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

from hub_client import (HubClient, add_broker_args, KW_WATCH_HR,         # noqa: E402
                        KW_PHONE_TO_HL2, KW_HL2_TO_PHONE)
from modules.devicebridge.phone import PhoneAdapter                     # noqa: E402
from modules.devicebridge.watch import WatchAdapter                     # noqa: E402

IDENTITY = b"DEVICEBRIDGE"
STATUS_PERIOD_S = 5.0


async def _status_loop(adapters) -> None:
    last = time.time()
    while True:
        await asyncio.sleep(STATUS_PERIOD_S)
        now = time.time()
        for a in adapters:
            print(a.status(now - last), flush=True)
        last = now


async def _main(args) -> None:
    # 와치는 올리기만 한다. 폰은 HL2_TO_PHONE 도 받아야 하므로 구독이 있는 클라이언트를
    # 따로 둔다. comm_hub 가 source 당 구독을 하나만 기억하므로 identity 도 다르다.
    watch_hub = HubClient(args.host, args.port, recv_kw=None, result_kw=KW_WATCH_HR,
                          identity=IDENTITY + b"_WATCH")
    phone_hub = HubClient(args.host, args.port, recv_kw=KW_HL2_TO_PHONE, result_kw=KW_PHONE_TO_HL2,
                          identity=IDENTITY + b"_PHONE", queue_size=64)   # 이벤트라 빠지면 안 된다
    watch = WatchAdapter(watch_hub, raw_audio=not args.no_raw_audio)
    phone = PhoneAdapter(phone_hub)
    try:
        await asyncio.gather(watch.serve(args.ws_host, args.ws_port),
                             phone.serve(args.tcp_host, args.tcp_port),
                             _status_loop([watch, phone]))
    finally:
        watch_hub.close()
        phone_hub.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="와치·폰처럼 ZMQ 를 못 쓰는 기기를 comm_hub 에 붙인다")
    add_broker_args(ap)
    ap.add_argument("--ws-host", default="0.0.0.0", help="와치 WebSocket 수신 주소")
    ap.add_argument("--ws-port", type=int, default=8765, help="와치 WebSocket 포트 (와치 앱 기본값 8765)")
    ap.add_argument("--tcp-host", default="0.0.0.0", help="폰 TCP 수신 주소")
    ap.add_argument("--tcp-port", type=int, default=8777, help="폰 TCP 포트 (폰 앱 기본값 8777)")
    ap.add_argument("--no-raw-audio", action="store_true",
                    help="WATCH_AUDIO 를 올리지 않는다. VAD 는 그대로 돌아 WATCH_VAD 만 나간다")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("종료", flush=True)


if __name__ == "__main__":
    main()
