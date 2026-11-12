"""와치(WebSocket)를 comm_hub 에 붙이는 bridge.

와치 앱은 ZMQ 를 못 쓰고 앱도 바꾸지 않기로 했다. 와치가 WebSocket 으로 보내는 메시지를 받아
comm_hub 로 올린다. 폰은 main_phonebridge.py 가 따로 맡는다.

    comm_hub.py            broker. 내용을 모른다
    main_watchbridge.py    이 파일. 와치(WebSocket) -> comm_hub
    main_phonebridge.py    폰(TCP) <-> comm_hub
    main_*.py              처리 모듈

실행
    python comm_hub.py
    python main_watchbridge.py           # 와치 앱 화면에서 서버 IP 를 이 PC 로 바꾼다

와치: WATCH_HR / WATCH_ACTIVITY / WATCH_AUDIO / WATCH_VAD 로 올라간다 (modules/devicebridge/watch.py).
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

from hub_client import HubClient, add_broker_args, KW_WATCH_HR          # noqa: E402
from modules.devicebridge.watch import WatchAdapter                     # noqa: E402

IDENTITY = b"DEVICEBRIDGE_WATCH"    # 와치·폰 bridge 를 나누기 전과 같은 identity
STATUS_PERIOD_S = 5.0


async def _status_loop(adapter) -> None:
    last = time.time()
    while True:
        await asyncio.sleep(STATUS_PERIOD_S)
        now = time.time()
        print(adapter.status(now - last), flush=True)
        last = now


async def _main(args) -> None:
    hub = HubClient(args.host, args.port, recv_kw=None, result_kw=KW_WATCH_HR, identity=IDENTITY)
    watch = WatchAdapter(hub, raw_audio=not args.no_raw_audio)
    try:
        await asyncio.gather(watch.serve(args.ws_host, args.ws_port), _status_loop(watch))
    finally:
        hub.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="와치를 comm_hub 에 붙인다")
    add_broker_args(ap)
    ap.add_argument("--ws-host", default="0.0.0.0", help="와치 WebSocket 수신 주소")
    ap.add_argument("--ws-port", type=int, default=8765, help="와치 WebSocket 포트 (와치 앱 기본값 8765)")
    ap.add_argument("--no-raw-audio", action="store_true",
                    help="WATCH_AUDIO 를 올리지 않는다. VAD 는 그대로 돌아 WATCH_VAD 만 나간다")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("종료", flush=True)


if __name__ == "__main__":
    main()
