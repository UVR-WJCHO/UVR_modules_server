"""폰 앱(RoverHandoff)의 TCP 스트림을 comm_hub 와 양방향으로 잇는다.

원래 폰은 HL2 앱 A 에 직접 TCP 로 붙었다(HL2 가 서버). HL2 앱 A 는 쓰지 않고 그 handoff
기능은 hl2앱 B 가 comm_hub 로 받는다. 폰 앱은 재빌드 때 서버 IP 입력만 추가되므로, 여기서
HL2 앱 A 가 열던 것과 같은 TCP 서버를 열어 폰을 받는다.

    폰  --TCP 8777-->  여기  --PHONE_TO_HL2-->  hl2앱 B
    폰  <--TCP 8777--  여기  <--HL2_TO_PHONE--  hl2앱 B

프레임: 4 바이트 big-endian 길이(본문만) + UTF-8 JSON. 본문은 NetMessage
{"type": "...", "payload": "..."} 다. 여기서는 열어보지 않고 바이트 그대로 양쪽에 넘긴다.
PartUpdate 는 텍스처 PNG 가 들어 50~300 KB 다.
명세: temp/DeviceCommunication/docs/PROTOCOL.md (§3 프레이밍, §4 메시지)

폰은 한 대다. 원래 HL2 도 한 연결만 받았다. 새 연결이 오면 이전 것을 닫는다.
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading

from hub_client import HubClient, KW_PHONE_TO_HL2

MAX_FRAME = 32 * 1024 * 1024        # PROTOCOL.md §3. 넘으면 스트림이 깨진 것으로 보고 끊는다


def _msg_type(body: bytes) -> str:
    try:
        return json.loads(body).get("type", "?")
    except (ValueError, AttributeError):
        return "?"


class PhoneAdapter:
    name = "phone"

    def __init__(self, hub: HubClient):
        """hub 는 HL2_TO_PHONE 을 구독하고 PHONE_TO_HL2 로 올리는 클라이언트."""
        self.hub = hub
        self._writer: asyncio.StreamWriter | None = None
        self._peer = "-"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._n = {"up": 0, "down": 0, "down_dropped": 0}
        self._last = {"up": "-", "down": "-"}

    async def serve(self, host: str, port: int) -> None:
        self._loop = asyncio.get_running_loop()
        threading.Thread(target=self._relay_down, daemon=True).start()
        server = await asyncio.start_server(self._handle, host, port)
        print(f"[{self.name}] tcp://{host}:{port} 대기. 폰 앱의 서버 IP 를 이 PC 로", flush=True)
        async with server:
            await server.serve_forever()

    # --- 폰 -> comm_hub --------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = "%s:%d" % writer.get_extra_info("peername")[:2]
        if self._writer is not None:
            print(f"[{self.name}] 새 연결 {peer}. 이전 {self._peer} 닫음", flush=True)
            self._writer.close()
        self._writer, self._peer = writer, peer
        print(f"[{self.name}] 연결 {peer}", flush=True)
        try:
            while True:
                n = struct.unpack(">I", await reader.readexactly(4))[0]
                if not 0 < n <= MAX_FRAME:
                    print(f"[{self.name}] 길이 {n} 비정상. 끊는다", flush=True)
                    break
                body = await reader.readexactly(n)
                self.hub.send(body, KW_PHONE_TO_HL2)
                self._n["up"] += 1
                self._last["up"] = _msg_type(body)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if self._writer is writer:
                self._writer = None
            writer.close()
            print(f"[{self.name}] 해제 {peer}", flush=True)

    # --- comm_hub -> 폰 --------------------------------------------------
    def _relay_down(self) -> None:
        """HubClient 의 수신 큐를 비워 이벤트 루프로 넘긴다. 소켓은 루프 스레드만 만진다."""
        while True:
            data = self.hub.get_latest(timeout=0.5)
            if data is not None:
                asyncio.run_coroutine_threadsafe(self._to_phone(data), self._loop)

    async def _to_phone(self, body: bytes) -> None:
        w = self._writer
        if w is None:
            self._n["down_dropped"] += 1        # 폰이 없으면 버린다. 원래도 큐가 없었다
            return
        try:
            w.write(struct.pack(">I", len(body)) + body)
            await w.drain()
            self._n["down"] += 1
            self._last["down"] = _msg_type(body)
        except ConnectionError:
            pass

    def status(self, elapsed: float) -> str:
        if self._writer is None:
            return f"  {self.name}  연결 없음  (hl2->폰 버림 {self._n['down_dropped']})"
        return (f"  {self.name}  conn={self._peer}  폰->hl2 {self._n['up']} (last {self._last['up']})  "
                f"hl2->폰 {self._n['down']} (last {self._last['down']})")
