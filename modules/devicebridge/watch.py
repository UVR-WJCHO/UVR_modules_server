"""와치 앱(WatchSensor)의 WebSocket 스트림을 comm_hub 로 올린다.

와치 앱은 WebSocket+JSON 만 말하고 재빌드하지 않기로 했다. 그래서 그 앱이 원래 붙던
server.py 와 똑같이 받아준다. 와치 화면에서 서버 IP 만 이 PC 로 바꾸면 앱은 차이를 모른다.

와치 -> 여기. 전부 JSON 텍스트 프레임 하나씩이다.

    {"type":"watch_hello"}                                        연결 직후 1 회
    {"type":"heart_rate","bpm":72,"status":0,"timestamp":ms}      연속
    {"type":"activity","state":"WALKING","entering":true,"timestamp":ms}   전이 때만
    {"type":"audio","pcm_b64":"...","timestamp":ms}               16 kHz int16 mono 20 ms, 초당 50 개

timestamp 는 Unix epoch 밀리초다. HL2 의 Time.time(앱 기동 후 초)과 축이 다르다.
서버 -> 와치 방향은 없다. 앱이 받은 메시지를 버린다.

comm_hub 로는 받은 JSON 바이트를 그대로 올린다. 구독자는 server.py 가 Unity 에 주던 것과
같은 형식을 받는다.

    WATCH_HR        heart_rate 원문
    WATCH_ACTIVITY  activity 원문
    WATCH_AUDIO     audio 원문. raw_audio=False 면 올리지 않는다
    WATCH_VAD       {"type":"vad","is_speaking":bool,"timestamp":ms}. 값이 바뀔 때만
"""
from __future__ import annotations

import asyncio
import base64
import json

import websockets

from hub_client import HubClient, KW_WATCH_ACTIVITY, KW_WATCH_AUDIO, KW_WATCH_HR, KW_WATCH_VAD

from .vad import VadProcessor


class WatchAdapter:
    name = "watch"

    def __init__(self, hub: HubClient, raw_audio: bool = True, vad_aggressiveness: int = 2):
        self.hub = hub
        self.raw_audio = raw_audio
        self.vad = VadProcessor(vad_aggressiveness)
        self.connections = 0
        self.speaking: bool | None = None       # 마지막으로 올린 VAD 상태. 상태 표시용
        self._n = {"heart_rate": 0, "activity": 0, "audio": 0, "vad": 0}
        self._prev = dict(self._n)

    async def serve(self, host: str, port: int) -> None:
        async with websockets.serve(self._handle, host, port, ping_interval=20, ping_timeout=10,
                                    max_size=2 * 1024 * 1024, compression=None):
            print(f"[{self.name}] ws://{host}:{port} 대기. 와치 앱의 서버 IP 를 이 PC 로", flush=True)
            await asyncio.Future()

    async def _handle(self, ws) -> None:
        peer = "%s:%d" % ws.remote_address[:2]
        is_watch = False
        speaking: bool | None = None             # 연결마다 따로 둔다. 와치가 둘이어도 섞이지 않는다
        self.connections += 1
        print(f"[{self.name}] 연결 {peer}", flush=True)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                t = msg.get("type")
                if t == "watch_hello":
                    is_watch = True
                    continue
                if not is_watch:
                    continue                      # 와치만 받는다. 다른 클라이언트는 comm_hub 로 오면 된다
                data = raw.encode() if isinstance(raw, str) else raw

                if t == "heart_rate":
                    self.hub.send(data, KW_WATCH_HR)
                elif t == "activity":
                    self.hub.send(data, KW_WATCH_ACTIVITY)
                elif t == "audio":
                    if self.raw_audio:
                        self.hub.send(data, KW_WATCH_AUDIO)
                    # webrtcvad 는 640 bytes 에 수십 µs 다. 이벤트 루프에서 바로 돌려도 된다.
                    now = self.vad.is_speech(base64.b64decode(msg.get("pcm_b64", "")))
                    if now != speaking:          # 바뀔 때만 올린다. server.py 와 같다
                        speaking = self.speaking = now
                        self.hub.send(json.dumps({"type": "vad", "is_speaking": now,
                                                  "timestamp": msg.get("timestamp", 0)}).encode(),
                                      KW_WATCH_VAD)
                        self._n["vad"] += 1
                else:
                    continue
                self._n[t] += 1
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections -= 1
            print(f"[{self.name}] 해제 {peer}", flush=True)

    def status(self, elapsed: float) -> str:
        """5 초마다 찍는 한 줄. 받은 것이 실제로 흐르는지 종류별로 본다."""
        d = {k: (self._n[k] - self._prev[k]) / elapsed for k in self._n}
        self._prev = dict(self._n)
        if self.connections == 0:
            return f"  {self.name}  연결 없음"
        vad = "-" if self.speaking is None else ("speaking" if self.speaking else "silent")
        return (f"  {self.name}  conn={self.connections}  hr {d['heart_rate']:.1f}/s  "
                f"act {self._n['activity']}  audio {d['audio']:.1f}/s  vad={vad}"
                f"({self._n['vad']} changes)  raw_audio={'on' if self.raw_audio else 'off'}")
