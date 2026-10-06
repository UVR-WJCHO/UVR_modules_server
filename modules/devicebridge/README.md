# devicebridge — ZMQ 를 못 쓰는 기기를 comm_hub 에 붙인다

HL2 앱은 comm_hub 에 직접 붙는다. 와치와 폰은 각자 프로토콜밖에 모르고 앱도 바꾸지 않기로
했다. 그래서 기기별 프로토콜을 받아 comm_hub 로 올리는 어댑터를 `main_devicebridge.py`
프로세스 하나에 모은다. 기기가 늘어도 프로세스는 늘지 않는다.

| 어댑터 | 상태 | 받는 방식 | 올리는 keyword |
|---|---|---|---|
| `watch.py` | 동작 | WebSocket 서버 (기본 8765) | `WATCH_HR` `WATCH_ACTIVITY` `WATCH_AUDIO` `WATCH_VAD` |
| 폰 (RoverHandoff) | 예정 | TCP 서버 (8777), 4 B 길이 프레이밍 | `PHONE_TO_HL2` / `HL2_TO_PHONE` |

## 실행

```bash
python comm_hub.py
python main_devicebridge.py                 # --ws-port 8765 기본
python main_devicebridge.py --no-raw-audio  # WATCH_AUDIO 는 안 올리고 WATCH_VAD 만
```

와치 앱 화면에서 서버 IP 를 이 PC 로 바꾼다. 포트는 앱 기본값 8765 그대로다.

5 초마다 한 줄 찍는다. 종류별 수신율과 VAD 상태로 실제로 흐르는지 본다.

```
  watch  conn=1  hr 1.0/s  act 1  audio 49.0/s  vad=speaking(4 changes)  raw_audio=on
```

## 와치 프로토콜 (WatchSensor 앱)

와치 -> 서버 한 방향. 전부 JSON 텍스트 프레임이다.

| type | 필드 | 주기 |
|---|---|---|
| `watch_hello` | 없음 | 연결 직후 1 회. 이걸 보낸 소켓만 와치로 본다 |
| `heart_rate` | `bpm` int, `status` (항상 0), `timestamp` | 연속 (Health Services) |
| `activity` | `state` STILL/WALKING/RUNNING/IN_VEHICLE/ON_BICYCLE/UNKNOWN, `entering` bool, `timestamp` | 전이 때만 |
| `audio` | `pcm_b64` base64, `timestamp` | 초당 50 개 |

- 오디오: 16 kHz, int16, mono, 20 ms = 640 bytes. 메시지당 약 900 B, 약 45 KB/s.
- `timestamp` 는 Unix epoch 밀리초. HL2 의 `Time.time`(앱 기동 후 초)과 축이 다르다.
- 서버 -> 와치 방향은 없다. 앱이 받은 메시지를 버린다. 켜고 끄는 것은 와치 화면에서 한다.

## comm_hub 에 올라가는 것

받은 JSON 바이트를 그대로 올린다. 구독자는 원래 `server.py` 가 Unity 에 주던 것과 같은
형식을 받는다.

- `WATCH_HR`, `WATCH_ACTIVITY`, `WATCH_AUDIO` — 와치 메시지 원문
- `WATCH_VAD` — `{"type":"vad","is_speaking":bool,"timestamp":ms}`. webrtcvad(aggressiveness 2)
  결과가 **바뀔 때만** 올린다. 연결마다 따로 추적하므로 와치가 둘이어도 섞이지 않는다.

`--no-raw-audio` 는 `WATCH_AUDIO` 만 끈다. VAD 는 그대로 돈다.

## 구독 예

```python
from hub_client import HubClient, KW_WATCH_VAD
c = HubClient(host, port, recv_kw=KW_WATCH_VAD, identity=b"MYMODULE_VAD")
msg = json.loads(c.get_latest())
```

comm_hub 는 source 당 수신 등록을 하나만 기억하므로 keyword 마다 identity 를 다르게 한다.
`WATCH_AUDIO` 처럼 연속인 스트림은 `queue_size` 를 크게 잡아야 청크가 빠지지 않는다.

## 의존성

`websockets`, `webrtcvad` (`pip install webrtcvad`).
