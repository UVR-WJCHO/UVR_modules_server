# HL2 녹화 / 재생 사용법

`record_hl2.py`, `play_hl2.py` — 홀로렌즈가 브로커로 보내는 `HL2DATA`(이미지 + 머리/손/시선 + `cam_to_world` + intrinsics + **depth**)를 파일로 녹화하고, 기기 없이 다시 재생하는 도구.

- 위치: 이 두 스크립트는 `WiseUIServer/tests/` 에 있고, 브로커/수신기는 `WiseUIServer/` 최상위에 있다.
- 브로커(`comm_hub.py`)가 먼저 떠 있어야 한다.
- 파일 포맷(한 레코드): `<d I I>`(상대초, fid 길이, data 길이) + fid + protobuf data.
- 기본 접속: `--host 143.248.96.81 --port 37001`.
- 환경: 통신 스크립트 전용 conda env `wiseui_commu` 사용 (`protobuf>=7.34.1 pyzmq opencv-python numpy`).
  녹화/재생은 raw bytes 를 다루므로 protobuf 없이도 동작하지만, 시작 시 `_comm/hl2_data_pb2` 로
  첫 프레임을 파싱해 **protobuf 요약(해상도·depth·손·intrinsics)** 을 출력한다.

---

## 녹화 — `record_hl2.py`

브로커에서 `HL2DATA`를 구독해 프레임마다 파일에 바로 기록(스트리밍)한다. 중간에 크래시 나도 그때까지는 보존된다.

```bash
conda activate wiseui_commu
cd WiseUIServer/tests
python record_hl2.py                       # rec.hl2rec 로 저장, 브로커 기본값
python record_hl2.py walk1.hl2rec --host 143.248.96.81 --port 37001
```

실행하면 **녹화 ON** 으로 시작. 첫 프레임에서 `[rec] protobuf 확인: ...` 로 유효성/depth 유무를 보여준다. 콘솔 조작:

- **Enter (아무 글자 + Enter)** = 녹화 ON/OFF 토글
- **q** = 저장하고 종료

인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `출력파일` | `rec.hl2rec` | 첫 번째 비옵션 인자 = 저장 파일명 |
| `--host` | `143.248.96.81` | 브로커 IP |
| `--port` | `37001` | 브로커 포트 |

---

## 재생 — `play_hl2.py`

녹화 파일을 읽어 `UPLOAD HL2DATA`로 브로커에 전송한다. **원래 프레임 타이밍을 보존**하므로 홀로렌즈가 실제로 보내는 것처럼 동작한다. `fid`는 재생 시 새로 매기므로 `--loop` 반복해도 충돌 없음. 시작 시 파일의 총 프레임 수와 첫 프레임 protobuf 요약을 출력한다.

```bash
conda activate wiseui_commu
cd WiseUIServer/tests
python play_hl2.py rec.hl2rec
python play_hl2.py walk1.hl2rec --host 143.248.96.81 --port 37001 --loop --speed 1.0 --id HoloLens2_User
```

인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `입력파일` | (필수) | 재생할 `.hl2rec` 파일 |
| `--host` | `143.248.96.81` | 브로커 IP |
| `--port` | `37001` | 브로커 포트 |
| `--loop` | off | 끝나면 처음부터 반복 |
| `--speed` | `1.0` | 재생 속도 (2.0 = 2배속) |
| `--id` | `HoloLens2_User` | 기기 식별자. HL2Client 의 `clientIdentity`와 같게 두면 진짜 기기처럼 동작 |

**Ctrl+C** 로 중단.

---

## 전형적인 흐름

경로 기준: 브로커/수신기는 `WiseUIServer/`, 녹화/재생은 `WiseUIServer/tests/`.

1. 브로커 실행 (`WiseUIServer/`): `python comm_hub.py --port 37001`
2. 홀로렌즈 접속 상태에서 녹화 (`WiseUIServer/tests/`): `python record_hl2.py walk1.hl2rec` → 움직이며 캡처 → `q`
3. 나중에 기기 없이 재생 (`WiseUIServer/tests/`): `python play_hl2.py walk1.hl2rec --loop`
4. 재생 확인용 뷰어 (`WiseUIServer/`): `python main_all_hl2_receiver.py --host <브로커IP> --port 37001`
