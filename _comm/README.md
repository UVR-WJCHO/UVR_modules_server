# _comm — HL2 ↔ Server 통신

HoloLens2(전송) ↔ 서버(수신) 간 ZeroMQ 통신용 protobuf 정의. 원래
`WiseUIAppUnity/Assets/Scripts/Server/` 아래에 있던 것을 서버 코드로 이전한 것.

실행 스크립트(`comm_hub.py`, `main_all_hl2_receiver.py`)는 다른 `main_*.py` 와 함께
**저장소 최상위**에 있다. 여기 `_comm/` 에는 protobuf 정의와, 진입점이 공통으로 쓰는
브로커 클라이언트(`hub_client.py`)를 둔다.

## 구성

| 위치 | 파일 | 역할 |
|---|---|---|
| `WiseUIServer/comm_hub.py` | 통신 서버 | ZeroMQ ROUTER 중앙 허브 (`RobustCentralHub`). UPLOAD/DOWNLOAD/NOTIFY 라우팅 |
| `WiseUIServer/main_all_hl2_receiver.py` | 수신 뷰어 | 수신 클라이언트 (DEALER). RGB(JPEG) + Depth(16bit PNG) + 센서 정보 파싱·시각화 |
| `_comm/hl2_data.proto` | 정의 | `HL2SensorPacket` (42필드, depth 포함). C# 전송 측 `HL2Data/Hl2Data.cs` 기준 재구성 |
| `_comm/hl2_data_pb2.py` | 생성물 | `hl2_data.proto` 컴파일 결과. `main_all_hl2_receiver.py` 가 sys.path 로 import |
| `_comm/hl2_forecast.proto` | 정의 | `HandForecast`. horizon grid 결과 (`main_handtrack_forecast.py`) |
| `_comm/hl2_forecast_pb2.py` | 생성물 | `hl2_forecast.proto` 컴파일 결과 |
| `_comm/hub_client.py` | 공용 | `HubClient`. 모든 진입점이 이 하나로 브로커에 붙는다 |

## 브로커 클라이언트

진입점은 zmq 를 직접 열지 않고 `hub_client.HubClient` 를 쓴다. 예전에는 같은
클래스가 `main_handtrack.py`, `main_handtrack_forecast.py`, `main_meshrecon.py` 에
각각 들어 있었고 이미 서로 달라지고 있었다.

```python
sys.path.insert(0, ".../_comm")
from hub_client import HubClient, add_broker_args, KW_SERVER_RESULT

add_broker_args(ap)                       # --host / --port 를 같은 이름으로 추가
hub = HubClient(args.host, args.port,     # recv_kw 기본값은 KW_HL2DATA
                result_kw=KW_SERVER_RESULT, identity=b"HANDTRACK")

data = hub.get_latest(timeout=1.0)        # 최신 한 프레임만 (conflate)
hub.send(payload)                         # result_kw 로 UPLOAD
hub.close()
```

`result_kw` 를 주지 않으면 읽기 전용 클라이언트가 된다. 수신은 백그라운드
스레드가 돌며 처리가 느리면 오래된 프레임을 버린다(`n_arrived` / `n_dropped` 로
확인). 송신은 별도 소켓을 써서 두 스레드가 같은 소켓을 만지지 않는다.

| 상수 | 값 | 쓰는 곳 |
|---|---|---|
| `KW_HL2DATA` | `HL2DATA` | HL2 -> 서버. 모든 진입점이 구독 |
| `KW_SERVER_RESULT` | `SERVER_RESULT` | `main_handtrack.py` 결과 |
| `KW_HAND_FORECAST` | `HAND_FORECAST` | `main_handtrack_forecast.py` 결과 |
| `KW_MESH_RESULT` | `MESH_RESULT` | `main_meshrecon.py` 결과 |

## 환경

GPU 알고리즘 진입점(`main_handtrack.py`,
`main_meshrecon.py`)은 `uvr_integ` 환경에서 실행한다. 브로커,
뷰어, 녹화/재생처럼 GPU 모델을 불러오지 않는 통신 도구는 가벼운
전용 환경 `wiseui_commu`를 사용해도 된다.

```bash
conda create -n wiseui_commu python=3.11
conda activate wiseui_commu
pip install "protobuf>=7.34.1" pyzmq opencv-python numpy
```

## 실행

```bash
conda activate wiseui_commu
cd WiseUIServer
python comm_hub.py --port 37001         # 터미널 1: 통신 서버(중앙 허브)
python main_all_hl2_receiver.py                   # 터미널 2: RGB/Depth 뷰어
#   python main_all_hl2_receiver.py --no-gui      # 콘솔만 (SSH/headless)
#   python main_all_hl2_receiver.py --host <IP>   # 허브가 다른 PC일 때
#   python main_all_hl2_receiver.py --save out    # 프레임 저장(depth는 16bit 원본 보존)
```
그다음 HoloLens2 앱을 붙이면 실제 depth까지 수신 확인 가능.

## proto 재생성

```bash
cd WiseUIServer/_comm
protoc --python_out=. hl2_data.proto
```

## 주의

- `hl2_data.proto` 의 필드 번호는 C# 전송 측(`Hl2Data.cs`)과 반드시 일치해야 한다.
- `HL2SensorPacket` 메시지명이 구버전 pb2와 같아, 한 프로세스에서 둘을 동시에 import하면
  protobuf descriptor pool 충돌이 난다. 하나로 통일해서 쓸 것.
