# _comm — HL2 ↔ Server 통신

HoloLens2(전송) ↔ 서버(수신) 간 ZeroMQ 통신용 protobuf 정의. 원래
`WiseUIAppUnity/Assets/Scripts/Server/` 아래에 있던 것을 서버 코드로 이전한 것.

실행 스크립트(`comm_hub.py`, `hl2_receiver.py`)는 다른 `main_*.py` 와 함께
**WiseUIServer 최상위**에 있고, 여기 `_comm/` 에는 protobuf 정의만 둔다.

## 구성

| 위치 | 파일 | 역할 |
|---|---|---|
| `WiseUIServer/comm_hub.py` | 통신 서버 | ZeroMQ ROUTER 중앙 허브 (`RobustCentralHub`). UPLOAD/DOWNLOAD/NOTIFY 라우팅 |
| `WiseUIServer/hl2_receiver.py` | 수신 서버 | 수신 클라이언트 (DEALER). RGB(JPEG) + Depth(16bit PNG) + 센서 정보 파싱·시각화 |
| `_comm/hl2_data.proto` | 정의 | `HL2SensorPacket` (42필드, depth 포함). C# 전송 측 `HL2Data/Hl2Data.cs` 기준 재구성 |
| `_comm/hl2_data_pb2.py` | 생성물 | `hl2_data.proto` 컴파일 결과. `hl2_receiver.py` 가 sys.path 로 import |

## 환경

프로토콜 gencode가 protobuf 런타임 major 7을 요구하므로 전용 env `wiseui_commu` 사용
(`main_handtrack.py` 의 `uvr_integ` 와 분리 — mediapipe/onnx 등이 낮은 protobuf 에 의존).

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
python hl2_receiver.py                   # 터미널 2: 수신 서버 (RGB/Depth 창 + 콘솔)
#   python hl2_receiver.py --no-gui      # 콘솔만 (SSH/headless)
#   python hl2_receiver.py --host <IP>   # 허브가 다른 PC일 때
#   python hl2_receiver.py --save out    # 프레임 저장(depth는 16bit 원본 보존)
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
