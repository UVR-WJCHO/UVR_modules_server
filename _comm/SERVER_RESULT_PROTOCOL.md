# SERVER_RESULT 수신 프로토콜 (HL2 / Unity 구현용)

서버(`main_handtrack_comm.py`)가 손 추론 결과를 comm_hub(ZeroMQ ROUTER 브로커)를 통해
HL2로 되돌린다. HL2(Unity) 쪽에 이 결과를 **수신**하는 파트를 새로 구현해야 한다.
이 문서는 그 구현에 필요한 전송 형태/포맷을 정의한다.

> 요약: **ZeroMQ DEALER**로 브로커에 붙어서 keyword `SERVER_RESULT`를 구독하고,
> `NOTIFY -> DOWNLOAD -> DATA_REPLY` 핸드셰이크로 **protobuf `ServerResult`** 바이트를 받아 파싱한다.
> 메인 payload는 **hand pose(손 관절)**, gesture는 덤이다.

---

## 1. 전송 경로 (Transport)

- 브로커: `comm_hub.py` (ZeroMQ **ROUTER**), 기본 포트 **37001**. HL2는 브로커 PC의 LAN IP:37001로 접속.
- HL2 클라이언트 소켓: **ZeroMQ DEALER**, 고유 IDENTITY 설정 필수 (예: `"HL2_CLIENT"`).
- 모든 메시지는 multipart이며 **첫 프레임은 빈 delimiter(`""`)** 다 (DEALER 규약).
- 서버가 결과를 올릴 때 쓰는 식별자:
  - keyword = `"SERVER_RESULT"`
  - source(보낸 주체) = `"HANDTRACK"`

이미 HL2가 서버로 **보내는**(HL2DATA 업로드) 쪽은 구현돼 있고, 여기서 필요한 건 **받는** 쪽이다.

---

## 2. 구독 + 수신 핸드셰이크

comm_hub는 "저장 후 알림(pull)" 모델이다. 데이터가 알림(NOTIFY)에 바로 실려오지 않고,
알림을 받으면 클라이언트가 DOWNLOAD를 요청해서 실제 데이터(DATA_REPLY)를 받는다.

프레임 구성 (DEALER가 send/recv 하는 프레임. ROUTER가 앞단에 identity를 붙였다 떼므로
**클라이언트 관점에선 맨 앞 빈 프레임부터** 보면 된다):

**(1) 구독 등록 — 접속 직후 1회 전송**
```
send: ["", "RECV_REG", "SERVER_RESULT", <MY_IDENTITY>, "ALL"]
```
- 4번째 프레임(Source)은 본인 식별자, 5번째(Target)는 `"ALL"`.

**(2) 알림 수신 → 다운로드 요청**
```
recv: ["", "NOTIFY", "SERVER_RESULT", "HANDTRACK", <FID>]
send: ["", "DOWNLOAD", "SERVER_RESULT", "HANDTRACK", <FID>]
```
- `<FID>`는 프레임 ID 문자열(바이트). NOTIFY에서 받은 값을 그대로 DOWNLOAD에 echo.

**(3) 데이터 수신**
```
recv: ["", "DATA_REPLY", "SERVER_RESULT", "HANDTRACK", <FID>, <ServerResult 直렬화 bytes>]
```
- 마지막 프레임이 `ServerResult` protobuf 직렬화 바이트. 이걸 파싱하면 됨.

수신 루프 의사코드:
```
sock = DEALER; sock.Identity = "HL2_CLIENT"; sock.Connect("tcp://<broker>:37001")
send(["", "RECV_REG", "SERVER_RESULT", "HL2_CLIENT", "ALL"])
loop:
    msg = recv_multipart()
    action = msg[1]
    if action == "NOTIFY":      # msg = ["", NOTIFY, kw, src, fid]
        send(["", "DOWNLOAD", msg[2], msg[3], msg[4]])
    elif action == "DATA_REPLY": # msg = ["", DATA_REPLY, kw, src, fid, data]
        result = ServerResult.Parse(msg[5])
        // ... 사용 ...
```

> 참고: 서버는 최신 프레임만 유지(conflate)하며 매 입력 프레임마다 결과를 올린다.
> 처리 지연 시 오래된 결과는 서버 측에서 드롭될 수 있다(실시간 우선).

---

## 3. 데이터 포맷 — `ServerResult` (protobuf, proto3)

C# gencode는 아래 `.proto`로 생성(`protoc --csharp_out=. hl2_data.proto`).
전체 정의는 `_comm/hl2_data.proto`에 있으며, 결과 메시지만 발췌:

```proto
syntax = "proto3";

message ServerResult {
  float  timestamp = 1;              // 결과가 계산된 원본 프레임의 timestamp (echo)
  repeated float cam_to_world = 2;   // PV cam->world 4x4 (row-major, 16 floats) (echo)
  float  fx = 3;                     // PV intrinsics (echo)
  float  fy = 4;
  float  cx = 5;
  float  cy = 6;
  repeated float hand = 7;           // ★ 메인: absolute 3D 관절 (21*3 xyz, meters, PV 카메라 프레임)
  int32  gesture_idx = 8;            // 덤: -1 = 없음, 그 외 0~14 (아래 매핑)
}
```

### 필드 상세

| 필드 | 타입 | 길이 | 의미 |
|---|---|---|---|
| `timestamp` | float | 1 | 이 결과가 계산된 **원본 HL2DATA 프레임의 timestamp**를 그대로 echo. HL2가 결과를 "그 프레임" 좌표계/시점에 매칭하는 데 사용 |
| `cam_to_world` | float[] | 16 | 원본 프레임의 **PV 카메라→월드 4x4 행렬(row-major)** echo |
| `fx,fy,cx,cy` | float | 각 1 | 원본 프레임의 **PV intrinsics** echo |
| `hand` | float[] | 63 | **absolute 3D 관절 21개 × (x, y, z)** flatten, **meters, PV 카메라 프레임**. 메인 데이터 |
| `gesture_idx` | int32 | 1 | 제스처 클래스 인덱스. **-1 = 제스처 없음**. 덤(부가정보) |

### `hand` (21×3, xyz) 해석
- 21개 관절, 각 `(x, y, z)` **meters**, **PV 카메라 프레임** (OpenCV 축: +X 오른쪽, +Y 아래, +Z 앞).
  Unity는 좌표계가 다르니(+Y 위) **경계에서 변환** 필요.
- **서버에서 이미 lift 된 절대 좌표**다: aligned depth 에서 얻은 **실제 wrist depth** + RGB 모델의
  **root-relative z** 로 서버가 21관절을 절대 3D로 복원해서 보낸다. → 기기에서 root depth 복원(ray
  search) 불필요. (extra/ 의 `RecoverRootAlongRay` 는 이 경우 안 씀; 리프팅된 pose 를 바로 받음)
- 관절 순서(0~20, = DexYCB):
  ```
  0 wrist,
  1 thumb_mcp, 2 thumb_pip, 3 thumb_dip, 4 thumb_tip,
  5 index_mcp, 6 index_pip, 7 index_dip, 8 index_tip,
  9 middle_mcp, 10 middle_pip, 11 middle_dip, 12 middle_tip,
  13 ring_mcp, 14 ring_pip, 15 ring_dip, 16 ring_tip,
  17 little_mcp, 18 little_pip, 19 little_dip, 20 little_tip
  ```

### `gesture_idx` 매핑 (덤)
```
0 CClock_index   1 CClock_thumb   2 Clock_index    3 Clock_thumb
4 Down_index     5 Down_thumb     6 Left_index     7 Left_thumb
8 Natural        9 Right_index   10 Right_thumb   11 Tap_index
12 Tap_thumb    13 Up_index      14 Up_thumb
```
- `-1`이면 이번 프레임엔 확정 제스처 없음. 서버에서 gesture 인식을 끄면 항상 -1.

---

## 4. 주의/한계 (구현 시 참고)

1. **hand 는 PV 카메라 프레임의 절대 3D(meters, OpenCV 축)** 다. Unity(+Y 위, 좌手계)로 가져올 때
   **경계에서 축 변환** 필요. depth 기반 보정(extra/)은 depth(AHAT) 카메라 프레임을 쓰므로,
   기기에서 **PV→AHAT extrinsic**(정적, 캡처 시점 near-simultaneous)으로 회전시켜 넣어야 한다.
2. **유효하지 않은 프레임**(손 미검출 또는 wrist depth hole 로 lift 실패): 현재 `hand`에
   더미값(21×3 전부 1.0) + `gesture_idx=-1`을 보낸다. **"유효" 명시 플래그가 없다** — 필요하면
   서버에 `hand_valid` bool 추가 요청 권장(전부 1.0 인지로 임시 판별 가능하나 취약).
3. **metric scale**: root-relative z 는 monocular(WILOR/MANO) 추정이라 절대 스케일이 근사일 수
   있다. wrist depth 는 실제 센서값이라 정확하지만, 손가락 등 상대 구조 스케일은 검증 필요.
4. `gesture_idx`는 쿨다운(0.5s)이 적용돼 확정 제스처가 떴을 때만 잠깐 값이 실리고 대부분 -1이다.
   메인 목적은 `hand`이므로 gesture는 옵션으로 취급.

---

## 5. C# 힌트

- ZeroMQ: Unity에선 **NetMQ**(DealerSocket) 사용이 일반적.
- protobuf: `protoc --csharp_out=<dir> hl2_data.proto` 로 `ServerResult` C# 클래스 생성 후
  `ServerResult.Parser.ParseFrom(bytes)`.
- IDENTITY는 HL2DATA를 보낼 때 쓰는 clientIdentity와 **다른 값**으로 두는 게 안전(같은 브로커에
  송신/수신 소켓을 따로 붙일 경우 identity 충돌 방지).
