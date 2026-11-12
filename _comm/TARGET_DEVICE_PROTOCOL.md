# 기기 지정 — HL2 여러 대가 한 comm_hub 를 쓸 때 (Unity 구현용)

## 0. 배경

- comm_hub 하나에 HL2 여러 대가 붙는다. 기기마다 `HL2Client.clientIdentity` 가 다르다
  (예: `HoloLens2_A`, `HoloLens2_B`, `HoloLens2_C`). 어떤 ID 가 몇 대 붙을지는 정해져 있지 않다.
- 서버 기능은 기기별로 나뉜다. 어느 기기인지는 서버를 실행할 때 정한다.
  - HQS 상태·패널, 손 gesture → 한 기기 (예: `HoloLens2_B`)
  - 폰 연동: HL2 → 폰(`HL2_TO_PHONE`)은 한 기기 (예: `HoloLens2_C`)가 보낸 것만 폰에 간다.
    폰 → HL2(`PHONE_TO_HL2`)는 모든 HL2 가 받는다
- 서버 출력 keyword 는 지금처럼 그것을 구독한 모든 HL2 에 간다. 그래서 서버가 출력 본문에 받을 기기를
  `target_device_id` 로 넣고, 각 HL2 는 자기 것만 반영한다. 필드 이름은 `OBJECT_STATE` 의
  `target_device_id` 와 같다.

서버는 아래 새 형식으로 보낸다. 넣는 값은 서버 실행 때 준 `--device` 이고, 주지 않으면 빈 문자열이다.
`HL2_MODE` 는 형식 자체가 바뀌므로 앱을 고치기 전에는 읽지 못한다.

## 1. 받는 쪽 규칙 (모든 서버 출력에 공통)

1. 구독은 지금처럼 앱 기능 기준으로 한다. 기기 ID 로 keyword 를 나누지 않는다.
2. `target_device_id` 가 자기 `clientIdentity` 와 **같으면 반영**, 다르면 버린다. 완전 일치, 대소문자 구분.
3. `target_device_id` 가 **없거나 빈 문자열이면 지금처럼 반영**한다. 서버 쪽 변경 전 기간과, 아직 이
   필드를 넣지 않는 서버 출력(4장)을 위해서다.

## 2. keyword 별 변경 (전 → 후)

### 2.1 `HL2_MODE` — 형식이 바뀐다

| | 본문 |
|---|---|
| 전 | UTF-8 문자열 `FOCUS` |
| 후 | UTF-8 JSON `{"target_device_id":"HoloLens2_B","mode":"FOCUS"}` |

- `mode` 값(`FOCUS` / `MOVING` / `TALKING` / `NONE`)과 전송 규칙(1 초마다, 바뀌면 즉시, 이어지면 5 초마다,
  나머지 `NONE`)은 그대로다. [HQS_PROTOCOL.md](HQS_PROTOCOL.md) 2장.
- 본문이 `{` 로 시작하지 않으면 옛 형식이다. 본문 전체를 `mode` 로 보고 반영한다.

```csharp
[Serializable] class Hl2ModeMsg { public string target_device_id; public string mode; }
```

### 2.2 `HL2_UI` — 필드 하나 추가

| | 본문 |
|---|---|
| 전 | `{"type":"social","source":"combined","value":0.62,"timestamp":1730000002.341,"highlight":true}` |
| 후 | `{"target_device_id":"HoloLens2_B","type":"social","source":"combined","value":0.62,"timestamp":1730000002.341,"highlight":true}` |

- 나머지 필드는 그대로다. 받는 클래스에 `public string target_device_id;` 를 더한다.

### 2.3 `SERVER_RESULT` (protobuf `ServerResult`) — 필드 하나 추가

| | 정의 (`_comm/hl2_data.proto`) |
|---|---|
| 전 | 필드 1~8 (`timestamp` … `gesture_idx`) |
| 후 | `string target_device_id = 9;` 추가 |

- proto 를 갱신한 뒤 C# 클래스를 다시 생성한다. 다시 생성하기 전의 클래스도 모르는 필드를 무시하므로
  파싱은 된다(걸러내지는 못한다).

### 2.4 `HL2_CONTROL` (protobuf `HL2Control`) — 필드 하나 추가

| | 정의 (`_comm/hl2_sensors.proto`) |
|---|---|
| 전 | `send_audio = 1`, `send_imu = 2`, `imu_rate_divisor = 3`, `send_render = 4` |
| 후 | `string target_device_id = 5;` 추가 |

- HQS 가 5 초마다 보내는 마이크 요청이다. 대상이 아니면 **메시지 전체를 무시**한다. `HL2Control` 은 켤
  스트림 전체를 한 번에 정하므로, 다른 기기 것을 반영하면 자기 스트림 상태가 바뀐다.

### 2.5 `PHONE_TO_HL2` — 변경 없음

- 폰 메시지는 모든 HL2 가 받아야 하므로 `target_device_id` 를 넣지 않는다. 본문은 폰이 보낸 NetMessage
  `{"type":"PartUpdate","payload":"..."}` 그대로다. 1장 규칙 3 에 따라 받은 HL2 는 모두 반영한다.

## 3. HL2 가 올리는 것 — Source 프레임 (형식 변경 없음, 확인 요청)

서버는 HL2 가 올린 메시지의 Source 프레임으로 어느 기기 것인지 가린다. 서버 실행 때 지정한 기기와
Source 가 같은 메시지만 처리하고 나머지는 받지 않는다.

```
["", "UPLOAD", <KEYWORD>, <SOURCE>, <FID>, <DATA>]
                           ^^^^^^^^ 여기에 clientIdentity
```

| keyword | 서버 쪽 처리 | Source 에 있어야 하는 값 |
|---|---|---|
| `HL2DATA` | HQS, 손 추적 | `clientIdentity` (HL2_SCENE_PROTOCOL.md 에 이미 그렇다고 되어 있다) |
| `HL2_AUDIO` | HQS 말소리 | `clientIdentity` |
| `HL2_TO_PHONE` | 폰에 넘김 | `clientIdentity` |

**확인됨 (2026-11-12, HL2 앱 쪽 답):** `HL2DATA`, `HL2_AUDIO`, `HL2_TO_PHONE`, `HL2_RENDER` 모두 Source 프레임에
`clientIdentity` 를 그대로 넣는다. `_AUDIO`, `_PHONE`, `_RENDER` 접미사는 소켓 identity 에만 붙는다.

## 4. 이번 변경에 없는 서버 출력

`PHONE_TO_HL2`(2.5), `HAND_FORECAST`, `MESH_RESULT`, 그리고 `main_adaptiveMM` 이 보내는 `HL2_CONTROL` 은
`target_device_id` 를 넣지 않는다. 1장 규칙 3 에 따라 지금처럼 모든 기기가 반영한다.
