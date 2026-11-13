# HQS → HL2 프로토콜 (Unity 구현용)

서버의 `main_all_hl2_receiver.py --hqs` 가 HL2 센서(HL2DATA, HL2_AUDIO), 와치(WATCH_*), 폰(PHONE_ACTIVITY)으로 사용자 상태를
계산해 comm_hub 로 두 종류를 올린다. HL2 앱은 이 둘을 받기만 하면 된다.

| keyword | 본문 | 용도 |
|---|---|---|
| `HL2_MODE` | UTF-8 JSON `{"target_device_id","mode"}`. mode 는 `FOCUS` / `MOVING` / `TALKING` / `NONE` | 모드 전환 창 |
| `HL2_UI` | UTF-8 JSON `{"target_device_id","type","source","value","timestamp","highlight"}` | HQS 패널(Emotional / Physical / Social / Digital) 값과 영역 하이라이트 |

서버는 HL2 기기(한 사용자)마다 그 기기에서 온 데이터로 따로 계산하고, 메시지마다 `target_device_id` 에 받을 기기의
`clientIdentity` 를 넣는다. 같은 broker 에 HL2 가 여러 대 붙어 있으면 모든 기기 앞으로 가는 메시지가 같은 keyword 로
섞여 오므로, `target_device_id` 가 자기 것과 같은 메시지만 반영한다(비어 있으면 모두 반영). [TARGET_DEVICE_PROTOCOL.md](TARGET_DEVICE_PROTOCOL.md)
와치 값과 폰 터치(digital)는 서버에서 와치·폰과 짝으로 정한 기기에만 간다. 다른 기기는 와치·폰이 없을 때처럼 받는다(3장).

## 1. 수신

comm_hub: ZeroMQ ROUTER, `tcp://<서버 IP>:37001`. HL2 는 DEALER 로 붙는다.

- 이미 쓰고 있는 수신 소켓(예: `HL2_RESULT_CLIENT`)의 RECV_REG keyword 목록에 `HL2_MODE`, `HL2_UI` 를 더한다.
  broker 는 identity 하나당 등록 하나만 기억하므로, 같은 identity 로 RECV_REG 를 따로 한 번 더 보내면 앞의 목록이 지워진다.
- NOTIFY 에는 본문이 없다. 받은 keyword / source / FID 를 그대로 DOWNLOAD 에 넣어 보내면 DATA_REPLY 로 본문이 온다.
- source 값은 고정으로 비교하지 않는다(서버 실행 옵션에 따라 바뀐다).
- 수신은 백그라운드 스레드, 창·UI 갱신은 Unity 메인 스레드에서 한다.

```
send: ["", "RECV_REG", "SERVER_RESULT", "HL2_MODE", "HL2_UI", <identity>, "ALL"]
recv: ["", "NOTIFY", <keyword>, <source>, <FID>]
send: ["", "DOWNLOAD", <keyword>, <source>, <FID>]
recv: ["", "DATA_REPLY", <keyword>, <source>, <FID>, <payload>]
```

위 RECV_REG 의 `SERVER_RESULT` 는 예시다. 실제로는 앱이 지금 등록하고 있는 keyword 들을 그대로 두고 두 개를 더한다.
DATA_REPLY 의 keyword(프레임 2)로 `HL2_MODE` / `HL2_UI` / 기존 keyword 를 가른다.

## 2. HL2_MODE

본문: UTF-8 JSON 한 개. `mode` 는 `FOCUS`, `MOVING`, `TALKING`, `NONE`.

```json
{"target_device_id": "HoloLens2_B", "mode": "FOCUS"}
```

본문이 `{` 로 시작하지 않으면 예전 서버의 형식(모드 문자열만)이다. 본문 전체를 `mode` 로 보고 반영한다.

| 모드 | 서버 판정 |
|---|---|
| `FOCUS` | 앉아 있고, 머리가 거의 돌지 않고, 말소리가 없는 상태가 5 초 이어짐 |
| `MOVING` | 앉아 있지 않음(일어난 순간부터. 서서 가만히 있어도 MOVING) |
| `TALKING` | 대화 거리(3.6 m) 안의 얼굴 + 말소리. 서서 마주 보고 말하면 MOVING 보다 우선 |
| `NONE` | 위 어느 것도 아님, 또는 아래의 '알릴 것 없음' 틱 |

전송 규칙 (서버가 1 초마다 하나씩 보낸다):

- 상태가 바뀐 틱: 새 모드를 보낸다.
- 같은 상태가 이어지면 5 초마다 같은 모드를 다시 보낸다.
- 그 밖의 모든 틱: `NONE`.

예: 앉아서 집중하다 일어나 걷고, 멈춰서 대화한 경우

```
FOCUS NONE NONE NONE NONE FOCUS NONE NONE MOVING NONE NONE NONE NONE MOVING NONE TALKING NONE ...
```

HL2 쪽 처리:

- `NONE` 은 창을 닫으라는 뜻이 아니다. 무시한다.
- 창은 새 모드가 왔을 때 띄운다. 새 모드 = 직전에 받은 `NONE` 아닌 모드와 다르거나, 그 모드를 마지막으로 받은 지
  6 초 넘게 지난 경우(이어지는 동안은 5 초마다 오므로, 그보다 길게 끊겼다면 상태가 한 번 끝났다가 다시 시작된 것이다).
- 상태가 끝나는 시점은 따로 오지 않는다. 창을 닫아야 하면 HL2 쪽에서 일정 시간 뒤 닫는다.
- 서버는 상태를 1 초 간격으로 보므로 실제 변화보다 최대 1 초 늦게 올 수 있다.

## 3. HL2_UI

본문: UTF-8 JSON 한 개.

```json
{"target_device_id": "HoloLens2_B", "type": "social", "source": "combined", "value": 0.62, "timestamp": 1730000002.341, "highlight": true}
```

| 필드 | 타입 | 내용 |
|---|---|---|
| `target_device_id` | string | 받을 HL2 의 `clientIdentity`. 자기 것과 같을 때만 반영. 비어 있으면 모두 반영 |
| `type` | string | `emotion` / `physical` / `social` / `digital` |
| `source` | string | 값을 만든 근거. 아래 표 |
| `value` | float | 패널에 바로 쓰는 0~1 값 |
| `timestamp` | double | 서버 시계 기준 Unix 초. 순서 확인용 |
| `highlight` | bool | `true` 면 이 type 의 영역을 하이라이트한다. 아래 3.2 |

| type | source | value | 언제 오나 |
|---|---|---|---|
| `emotion` | `heart_rate` | 심박 <60: 0.0, 60~100: 0.4, 100~130: 0.8, >130: 1.0 | 와치 심박이 올 때마다(약 1 Hz) |
| `physical` | `activity` | 와치 활동 STILL 0.0, WALKING 0.5, RUNNING 1.0 | 와치 활동이 바뀔 때 |
| `physical` | `hqs` | HL2 머리 이동이 걷기면 0.5, 아니면 0.0 | 와치가 없을 때만. 값이 바뀔 때 |
| `social` | `combined` | 0~1 연속값. 아래 3.1 | 마지막으로 보낸 값과 0.05 이상 달라질 때 |
| `digital` | `phone_touch` | 0~1, 0.1 단위. 아래 3.3 | 값이 바뀔 때(폰 보고마다, 최대 약 1 Hz) |

- 매번 값이 오는 게 아니라 바뀔 때 온다. HL2 는 type 별 마지막 값을 들고 있다가 새 값이 오면 바꾼다.
- 와치(서버의 main_watchbridge)가 없거나 5 초 넘게 끊기면 emotion 은 오지 않고, physical 은 `source: "hqs"` 로 온다.
  와치가 돌아오면 다음 와치 활동 변화부터 다시 `activity` 로 온다.
- 폰 앱의 터치 보고가 3 초 넘게 끊기면 digital 은 오지 않는다. 돌아오면 0 부터 다시 쌓여 올라간다.
- Unity `JsonUtility.FromJson` 으로 그대로 읽힌다(type, source: string / value: float / timestamp: double /
  highlight: bool).

### 3.1 social 계산

```
social = 0.5 x (face + voice)          각 항 0~1, 결과 0~1 연속값
```

- `face`: 대화 거리 안에 사람 얼굴이 보이는 정도.
  1. HL2 RGB 에서 얼굴을 초당 10 번 찾는다(OpenCV YuNet).
  2. 얼굴마다 거리를 잰다. depth 가 있으면 얼굴 영역 depth 의 중앙값, 없으면 얼굴 폭(0.15 m)과 카메라 내부 파라미터로 잰다.
  3. 거리를 `sigmoid((3.6 m - 거리) / 0.4 m)` 로 바꾼다. 3.6 m(사회적 대화 거리의 바깥 경계)에서 0.5, 2.6 m 에서 0.92, 4.6 m 에서 0.08.
     얼굴이 여럿이면 가장 큰 값, 없으면 0.
  4. 순간값이 아니라 1 초 시정수로 평활한다. 얼굴이 나타나면 약 1 초에 0.63, 3 초에 0.95 까지 오른다.
- `voice`: 최근 3 초 동안 말소리가 있던 정도.
  1. 마이크 소리를 20 ms 씩 webrtcvad 로 말소리 여부를 가른다.
  2. 최근 3 초 중 말소리였던 시간 비율(밀도)을 낸다.
  3. 밀도를 `sigmoid((밀도 - 0.3) / 0.08)` 로 바꾼다. 밀도 0 에서 0.02, 0.3 에서 0.5, 0.5 이상에서 0.92 이상.
  4. HL2 마이크(HL2_AUDIO)만 쓴다. 모드(TALKING·FOCUS)의 말소리와 같은 값이다. 와치 마이크(WATCH_VAD)는 쓰지 않는다.
- 대략: 아무도 없고 조용함 ≈ 0.01, 가까이 사람이 있고 조용함 ≈ 0.5, 가까이 사람과 대화 중 ≈ 1.0. 그 사이는 연속으로 움직인다.

### 3.2 highlight

서버가 type 별로 따로 판정한다. 한 메시지의 `highlight` 는 그 메시지 `type` 의 영역에만 해당한다.

- 새 값이 같은 type 의 최근 W 초 동안의 값들과 D 이상 다르면 `true`.
- 한 번 `true` 가 되면 그 값부터 다시 잰다(같은 변화로 연달아 켜지지 않는다).
- 같은 값이 다시 온 메시지(emotion 은 약 1 Hz 로 같은 값이 반복된다)와, 와치가 끊기거나 돌아와 source 가 바뀐 첫 메시지는 `false`.
- 하이라이트를 얼마나 보여 줄지는 HL2 가 정한다. 끄는 메시지는 따로 오지 않는다.

현재 기준(서버 `modules/modules_HQS.py` 의 `HIGHLIGHT`, 디버깅하며 바꾼다):

| type | D | W | 뜻 |
|---|---|---|---|
| `emotion` | 0.4 | 3 s | 심박 구간이 한 칸 이상 바뀜(0.8 ↔ 1.0 은 제외) |
| `physical` | 0.5 | 3 s | 활동이 한 단계 이상 바뀜 |
| `social` | 0.3 | 3 s | 3 초 안에 0.3 이상 바뀜 |
| `digital` | 0.5 | 5 s | 폰을 쓰기 시작하거나 멈춘 상태가 5 초 이어짐(digital 은 초당 최대 0.1 씩 움직인다) |

### 3.3 digital 계산

```
digital = 최근 폰 보고 10 개(약 10 초) 중 터치가 있던 보고의 수 / 10
```

- 폰 앱이 1 초마다 그 1 초 동안의 터치 요약을 comm_hub(`PHONE_ACTIVITY`)로 올린다. 손가락이 화면에 조금이라도
  닿았으면(`active_ms > 0`) 그 1 초는 "터치 있음"이다. 탭·스크롤·타이핑을 구분하지 않는다.
- 대략: 계속 스크롤하거나 타이핑 ≈ 0.8~1.0, 2 초에 한 번 탭 ≈ 0.5, 내려놓음 = 0.
- 터치 없이 화면만 보는 사용(영상 시청, 스크롤 없는 읽기)은 0 이다.
- 폰이 붙은 직후나 끊겼다 돌아온 직후는 보고가 10 개가 될 때까지 없는 것을 0 으로 세므로 천천히 올라간다.

## 4. HL2 앱이 이미 해 주고 있어야 하는 것

- HL2DATA 전송(머리 자세, RGB, depth). 상태 판정의 입력이다.
- `HL2_CONTROL` 수신과 `HL2_AUDIO` 전송. 서버가 5 초마다 `HL2Control(send_audio=true, target_device_id=<기기>)` 를
  보내 마이크를 요청한다. `target_device_id` 가 자기 것이 아니면 메시지 전체를 무시한다.
  마이크가 오지 않으면 말소리가 0 이라 TALKING 이 나오지 않고, social 은 얼굴만으로 최대 0.5 다.
