# HL2_SCENE — 기기별 공유 콘텐츠 자세 (HL2 → 서버)

여러 HL2 가 comm_hub 에 붙었을 때, 각 기기가 **자기 월드 좌표계에서 본 공유 콘텐츠(로버)의
자세**를 서버로 올린다. 서버는 이걸로 기기 월드들 사이의 변환을 구해 하나의 좌표계로 모은다.
2026-10-17 HL2 앱 B 에 추가됨. 서버 → HL2 역방향은 아직 없다.

## 전송

```
["", "UPLOAD", "HL2_SCENE", <SOURCE>, <FID>, <DATA>]
```

| 프레임 | 값 |
|---|---|
| SOURCE | `HL2Client.clientIdentity` — HL2DATA 와 같은 값. **이 기기의 월드 정합을 찾는 키** |
| 소켓 identity | `<clientIdentity>_SCENE` (전용 DEALER 1 개, RECV_REG 없음) |
| FID | 소켓별 증가 정수 |
| DATA | UTF-8 JSON. `type` 으로 종류 구분. 현재 `rover_pose` 하나 |

서버 쪽 상수: `_comm/hub_client.py` 의 `KW_HL2_SCENE`. 구독 예:

```python
c = HubClient(host, port, recv_kw=KW_HL2_SCENE, identity=b"MYMODULE_SCENE", queue_size=64)
```

상태 전이 메시지라 conflate 하면 안 된다(`queue_size` 를 넉넉히). 현재 서버에 구독자가 없다 —
broker 가 keyword 당 64 개까지 캐시했다가 버린다.

## `rover_pose`

```json
{
  "type": "rover_pose",
  "device": "HoloLens2_User",
  "rover": "rover",
  "state": "anchored",
  "seq": 0,
  "t_unity": 123.45,
  "t_utc_ms": 1760000000000,
  "frame": "hl2_world",
  "root":  { "p": [x, y, z], "r": [qx, qy, qz, qw], "s": [1, 1, 1] },
  "parts": [
    { "id": "body", "active": true,
      "local": { "p": [...], "r": [...], "s": [...] },
      "world": { "p": [...], "r": [...], "s": [...] } }
  ]
}
```

| 필드 | 의미 |
|---|---|
| `frame` | 이 HL2 의 Unity 월드. **앱 시작 시 머리 자세가 원점.** 왼손 좌표계, +Y 위, m, 쿼터니언 (x, y, z, w) |
| `root` | 로버 루트의 월드 자세. 책상 더블 탭 지점, 바닥 정렬 피벗 |
| `parts[].local` | root 기준 자세 |
| `parts[].world` | 이 HL2 월드 기준 자세. `world = root ∘ local` |
| `parts[].id` | 폰 핸드오프 `PartData.id` 와 같다 (`body`, `camera`, `lights`, `parts`, `spectrometer`, `wheels` 중 카탈로그에 있는 것) |
| `parts[].active` | `false` = 폰으로 넘어가 로버에서 빠진 파츠 |
| `state` | 아래 |
| `seq` | 기기별 증가 |
| `t_unity` | Unity `Time.time` (s). HL2DATA / HL2_AUDIO 의 timestamp 와 같은 축 |
| `t_utc_ms` | UTC epoch ms. 기기 간 비교용 |

### `state`

```
anchored   착지 애니메이션이 끝난 순간 1 회
moved      root 나 파츠가 2 mm / 0.5° 넘게 바뀌면, 최대 10 Hz (분해 제스처·핸드오프)
hidden     로버가 사라지거나 단계를 떠날 때 1 회. parts 는 비어 있다
```

스크린 앞 미리보기 단계에서는 보내지 않는다.

## 서버가 전제하는 것 (정합 모듈을 만들 때 확인할 것)

- **로버가 물리적으로 같은 자리에 있다.** 그래야 기기 A 의 `root` 와 기기 B 의 `root` 가 같은
  물체를 가리키고, 두 월드 사이 변환이 `T_B←A = root_B ∘ root_A⁻¹` 로 나온다. 로버 루트가 곧
  공유 anchor 다.
- **`root.r` 의 yaw 가 기기마다 같은 물리 방향이어야 한다.** 더블 탭 지점이 같아도 방향이
  사용자 시선 등으로 정해지면 그 차이가 그대로 정합 오차가 된다. 위치만 믿고 회전은 따로
  검증하는 쪽이 안전하다.
- 좌표계가 Unity(왼손, +Y 위)다. 서버의 손 결과(`ServerResult.hand`)는 OpenCV 카메라
  프레임(오른손, +Y 아래)이라 섞을 때 변환이 필요하다.
- 메시지는 이벤트성이라 마지막 상태만 들고 있으면 되지만, `hidden` 을 놓치면 사라진 로버를
  계속 그리게 된다. 구독자는 conflate 하지 않고 `state` 를 순서대로 처리해야 한다.
