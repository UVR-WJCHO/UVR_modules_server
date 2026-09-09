# Delay-Conditioned Causal Skeleton Nowcasting

실행 명세는 [`../B_DELAY_CONDITIONED_NOWCASTING_PLAN_KO.md`](../B_DELAY_CONDITIONED_NOWCASTING_PLAN_KO.md)
에 있다. 이 README 는 코드 쪽 계약과 현재 진행 상태만 적는다.

```bash
conda activate uvr_integ
```

## 현재 상태

| Phase | 상태 |
|---|---|
| 0 — scaffold 와 계약 고정 | 완료 |
| 1 — HOT3D adapter 와 baseline | 완료, Gate 1 통과 |
| 2 — Residual MLP feasibility | 완료, Gate 2 통과 |
| 3 — Delay-conditioned TCN | 미착수 |

## Phase 0 이 고정한 것

- **canonical skeleton** (`data/canonical_schema.py`): 21-joint 순서, 20-bone tree,
  fingertip index, meter/nanosecond 단위, rigid transform 유틸.
  이 순서는 `_comm/SERVER_RESULT_PROTOCOL.md` §3 의 서버 출력 순서(DexYCB)와 같다.
- **joint mapping** (`configs/joint_maps/mano_to_canonical.json`): smplx MANO joint(16)
  + tip vertex(5) → canonical 21. 매핑은 코드가 아니라 이 JSON 에만 둔다.
- **config identity** (`config.py`): YAML 로딩, 내용 해시, git commit/환경 기록,
  결정론적 `sample_key` 와 `derive_seed`.
- **split** (`data/splits.py`): subject-disjoint v1 manifest 와 교집합 검증.
- **result 표 schema** (`evaluation/result_schema.py`): sequence 단위 long-format 한 장.

## Phase 1 결과

canonical cache 960,750 row (= 480,375 frame x 2 hand), window index는 train 2,868,825 /
val 721,506 / test 994,556 sample (horizon 0/33/66/100/133 ms).

**Gate 1 통과.** val 기준 absolute MPJPE (mm), sequence 평균:

| method | 33 ms | 66 ms | 100 ms | 133 ms |
|---|---|---|---|---|
| hold | 7.44 | 14.78 | 21.94 | 28.90 |
| cv_2frame | 1.70 | 4.81 | 9.13 | 14.43 |
| cv_robust (4-frame fit) | 3.05 | 7.30 | 12.54 | 18.59 |
| const_accel | 1.46 | 4.44 | 9.15 | 15.69 |
| kalman_cv | 1.72 | 4.79 | 9.06 | 14.34 |

wrist speed high tercile에서는 격차가 훨씬 크다 (100 ms: hold 45.6 / cv_2frame 15.8).
horizon 0 에서 Hold 오차는 정확히 0 이고, 곡선은 단조 증가한다.

Phase 1 에서 확인된 사실 중 이후 설계에 영향을 주는 것:

- **`cv_robust`(4-frame linear fit)가 `cv_2frame`보다 나쁘다.** GT MANO history 는 노이즈가
  없어 2-frame 차분이 곧 순간 속도이고, 4-frame fit 은 약 100 ms 를 평균해 가속 구간에서
  뒤처진다. noisy WiLoR history(Phase 5)에서는 순서가 뒤집힐 것으로 예상되므로, §6.2 의
  "best non-learned baseline" 은 history 조건마다 다시 골라야 한다.
- **tip bone 5 개는 강체가 아니다.** MANO 의 joint-to-joint bone 은 상대 표준편차 ~1e-7 로
  고정이지만, fingertip 은 joint regressor 가 아니라 mesh vertex 에서 나와 굽힘에 따라
  ~1e-2 수준으로 변한다. WiLoR 출력도 같은 성질이다. §7.5 의 bone-length loss 는 tip bone 을
  그대로 쓰면 안 된다.
- **palm frame 은 chirality 를 정의상 지운다.** x 축이 `index_mcp - little_mcp` 라 좌수를
  자기 palm frame 에서 그리면 우수와 겹친다. 좌/우 검증은 world 좌표의
  `canonical_schema.chirality` 부호로 한다 — 전체 924,513 유효 frame 에서 LEFT 100% 음수,
  RIGHT 100% 양수.

test split 은 아직 한 번도 평가하지 않았다 (§9.5: 최종 config 당 1회).

## Phase 2 결과 — Gate 2 통과

Residual MLP (312,895 param, `< 1M` 기준 충족), seed 0/1/2 의 val selection metric
(66/100 ms 평균)은 4.005 / 3.997 / 4.045 mm 로 seed 간 편차가 거의 없다.

seed 0 을 sequence-hand 단위 paired bootstrap 으로 비교한 결과 (val, 44 units,
비교 대상은 horizon 마다 다시 고른 best non-learned baseline):

| horizon | best baseline | baseline | model | 개선 | 95% CI | win rate |
|---|---|---|---|---|---|---|
| 33 ms | const_accel | 1.46 mm | 1.00 mm | 31.2% | [+29.5%, +32.9%] | 44/44 |
| 66 ms | const_accel | 4.44 mm | 2.74 mm | 38.2% | [+36.7%, +39.8%] | 44/44 |
| 100 ms | kalman_cv | 9.06 mm | 5.28 mm | 41.7% | [+39.9%, +43.3%] | 44/44 |
| 133 ms | kalman_cv | 14.34 mm | 8.61 mm | 39.9% | [+38.3%, +41.4%] | 44/44 |

Gate 2 기준은 2% 인데 평균 37.8% 다. 이 정도 격차는 leakage 를 먼저 의심해야 해서
`tests/test_no_leakage.py` 로 확인했다 — target frame 좌표를 1 m 오염시켜도 예측이
비트 단위로 동일하고, history 는 항상 target 보다 앞서며, train/val subject 는 서로 겹치지
않는다. leakage 는 아니다.

### 다만 이 수치는 조건 A(GT history) 상한이다

HOT3D 의 MANO GT 는 시간 정규화된 multi-view fit 이라 **프레임별 노이즈가 사실상 없다.**
측정값: 이웃 4 프레임(t±1, t±2)으로 3차 보간해 가운데 프레임을 맞히면 잔차가
**median 0.176 mm** 다. 같은 구간의 33 ms causal CV 외삽 오차(median 0.986 mm)의 1/5.6 이다.

즉 이 데이터에서는 궤적이 거의 전부 저차 다항으로 설명되고, 학습 모델은 그 매끄러움을
CV 보다 잘 활용한다. WiLoR 의 프레임별 추정에는 이런 매끄러움이 없으므로 **37.8% 를
배포 성능으로 주장하면 안 된다** (§10.3: A/B/C 표를 분리, §16: "clean GT 에서만 개선됨"
분기). Phase 4 의 noise augmentation scale 은 이 0.176 mm 가 아니라 실제 WiLoR residual
분포로 정해야 한다 (§4.5).

## HOT3D skeleton 소스 결정 (2026-08-17)

`HOT3D_cache/sequences/*/hands.parquet` 의 21 landmark 는 UmeTrack 규약이라
canonical 21 과 1:1 대응되지 않는다.

```
HOT3D cache : WRIST + THUMB{INTERMEDIATE,DISTAL} + {4 finger}×{PROXIMAL,INTERMEDIATE,DISTAL}
              + 5 FINGERTIP + PALM_CENTER
canonical   : wrist + 손가락 5개 × {mcp, pip, dip, tip}
```

thumb MCP 에 해당하는 landmark 가 없고 PALM_CENTER 는 대응 슬롯이 없어 20/21 만 채워진다.
따라서 canonical cache 는 cache 의 landmark 가 아니라 **raw HOT3D 의 MANO parameter**
(`<seq>/mano_hand_pose_trajectory.jsonl`) 를 `smplx` + `MANO_RIGHT.pkl` 로 풀어 만든다.
배포 경로의 WiLoR 도 같은 MANO 21-joint 를 내보내므로 학습과 배포의 skeleton 이 일치하고,
이후 ReInterHand/GRAB(둘 다 MANO/SMPL-X 기반) 확장도 같은 규약을 쓴다.

미검증으로 남은 항목(Phase 1 의 2D overlay 검증에서 확정):
`flat_hand_mean`, 좌수 mirroring 의 정확도, MANO trajectory(≈16.6 ms 간격)와
RGB frame(30 Hz)의 timestamp 매칭.

## 검증된 리소스 (2026-08-17)

- `uvr_integ`: Python 3.10.20 / torch 2.7.0+cu128 (CUDA 가용) / polars 1.33.1 /
  onnxruntime 1.22.0 (TensorRT·CUDA·CPU EP) / onnx 1.22.0 / numpy 2.0.1 / smplx 설치됨.
  **pyarrow 없음** → Parquet I/O 는 polars 로 통일한다.
- HOT3D cache: 136 sequence / 480,375 frame / 42-joint 완전 annotation 478,268 frame.
  결측 2,107 frame 은 전부 "hand row 0개" 형태이고 부분 결측(21개 미만)은 없다.
  샘플링 간격은 mean 33.333 ms, min 31.33 / max 35.43 ms — frame index 를 시간으로
  가정하면 안 된다.
- `hands.parquet` 좌표계는 headset trajectory 와 같은 world/scene frame
  (P0001_10a27bf7 에서 corr(wrist, head_t) = +0.45~+0.84, std(wrist−head) < std(wrist)).
  단일 sequence 표본이므로 Phase 1 에서 전 sequence 재확인 대상.
- split v1 실측: train 83 seq / 299,349 frame ≈ 166.3분, val 22 seq / 74,874 ≈ 41.6분,
  test 31 seq / 106,152 ≈ 59.0분.
- DexYCB / HO3D 는 여전히 Permission denied. GRAB / ReInterHand 는 접근 가능.

## 실행

```bash
CFG=modules/delay_nowcasting/configs/data/hot3d_v1.yaml

python -m modules.delay_nowcasting.data.splits        --config $CFG
python -m modules.delay_nowcasting.data.build_cache   --config $CFG
python -m modules.delay_nowcasting.data.build_windows --config $CFG --split val \
  --horizons-ms 0 33 66 100 133
python -m modules.delay_nowcasting.evaluation.evaluate --config $CFG --split val
python -m modules.delay_nowcasting.evaluation.figures  --config $CFG --split val

MODEL=modules/delay_nowcasting/configs/model/residual_mlp.yaml
OUT=research_data/hot3d_v1/residual_mlp
python -m modules.delay_nowcasting.training.train --config $MODEL          # seed 0,1,2
python -m modules.delay_nowcasting.evaluation.evaluate --config $CFG --split val \
  --tag phase2 --checkpoint $OUT/seed{0,1,2}/best_model.pt
python -m modules.delay_nowcasting.evaluation.summarize --config $CFG --split val \
  --tag phase2 --method residual_mlp/seed0 --horizons-ms 33 66 100 133 --gate 2
python -m modules.delay_nowcasting.evaluation.figures --config $CFG --split val \
  --tag phase2 --paired-method residual_mlp/seed0

python -m pytest modules/delay_nowcasting/tests -q
```

그림은 저장소 최상위 `output/delay_nowcasting/` 에 저장된다 (research/ 안쪽보다 찾기 쉽게).
나머지 산출물은 `research_data/hot3d_v1/` 에 있다.

이 연구의 소스와 산출물은 전부 `research/` 안에만 두고, `research/` 는 저장소 `.gitignore`
에 올라가 있어 git 에 들어가지 않는다. production 경로(`main_handtrack.py`, `_comm/`)
변경은 Gate 2 통과 이후 별도 커밋으로만 반영한다 (B 계획 §2.5).
