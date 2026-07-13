# Model Weights / Checkpoints

이 프로젝트가 사용하는 모델 가중치는 모두 **git에 포함되지 않습니다** (`.gitignore` 처리). 새 환경에서 구동하려면 아래 위치에 파일을 직접 배치해야 합니다. 경로는 모두 repo root 기준입니다.

총 용량 ≈ **16 GB**.

---

## 1. Mesh Reconstruction (TRELLIS) — `pretrained/meshrecon/`

소비: [modules/modules_mesh.py](modules/modules_mesh.py) `MeshReconstructor` → 진입점 `main_meshrecon.py`, `main_meshrecon_webcam.py`
로딩: `TrellisImageTo3DPipeline.from_pretrained("pretrained/meshrecon/diffusion")` → `diffusion/pipeline.json`이 아래 파일들을 참조

### 1-1. 커스텀 학습 가중치 — `pretrained/meshrecon/diffusion/ckpts_new/`
| 파일 | 크기 |
|---|---|
| `denoiser_step0700000.pt` | 2.7 GB |
| `denoiser_phy_step0700000.pt` | 1.6 GB |
| `decoder_step0100000.pt` | 1.2 GB |
| `property_output_step0100000.pt` | 1.1 GB |
| `property_decoder_step0100000.pt` | 1.1 GB |

### 1-2. TRELLIS 베이스 가중치 — `pretrained/meshrecon/trellis/ckpts/`
| 파일 | 크기 |
|---|---|
| `slat_flow_img_dit_L_64l8p2_fp16.safetensors` | 1.2 GB |
| `ss_flow_img_dit_L_16l8_fp16.safetensors` | 1.1 GB |
| `slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors` | 174 MB |
| `slat_enc_swin8_B_64l8_fp16.safetensors` | 166 MB |
| `slat_dec_rf_swin8_B_64l8r16_fp16.safetensors` | 164 MB |
| `slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors` | 164 MB |
| `ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB |
| `ss_enc_conv3d_16l8_fp16.safetensors` | 114 MB |

> 베이스 가중치 출처: Microsoft TRELLIS (HuggingFace `microsoft/TRELLIS-image-large`).

---

## 2. Object Detection (YOLO) — `pretrained/object/`

| 파일 | 크기 | 소비 | 코드 |
|---|---|---|---|
| `yolo_100doh_best.pt` | 20 MB | HoTrack 손-객체 검출 | [modules_hotrack.py:106](modules/modules_hotrack.py#L106) (`InteractiveHoTrackSegmentor`) |
| `yolo11m.pt` | 39 MB | 객체 검출 `ObjTracker` | [modules_obj.py:16](modules/modules_obj.py#L16) (→ `main_handtrack.py`) |
| `yolo11n.pt` | 5.4 MB | (현재 미사용) | — |

> 출처: Ultralytics YOLO11 (`yolo11m.pt`, `yolo11n.pt`). `yolo_100doh_best.pt`는 100DOH 기반 커스텀 학습본.

---

## 3. SAM2 Segmentation — `modules/segmentor/sam2_realtime/checkpoints/`

소비: [modules_segment.py:33](modules/modules_segment.py#L33) (`HOSegmentor`) 및 HoTrack의 SAM2 백엔드 ([modules_hotrack.py:21](modules/modules_hotrack.py#L21))

| 파일 | 크기 | 비고 |
|---|---|---|
| `sam2.1_hiera_large.pt` | 857 MB | `HOSegmentor` 기본 (`modules_segment.py`) |
| `sam2.1_hiera_base_plus.pt` | 309 MB | variant `base_plus` |
| `sam2.1_hiera_small.pt` | 176 MB | variant `small` |
| `sam2.1_hiera_tiny.pt` | 149 MB | HoTrack 기본 variant (`sam2_variant="tiny"`) |

추가로 같은 `segmentor`의 손 검출용:
| 파일 | 위치 | 소비 |
|---|---|---|
| `100DOH_small.pt` | `modules/segmentor/100DOH_small.pt` | [modules_segment.py:37](modules/modules_segment.py#L37) (`HOSegmentor`) — *현재 파일 없음, 배치 필요* |

> 출처: Meta SAM 2.1 (`facebook/sam2.1-hiera-*`).

---

## 4. Hand Tracking — `modules/handtracker*/`

진입점: `main_handtrack.py` ([modules_hand.py](modules/modules_hand.py))

| 파일 | 위치 | 크기 | 소비 |
|---|---|---|---|
| `checkpoint.pth` | `modules/handtracker/checkpoint/SAR_AGCN4_cross_wBGaug_extraTrue_resnet34_s0_Epochs50/` | 675 MB | SARTE `HandTracker_our` ([config.py:15](modules/handtracker/config.py#L15)) |
| `wilor_final.ckpt` | `modules/handtracker_wilor/pretrained_models/` | 2.4 GB | WiLoR `HandTracker_our_wilor` ([module_WILOR.py:73](modules/handtracker_wilor/module_WILOR.py#L73)) |
| `detector.pt` | `modules/handtracker_wilor/pretrained_models/` | 52 MB | WiLoR 손 검출 ([module_WILOR.py:77](modules/handtracker_wilor/module_WILOR.py#L77)) |

### ONNX 변형 (WILOR-ONNX, `main_handtrack.py` v3) — `pretrained/handtracker_onnx/`

소비: [modules_hand.py](modules/modules_hand.py) `HandTracker_onnx` → `modules/handtracker_onnx/` (self-contained; ONNXRuntime + YOLO). `main_handtrack.py`에서 space 키로 v3 선택. 경로는 `WilorHandTrackerONNX(onnx_path=..., detector_path=...)` 인자로 오버라이드 가능.

| 파일 | 위치 | 크기 | 비고 |
|---|---|---|---|
| `wilor_final_standard.onnx` (+ external-data weight 411개) | `pretrained/handtracker_onnx/` | ≈ 2.4 GB | ONNX 그래프(849 KB) + 외부 weight. .onnx와 weight 파일들은 **같은 폴더**에 있어야 로드됨 |
| `detector.pt` | `pretrained/handtracker_onnx/` | 52 MB | YOLO 손 검출 (WiLoR와 동일 파일) |
| `model_config.yaml` | `pretrained/handtracker_onnx/` | 2 KB | WiLoR 모델 config |

> `onnxruntime-gpu` 필요. MANO 파일은 ONNX 추론에 불필요. `uvr_integ` 환경에서 ~12ms/추론.

### MANO 모델 (손 메쉬 파라미터)
| 파일 | 위치 | 크기 |
|---|---|---|
| `MANO_RIGHT.pkl`, `MANO_LEFT.pkl` | `modules/handtracker/mano_data/mano/models/` | 3.7 MB each |
| `MANO_RIGHT.pkl` | `modules/handtracker_wilor/mano_data/` | 3.7 MB |

> 출처: WiLoR (공식 릴리스), MANO (`mano.is.tue.mpg.de`, 라이선스 동의 필요).

---

## 5. Gesture Classification — `modules/gestureclassifier/checkpoints/`

| 파일 | 크기 | 소비 |
|---|---|---|
| `checkpoint-40.tar` | 13 MB | `GestureClassfier` ([main_handtrack.py:62](main_handtrack.py#L62)) |

---

## 6. (참고) weight는 아니지만 git-ignore되는 대용량 데이터

| 경로 | 내용 |
|---|---|
| `_calibration/rm_depth_ahat/*.bin` | HL2 깊이 센서 캘리브레이션 (연결 시 자동 생성) |
| `output/*` | 파이프라인 실행 결과물 (rgb/depth/mesh.glb/property.json 등) |
| `modules/behavior/vlm_input/`, `modules/behavior/data/` | behavior VLM 입력/예시 데이터 |
| `.env` | API 키 (`OPENAI_API_KEY`) |

---

## .gitignore 패턴 요약

```
pretrained/*
!pretrained/.gitkeep
modules/segmentor/sam2_realtime/checkpoints/*
modules/segmentor/100DOH_small.pt
modules/handtracker/checkpoint/*
modules/handtracker_wilor/pretrained_models/*
modules/handtracker_wilor/mano_data/*
modules/gestureclassifier/checkpoints/*
modules/behavior/vlm_input/
modules/behavior/data/
_calibration/rm_depth_ahat/*
output/*
.env
```
