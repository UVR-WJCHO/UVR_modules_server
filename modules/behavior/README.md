# 메타 객체 (Meta-Object)

이 프로젝트는 3D 모델 파일(`.glb`)을 처리하여 파트별 VLM 입력 이미지를 생성하고, VLM에 재질 및 어포던스 정보를 질의하여 그 결과를 시각화하는 파이프라인을 제공합니다.

## Setup

```bash
conda env create -f environment.yml  
conda activate metaobj
```

```bash
mkdir data  # 본 경로에 .glb파일을 저장해주세요
```

API 키 설정:
```
OPENAI_API_KEY=...       
```

## 디렉토리 구조

```
├── data
│   └── SaturnV.glb                         # 입력 3D 모델 파일 (예시)
├── environment.yml
├── example.env
├── main.py                                     # 전체 파이프라인 실행
├── README.md
├── utils
│   ├── config.py                               # 재질 라이브러리 및 어포던스 정의
│   ├── preprocess_glb_for_vlm.py               # GLB -> VLM 입력 이미지 생성
│   ├── query_vlm.py                            # VLM 질의 및 결과 저장
│   ├── visualize.py                            # 결과 시각화
│   └── vlm_utils.py                            # VLM API 래퍼
└── vlm_input                                   # main.py 실행 시 생성
    └── SaturnV
        ├── gpt_input                           # 파트별 X-Ray 이미지 (뷰별 서브폴더)
        │   ├── 01
        │   │   ├── 01.png
        │   │   ├── 02.png
        │   │   └── ...
        │   └── ...
        ├── images                              # 뷰별 전체 씬 렌더링
        │   ├── 001.png
        │   └── ...
        ├── seg                                 # 파트 세그멘테이션 맵 (.npy)
        │   ├── 001_s.npy
        │   └── ...
        ├── property_visualizations             # 속성별 시각화 결과
        │   ├── caption
        │   ├── material
        │   ├── affordance
        │   ├── youngs_modulus_GPa
        │   ├── density_g_cm3
        │   ├── poissons_ratio
        │   ├── tensile_strength_MPa
        │   ├── hardness
        │   └── thermal_conductivity_W_mK
        ├── vlm_result.json                     # 최종 JSON 결과
        └── vlm_result.txt                      # VLM 초기 응답 로그
```

## 실행

```bash
python main.py \
    --glb_path ./data/SaturnV/SaturnV.glb \
    --vlm_input_dir ./vlm_input
```

<details>
<summary>CLI arguments</summary>

- `--glb_path` : 입력 `.glb` 파일 경로 (기본값: `./data/SaturnV/SaturnV.glb`)
- `--vlm_input_dir` : VLM 입력 데이터 및 결과 저장 디렉토리 (기본값: `./vlm_input`)
- `--output_json` : 최종 JSON 결과 저장 경로 (기본값: `vlm_input_dir/<case_name>/vlm_result.json`)
</details>

1. **VLM 입력 전처리**: `.glb` 파일에서 파트별 X-Ray 오버레이 이미지 및 세그멘테이션 맵을 생성합니다.
2. **VLM 질의**: VLM으로 각 파트의 caption, 재질, 어포던스를 추론하고 `vlm_result.json`으로 저장합니다.
3. **시각화**: 추정한 재질 및 재질에 따른 물성 속성을 뷰별 세그멘테이션 맵에 투영하여 `property_visualizations/` 에 저장합니다.


## 실행 결과

- VLM 질의 결과 예시 (`vlm_result.json`)
```json
{
    "version": 1,
    "parts": [
        {
            "partId": 1,
            "caption": "rocket nose cone",
            "material": "aluminium",
            "affordance": "attach",
            "youngs_modulus_GPa": [68.0, 75.0],
            "density_g_cm3": [2.6, 2.85],
            "poissons_ratio": [0.32, 0.36],
            "tensile_strength_MPa": [70.0, 600.0],
            "hardness": [20.0, 180.0],
            "thermal_conductivity_W_mK": [120.0, 235.0]
        },
        ...
    ]
}
```

- 시각화 예시 (`property_visualizations/`)

![Property Visualization](./vlm_input/SaturnV/property_visualizations/material/vis_001_s.png)
