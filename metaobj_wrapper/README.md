# GLB Rocket Combiner

`inputs/glbs/`에 있는 GLB 파드 5개를 순서대로 읽고, `inputs/transforms.json`의 transform을 적용해서 하나의 로켓 GLB로 export합니다.

## Conda 환경

메인 서버에서 `main_meshrecon_comm.py`가 이 래퍼를 직접 import하므로,
프로젝트의 GPU 실행 환경과 같은 `uvr_integ`를 사용합니다.

```bash
cd <UVR_modules_server>/metaobj_wrapper
conda activate uvr_integ
```

## 기본 입력

현재 코드는 아래 파일을 기본값으로 사용합니다.

```text
inputs/glbs/mesh_0.glb
inputs/glbs/mesh_1.glb
inputs/glbs/mesh_2.glb
inputs/glbs/mesh_3.glb
inputs/glbs/mesh_4.glb
inputs/transforms.json
```

`inputs/transforms.json`의 `parts` 배열은 GLB 파일 순서와 1:1로 매칭됩니다.

```text
mesh_0.glb <- parts[0]
mesh_1.glb <- parts[1]
mesh_2.glb <- parts[2]
mesh_3.glb <- parts[3]
mesh_4.glb <- parts[4]
```

## 실행

```bash
python combine_rocket_glb.py
```

출력 파일:

```text
outputs/rocket.glb
outputs/output.json
```

## Transform 형식

각 part는 아래 필드를 사용할 수 있습니다.

```json
{
  "name": "stage_1_start",
  "translation": [-0.005876, -0.003636, 0.71148],
  "rotation_euler_degrees": [0, 0, 0],
  "scale": [1.0, 1.0, 1.0]
}
```

지원 필드:

- `translation`: `[x, y, z]`
- `rotation_euler_degrees`: degree 단위 `[x, y, z]`
- `rotation_euler_radians`: radian 단위 `[x, y, z]`
- `rotation_quaternion`: `[w, x, y, z]`
- `rotation_quaternion_xyzw`: `[x, y, z, w]`
- `scale`: 숫자 1개 또는 `[x, y, z]`
- `matrix`: 4x4 matrix 또는 길이 16 flat matrix

현재 transform 값은 Blender 좌표계로 해석합니다. GLB export 좌표계에 맞게 내부에서 다음 변환을 적용합니다.

```text
Blender (x, y, z) -> glTF/GLB (x, z, -y)
```

GLB 파드의 mesh 좌표는 vertex에 bake하지 않고 node transform으로 배치합니다. Blender에서 import했을 때 각 stage node가 transform을 가진 상태로 들어갑니다.

## GLB 내부 구조

생성된 `outputs/rocket.glb`는 `test.glb`와 같은 형태로 root node가 mesh node들을 children으로 묶습니다.

```text
scene: rocket
nodes:
  0 control_parts0 -> mesh 0
  1 control_parts1 -> mesh 1
  2 control_parts2 -> mesh 2
  3 control_parts3 -> mesh 3
  4 control_parts4 -> mesh 4
  5 rocket -> children [0, 1, 2, 3, 4]
```

`control_parts0`부터 `control_parts4`까지의 이름은 `outputs/output.json`의 `objectName`과 맞춰집니다.

mesh primitive attribute도 `test.glb`와 최대한 비슷하게 맞춥니다.

```text
POSITION
NORMAL
TANGENT
TEXCOORD_0
TEXCOORD_1
```

입력 GLB에 없는 `NORMAL`, `TANGENT`, `TEXCOORD_1`은 export 후 binary chunk에 추가합니다. `trimesh`가 붙이는 mesh `extras`, 기본값인 primitive `mode: 4`, `doubleSided: false`는 제거합니다. `asset.generator`는 사용하는 exporter가 달라서 `test.glb`와 다를 수 있습니다.

## Output JSON

`python combine_rocket_glb.py`를 실행하면 `outputs/output.json`도 같이 생성됩니다.

```json
{
  "version": 1,
  "parts": [
    {
      "objectName": "control_parts0",
      "colorHex": "#00FFFF",
      "colorAlpha": 1.0,
      "metalic": 0.0,
      "smoothness": 0.5,
      "texture": "",
      "weight": "",
      "temperature": "",
      "textureList": [],
      "material": "aluminium"
    }
  ]
}
```

기본으로 항상 들어가는 프로퍼티는 [combine_rocket_glb.py](combine_rocket_glb.py)의 `PART_METADATA_BASE_PROPERTIES`에 있습니다.

```python
PART_METADATA_BASE_PROPERTIES = {
    "colorHex": "#FFFFFF",
    "colorAlpha": 1.0,
    "metalic": 0.0,
    "smoothness": 0.5,
    "texture": "",
    "weight": "",
    "temperature": "",
    "textureList": [],
}
```

추가 공통 프로퍼티는 `PART_METADATA_ADDITIONAL_PROPERTIES`에 정의합니다. 새 속성을 추가하려면 여기에 key/value를 넣고, 필요 없는 속성은 이 dict에서 지우면 됩니다.

```python
PART_METADATA_ADDITIONAL_PROPERTIES = {
    "material": "aluminium",
    "affordance": "attach",
    "youngs_modulus_GPa": [68.0, 75.0],
}
```

part별로 다른 값을 넣으려면 `inputs/transforms.json`의 각 part 항목에 `metadata` 객체를 추가하세요. 기본값은 유지되며, `metadata`에 입력된 값이 덮어써집니다.

```json
{
  "name": "control_parts1",
  "translation": [0.0, 1.0, 0.0],
  "metadata": {
    "material": "steel",
    "density_g_cm3": [7.8, 7.9],
    "texture": "metallic"
  }
}
```

`PART_METADATA_OVERRIDES`는 여전히 정적인 part index별 기본값을 설정할 때 사용합니다.

```python
PART_METADATA_OVERRIDES = {
    0: {"colorHex": "#00FFFF"},
    1: {"colorHex": "#FF0000", "material": "steel"},
}
```

## 경로를 직접 지정하는 실행

필요하면 기본 하드코딩 경로를 CLI 인자로 덮어쓸 수 있습니다.

```bash
python combine_rocket_glb.py \
  --parts a.glb b.glb c.glb d.glb e.glb \
  --transforms inputs/transforms.json \
  --output outputs/rocket.glb \
  --metadata-output outputs/output.json
```

## 확인

생성 후 간단히 검증하려면:

```bash
conda run -n rocket_glb python -c "import trimesh; s=trimesh.load('outputs/rocket.glb', force='scene', process=False); print(len(s.geometry)); print(s.bounds.tolist())"
```

정상이라면 geometry 개수는 `5`입니다.

Blender에서 확인할 때는 기존에 import된 오브젝트를 삭제한 뒤 `outputs/rocket.glb`를 새로 import하세요.
