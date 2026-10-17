"""공개 3D 모델로 metaobj 예시를 만든다 (.glb + .json + 티저 .png, 객체별 폴더).

HL2 쪽 테스트에 쓸 다양한 객체가 필요한데 실제 파이프라인은 캡처·정합을 거쳐야 파츠가
나온다. 대신 이미 파츠로 나뉜 공개 GLB(Khronos glTF-Sample-Assets)를 받아 파츠별로 쪼개고,
서버와 같은 export_metaobj 로 합본·물성 추정·티저까지 돌린다.

behavior 모듈의 재질 사전이 아직 금속 9 종뿐이라 비금속 파츠도 금속으로 찍힌다. 그래서
build 뒤에 사람이 OVERRIDES 에 파츠별 재질을 적고 apply 로 덮어쓴다. 비금속 물성은
NONMETAL_PRIORS 의 임시값이다. 모듈이 비금속을 내게 되면 이 둘은 지운다.

    python metaobj_wrapper/make_examples.py build ToyCar Lantern    # 다운로드 -> 파츠 -> metaobj
    python metaobj_wrapper/make_examples.py apply ToyCar Lantern    # OVERRIDES 로 재질 덮어쓰기
    python metaobj_wrapper/make_examples.py all   ToyCar Lantern

출력: output/metaobj_examples/<이름>/<이름>.glb  .json  .png  — 객체 폴더에는 이 셋만 둔다.
      출처·라이선스·파츠 매핑·덮어쓰기 기록은 output/metaobj_examples/SOURCES.md 한 파일에 모은다.
      다운로드·파츠·VLM 렌더 같은 중간물은 output/metaobj_examples/_work/<이름>/ 에 두고 지워도 된다.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "modules"))
from modules_metaobj import export_metaobj, part_nodes_in_vlm_order, ranges_for   # noqa: E402

ASSETS = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
OUT_ROOT = _REPO / "output" / "metaobj_examples"
WORK = OUT_ROOT / "_work"
MAX_PARTS = 12          # VLM 호출이 파츠 수에 비례한다. 넘치면 상위 노드 단위로 묶는다
TARGET_SIZE_M = 0.3     # 공개 모델은 단위가 제각각이라 가장 긴 변을 이 길이로 맞춘다

# 파츠별 재질 덮어쓰기. build 결과의 parts.json(인덱스 -> 원본 노드 이름)과 vlm_result.json 을
# 보고 사람이 적는다. 없는 파츠는 VLM 값 그대로.
OVERRIDES: dict[str, dict[str, str]] = {
    # objectName 은 parts.json 의 index 순서. 주석은 원본 노드.
    "ToyCar":  {"control_parts0": "glass",       # Glass  (창)
                "control_parts1": "fabric",      # Fabric (시트)
                "control_parts2": "zinc_alloy"}, # ToyCar (다이캐스트 차체)
    "Lantern": {"control_parts2": "wood"},       # LanternPole_Body (나무 기둥). 등·사슬은 VLM 값 유지
    "ABeautifulGame": {                          # 대리석 체스. 킹 둘은 VLM 대로 유리
        **{f"control_parts{i}": "stone" for i in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)},
        "control_parts5": "glass", "control_parts6": "glass"},
    "AnisotropyBarnLamp": {"control_parts0": "glass",           # Lamp Glass
                           "control_parts1": "stainless_steel", # Lamp Filament
                           "control_parts2": "carbon_steel"},   # Lamp Metal (법랑 철)
    "AntiqueCamera": {"control_parts0": "wood",                 # tripod
                      "control_parts1": "brass"},               # camera
    "CesiumMilkTruck": {"control_parts0": "carbon_steel",       # 차체 (텍스처)
                        "control_parts1": "glass",              # 어두운 녹색 = 창
                        "control_parts2": "plastic",            # 검정 트림
                        "control_parts3": "rubber", "control_parts4": "rubber"},   # Wheels
    "CommercialRefrigerator": {"control_parts0": "stainless_steel",  # FridgeCase
                               "control_parts1": "plastic",         # FridgeInterior
                               "control_parts2": "glass", "control_parts3": "glass", "control_parts4": "glass",  # 병
                               "control_parts5": "stainless_steel", # FridgeDoor
                               "control_parts6": "glass",           # FridgeGlass
                               "control_parts7": "plastic"},        # FridgeDoorInner
    "LightsPunctualLamp": {"control_parts0": "paper",           # '2' 신문지 갓
                           "control_parts1": "glass",           # '1' 전구 (18 verts, BLEND)
                           "control_parts2": "wood"},           # '0' 스탠드
    "SheenChair": {"control_parts0": "fabric",                  # label
                   "control_parts1": "stainless_steel",         # metal
                   "control_parts2": "wood", "control_parts3": "fabric"},
    "SheenWoodLeatherSofa": {"control_parts0": "leather",       # Brown
                             "control_parts1": "fabric", "control_parts2": "fabric",   # Stripes, Paisley
                             "control_parts3": "fabric",        # Frame_Fabric
                             "control_parts4": "wood",          # Frame
                             "control_parts5": "fabric"},       # Fringe
}



def download(name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    glb = dest / f"{name}.glb"
    if not glb.exists():
        urllib.request.urlretrieve(f"{ASSETS}/{name}/glTF-Binary/{name}.glb", glb)
    if not (dest / "metadata.json").exists():
        urllib.request.urlretrieve(f"{ASSETS}/{name}/metadata.json", dest / "metadata.json")
    return glb


class _CachedEstimator:
    """_work 에 vlm_result.json 이 있으면 VLM 을 다시 부르지 않는다. 어댑터만 고쳐 다시 돌릴 때 쓴다."""

    def __init__(self, inner, cache: Path):
        self.inner, self.cache = inner, cache

    def run(self, glb_path, output_json=None, vlm_input_dir=None):
        if self.cache.exists():
            print(f"[examples] VLM 결과 재사용: {self.cache}", flush=True)
            if output_json:
                shutil.copy(self.cache, output_json)
            return output_json or str(self.cache)
        return self.inner.run(glb_path, output_json=output_json, vlm_input_dir=vlm_input_dir)


def write_sources() -> None:
    """출처·라이선스·파츠 매핑·덮어쓰기 기록을 SOURCES.md 하나에 다시 쓴다. _work 가 있는 객체만."""
    lines = ["# metaobj examples — sources, parts, and overrides\n",
             f"All models from Khronos glTF-Sample-Assets ({ASSETS}). Scaled to {TARGET_SIZE_M} m longest side.",
             "Materials marked 'override' were set by hand from the source node names; the VLM's own answer is",
             "shown beside them. Property ranges are per-material table values, not measurements.\n"]
    for d in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir() and p.name != "_work"):
        name, work = d.name, WORK / d.name
        meta = d / f"{name}.json"
        if not meta.exists() or not (work / "parts.json").exists():
            continue
        legal = json.loads((work / "source" / "metadata.json").read_text()).get("legal", []) \
            if (work / "source" / "metadata.json").exists() else []
        lic = "; ".join(f"{e.get('license')} — {e.get('artist')} ({e.get('year')})" for e in legal) or "?"
        parts = json.loads((work / "parts.json").read_text())
        final = {p["objectName"]: p["material"] for p in json.loads(meta.read_text())["parts"]}
        vlm = {}
        if (work / "vlm_result.json").exists():
            order = part_nodes_in_vlm_order(d / f"{name}.glb")
            vlm = {order[p["partId"] - 1]: p["material"]
                   for p in json.loads((work / "vlm_result.json").read_text())["parts"]}
        lines.append(f"## {name}\n\nLicense: {lic}\n\n| objectName | source node | material | VLM said |\n|---|---|---|---|")
        for p in parts:
            n = p["objectName"]; v = vlm.get(n, "?"); m = final[n]
            lines.append(f"| {n} | {p['source_node']} | {m}{' (override)' if m != v else ''} | {v} |")
        lines.append("")
    (OUT_ROOT / "SOURCES.md").write_text("\n".join(lines))


def split_parts(scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh]]:
    """(이름, 월드 좌표로 구운 메시) 목록. geometry 노드 하나가 파츠 하나다.

    노드가 MAX_PARTS 를 넘으면 루트 바로 아래 자식 단위로 묶는다. 그래도 넘치면 큰 것부터
    MAX_PARTS-1 개를 두고 나머지를 하나로 합친다.
    """
    def baked(node):
        T, g = scene.graph[node]
        m = scene.geometry[g].copy()
        m.apply_transform(T)
        return m

    nodes = [n for n in scene.graph.nodes_geometry if len(scene.geometry[scene.graph[n][1]].vertices)]
    if len(nodes) <= MAX_PARTS:
        return [(n, baked(n)) for n in nodes]

    groups: dict[str, list] = {}
    for n in nodes:
        top = n
        while (p := scene.graph.transforms.parents.get(top)) not in (None, scene.graph.base_frame):
            top = p
        groups.setdefault(top, []).append(baked(n))
    parts = sorted(((k, trimesh.util.concatenate(v)) for k, v in groups.items()),
                   key=lambda kv: -len(kv[1].vertices))
    if len(parts) > MAX_PARTS:
        head, rest = parts[:MAX_PARTS - 1], parts[MAX_PARTS - 1:]
        parts = head + [("misc", trimesh.util.concatenate([m for _, m in rest]))]
    return parts


def build(name: str, estimator) -> None:
    out, work = OUT_ROOT / name, WORK / name
    src = download(name, work / "source")
    scene = trimesh.load(str(src), force="scene")
    parts = split_parts(scene)

    # 크기 정규화: 전체 bbox 기준으로 같은 배율을 모든 파츠에 적용해 배치가 유지되게 한다
    allv = np.vstack([m.vertices for _, m in parts])
    scale = TARGET_SIZE_M / float((allv.max(0) - allv.min(0)).max())
    center = (allv.max(0) + allv.min(0)) / 2.0

    pdir = work / "parts"
    shutil.rmtree(pdir, ignore_errors=True)
    (pdir / "glbs").mkdir(parents=True)
    files, specs, index = [], [], []
    for i, (node, mesh) in enumerate(parts):
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        f = pdir / "glbs" / f"mesh_{i}.glb"
        trimesh.Scene(mesh).export(str(f))
        files.append(f)
        specs.append({"name": node, "matrix": np.eye(4).tolist()})
        index.append({"index": i, "objectName": f"control_parts{i}", "source_node": node,
                      "vertices": int(len(mesh.vertices))})
    (pdir / "transforms.json").write_text(json.dumps({"parts": specs}, indent=2))
    (work / "parts.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[{name}] {len(parts)} parts: {[n for n, _ in parts]}  scale x{scale:.4f}")

    glb, meta, png = export_metaobj(files, pdir / "transforms.json", out, name,
                                    _CachedEstimator(estimator, work / "vlm_result.json") if estimator else None,
                                    work_dir=work / "vlm")
    # export_metaobj 는 vlm_result.json 을 결과 폴더에 남긴다. 예시 폴더에는 세 파일만 둔다.
    if (out / "vlm_result.json").exists():
        shutil.move(str(out / "vlm_result.json"), work / "vlm_result.json")
    print(f"[{name}] -> {glb.name} {meta.name} {png.name if png else '(no teaser)'}")
    write_sources()


def apply(name: str) -> None:
    """OVERRIDES 의 재질을 JSON 에 덮어쓴다. 범위는 modules_metaobj.ranges_for 가 정한다."""
    sys.path.insert(0, str(_REPO / "modules" / "behavior"))
    from utils.config import MATERIAL_PROPERTY_PRIORS

    out = OUT_ROOT / name
    meta = out / f"{name}.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    ov = OVERRIDES.get(name, {})
    log = []
    for part in data["parts"]:
        mat = ov.get(part["objectName"])
        if mat is None:
            continue
        # 금속이면 behavior 사전의 범위를 VLM 결과 모양으로 넘겨 같은 경로로 9 개를 만든다
        prior = MATERIAL_PROPERTY_PRIORS.get(mat)
        fake = ({k: list(v) for k, v in prior.items()} | {"material": mat}) if prior else None
        norm, ranges = ranges_for(mat, fake)
        if norm != part["material"]:
            log.append(f"{part['objectName']} {part['material']}->{norm}")
        part["material"] = norm
        part["rangeProperties"] = ranges
    meta.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{name}] 재질 덮어쓰기 {len(log)}건: " + ", ".join(log))
    write_sources()


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    est = None
    if cmd in ("build", "all"):
        from modules_behavior import BehaviorPropertyEstimator
        est = BehaviorPropertyEstimator()
    for n in names:
        if cmd in ("build", "all"):
            build(n, est)
        if cmd in ("apply", "all"):
            apply(n)
