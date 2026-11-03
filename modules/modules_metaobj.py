"""합본 GLB 를 metaobj(.glb + .json + 티저 .png)로 내보낸다.

main_metaobjrecon.py 의 ENTER 단계 뒤쪽이다. metaobj_wrapper/combine_rocket_glb.py 가 합본과
JSON 틀을 만들고, modules_behavior.py 가 재질·어포던스·물성을 추정한다. 이 파일은 둘을
잇는다. behavior 결과를 JSON 틀이 받는 모양으로 바꾸고, 파츠를 **이름으로** 맞춘다.

파츠를 이름으로 맞추는 이유: behavior 는 trimesh 의 nodes_geometry 순회 순서로 partId 를
매기는데, 그 순서가 control_parts 인덱스와 같다는 보장이 없다. 실제로 5 파츠 합본에서
partId 1 이 control_parts4 였다. 순서로 매핑하면 재질이 뒤집힌다. behavior 는 렌더할 때
partId 와 GLB 노드 이름을 part_manifest.json 에 남기고 vlm_result.json 의 objectName 에
노드 이름을 넣는다. 여기서는 그 objectName 으로 맞춘다.

단독 실행:
    python modules/modules_metaobj.py --parts a.glb b.glb --transforms t.json --out dir --name NAME
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "metaobj_wrapper"))
from combine_rocket_glb import (OBJECT_NAME_PREFIX, combine_parts,       # noqa: E402
                                load_part_specs, specs_for_parts, write_metadata_json)

# HL2 가 읽는 라벨. 이 이름 그대로, 이 순서로 나간다.
PROPERTY_LABELS = [
    "youngs_modulus_GPa", "density_g_cm3", "poissons_ratio", "tensile_strength_MPa",
    "hardness_HV", "thermal_conductivity_W_mK",
    "electrical_conductivity_MS_m", "thermal_expansion_coeff_1e-6_K", "fracture_toughness_MPa_sqrt_m",
]

# VLM 이 금속 사전에 없는 재질(유리·고무·플라스틱 등)을 내면 behavior 는 물성 범위를 비워
# 보낸다. 그 파츠만 여기서 채운다. 9 개 전부, PROPERTY_LABELS 순서. 임시값이다.
NONMETAL_PRIORS = {
    #            E GPa          g/cm3        poisson       tensile MPa   HV            k W/mK       elec MS/m  exp 1e-6/K   K_IC
    "wood":     [(9, 13),       (0.6, 0.9),  (0.3, 0.4),   (50, 120),    (1, 5),       (0.1, 0.2),  (0, 0),    (5, 40),     (0.2, 0.6)],
    "plastic":  [(1.8, 2.5),    (1.02, 1.2), (0.35, 0.4),  (30, 55),     (5, 15),      (0.15, 0.3), (0, 0),    (60, 100),   (2, 4)],
    "rubber":   [(0.001, 0.01), (0.9, 1.2),  (0.48, 0.5),  (10, 25),     (0.01, 0.1),  (0.1, 0.2),  (0, 0),    (150, 250),  (0.1, 0.5)],
    "glass":    [(65, 75),      (2.4, 2.6),  (0.2, 0.25),  (30, 90),     (450, 600),   (0.8, 1.2),  (0, 0),    (8, 9),      (0.7, 0.9)],
    "leather":  [(0.1, 0.5),    (0.8, 1.0),  (0.3, 0.45),  (10, 30),     (0.01, 0.1),  (0.1, 0.2),  (0, 0),    (50, 150),   (0.5, 2)],
    "fabric":   [(0.05, 0.5),   (0.3, 1.0),  (0.2, 0.4),   (5, 50),      (0.01, 0.1),  (0.03, 0.1), (0, 0),    (50, 200),   (0.1, 1)],
    "ceramic":  [(300, 400),    (3.8, 4.0),  (0.2, 0.25),  (200, 350),   (1500, 2000), (20, 35),    (0, 0),    (7, 9),      (3, 5)],
    "paper":    [(2, 4),        (0.7, 0.9),  (0.2, 0.35),  (20, 60),     (0.01, 0.1),  (0.05, 0.1), (0, 0),    (5, 30),     (0.1, 0.5)],
    "stone":    [(50, 70),      (2.6, 2.8),  (0.2, 0.3),   (5, 20),      (100, 200),   (2, 3),      (0, 0),    (5, 10),     (0.5, 1.5)],
}
_GENERIC_ALL = [(1, 100), (0.5, 8), (0.2, 0.45), (10, 500), (1, 500), (0.1, 50), (0, 30), (5, 100), (0.5, 50)]
_ALIASES = {"aluminum": "aluminium", "steel": "carbon_steel", "iron": "cast_iron",
            "polymer": "plastic", "abs": "plastic", "pla": "plastic", "nylon": "plastic",
            "polyurethane": "plastic", "foam": "plastic",
            "cloth": "fabric", "textile": "fabric", "upholstery_fabric": "fabric", "velvet": "fabric",
            "timber": "wood", "plywood": "wood", "cardboard": "paper",
            "marble": "stone", "granite": "stone", "concrete": "stone"}


def _metal_priors() -> dict:
    """behavior 의 금속 사전. sys.path 를 건드리지 않으려고 파일 경로로 읽는다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_behavior_config", _REPO / "modules" / "behavior" / "utils" / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MATERIAL_PROPERTY_PRIORS


def ranges_for(material: str, vlm_part: dict | None = None) -> tuple[str, list[dict]]:
    """재질 이름 -> (정규화한 이름, PROPERTY_LABELS 순서의 9 개 범위).

    VLM 이 범위를 준 금속이면 그 9 개. VLM 이 'steel' 처럼 사전 밖 이름을 내면
    별칭으로 사전 금속에 맞춰 사전 범위를 쓴다. 비금속은 NONMETAL_PRIORS, 그것도 없으면
    아주 넓은 공통 범위다. 어느 경우든 9 개가 나간다.
    """
    name = (material or "").strip().lower().replace(" ", "_")
    name = _ALIASES.get(name, name)
    vals: list[tuple[float, float]] = []
    if vlm_part is not None:
        for label in PROPERTY_LABELS:
            v = vlm_part.get(label)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                vals.append((float(v[0]), float(v[1])))
    if len(vals) != len(PROPERTY_LABELS):
        metals = _metal_priors()
        if name in metals:
            vals = [tuple(map(float, metals[name][l])) for l in PROPERTY_LABELS]
        elif name in NONMETAL_PRIORS:
            vals = NONMETAL_PRIORS[name]
        else:
            vals = _GENERIC_ALL
    return name, [{"label": l, "min": float(lo), "max": float(hi)} for l, (lo, hi) in zip(PROPERTY_LABELS, vals)]


def _to_metadata(part: dict) -> dict:
    material, ranges = ranges_for(part.get("material", ""), part)
    if not any(isinstance(part.get(l), (list, tuple)) for l in PROPERTY_LABELS):
        src = ("금속 사전(별칭)" if material in _metal_priors() else
               "NONMETAL_PRIORS" if material in NONMETAL_PRIORS else "공통 범위")
        print(f"[metaobj] partId={part.get('partId')} VLM 재질 {part.get('material')!r} -> {material!r}, "
              f"범위는 {src}", flush=True)
    return {"material": material, "affordance": part.get("affordance", ""), "rangeProperties": ranges}


def properties_from_vlm(vlm_json, n_parts: int) -> list[dict]:
    """vlm_result.json -> control_parts 인덱스 순서의 metadata 목록. objectName 으로 맞춘다."""
    data = json.loads(Path(vlm_json).read_text(encoding="utf-8"))
    expected = [f"{OBJECT_NAME_PREFIX}{i}" for i in range(n_parts)]
    names = [p.get("objectName") for p in data["parts"]]
    if sorted(map(str, names)) != sorted(expected):
        raise RuntimeError(f"VLM 결과의 objectName 이 합본 노드와 다르다: {names} vs {expected}")
    by_node = dict(zip(names, data["parts"]))
    return [_to_metadata(by_node[n]) for n in expected]


def write_teaser(render_png, out_png, margin: float = 0.1, max_side: int = 1024) -> None:
    """behavior 의 전체 씬 렌더에서 물체 영역만 잘라 티저로 저장한다. 배경은 검정이다."""
    im = cv2.imread(str(render_png))
    ys, xs = np.where(im.max(axis=2) > 8)
    if len(xs) == 0:
        shutil.copy(render_png, out_png)
        return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(margin * max(x1 - x0, y1 - y0))
    h, w = im.shape[:2]
    crop = im[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    s = max_side / max(crop.shape[:2])
    if s < 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), crop)


def export_metaobj(part_files, transforms_json, out_dir, name: str, estimator=None, work_dir=None):
    """합본 -> (behavior) -> JSON -> 티저. (glb, json, png 또는 None) 을 돌려준다.

    estimator 가 없거나 실패하면 JSON 은 combine_rocket_glb 의 상수 기본값으로 나가고 티저는
    없다. 실패는 크게 찍되 GLB 와 JSON 은 낸다. HL2 가 받을 것이 있어야 한다.
    """
    part_files = [Path(p) for p in part_files]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb, meta, png = out_dir / f"{name}.glb", out_dir / f"{name}.json", out_dir / f"{name}.png"

    specs = specs_for_parts(load_part_specs(Path(transforms_json)))
    combine_parts(part_files, specs, glb, root_name=name)

    teaser = None
    if estimator is not None:
        work = Path(work_dir or out_dir / "vlm")
        try:
            vlm_json = estimator.run(str(glb), output_json=str(out_dir / "vlm_result.json"),
                                     vlm_input_dir=str(work))
            for spec, props in zip(specs, properties_from_vlm(vlm_json, len(part_files))):
                spec["metadata"] = props
            render = work / name / "images" / "001.png"
            if render.exists():
                write_teaser(render, png)
                teaser = png
            else:
                print(f"[metaobj] 렌더가 없어 티저를 못 만든다: {render}", flush=True)
        except Exception:
            line = "!" * 70
            print(f"\n{line}\n!!  behavior 실패. JSON 은 상수 기본값으로 나간다\n{line}", flush=True)
            traceback.print_exc()
            for spec in specs:
                spec.pop("metadata", None)

    write_metadata_json(meta, specs)
    return glb, meta, teaser


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GLB 파츠들 -> metaobj (.glb + .json + 티저 .png)")
    ap.add_argument("--parts", nargs="+", required=True, help="mesh_0.glb mesh_1.glb ... 순서대로")
    ap.add_argument("--transforms", required=True, help="transforms.json")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--name", required=True, help="파일 이름이자 GLB 루트 노드 이름")
    ap.add_argument("--no-behavior", action="store_true", help="VLM 없이 상수 기본값으로")
    ap.add_argument("--vlm_type", default="qwen", choices=["qwen", "gpt4"],
                    help="qwen = 연구실 서버 Qwen3.8-27B, gpt4 = OpenAI gpt-4o-mini")
    args = ap.parse_args()

    est = None
    if not args.no_behavior:
        sys.path.insert(0, str(_REPO / "modules"))
        from modules_behavior import BehaviorPropertyEstimator
        est = BehaviorPropertyEstimator(vlm_type=args.vlm_type)
    g, j, t = export_metaobj(args.parts, args.transforms, args.out, args.name, est)
    print(f"[metaobj] {g}\n[metaobj] {j}\n[metaobj] {t or '티저 없음'}")
