"""합본 GLB 를 metaobj(.glb + .json + 티저 .png)로 내보낸다.

main_metaobjrecon.py 의 ENTER 단계 뒤쪽이다. metaobj_wrapper/combine_rocket_glb.py 가 합본과
JSON 틀을 만들고, modules_behavior.py 가 재질·어포던스·물성을 추정한다. 이 파일은 둘을
잇는다. behavior 결과를 JSON 틀이 받는 모양으로 바꾸고, 파츠를 **이름으로** 맞춘다.

파츠를 이름으로 맞추는 이유: behavior 는 trimesh 의 nodes_geometry 순회 순서로 partId 를
매기는데, 그 순서가 control_parts 인덱스와 같다는 보장이 없다. 실제로 5 파츠 합본에서
partId 1 이 control_parts4 였다. 순서로 매핑하면 재질이 뒤집힌다.

behavior 가 아직 내지 않는 세 물성(전기전도도·열팽창계수·파괴인성)은 재질별 상수표로
채운다. 모듈이 그 셋을 내게 되면 MISSING_PROPERTY_PRIORS 와 그 분기를 지우면 된다. JSON
안에서는 추정값과 구분되지 않으므로 실행 로그에 남긴다.

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
import trimesh

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
_VLM_KEY = {"hardness_HV": "hardness"}          # behavior 쪽 이름이 다른 것
_N_FROM_VLM = 6                                  # 앞 6 개는 behavior 가 낸다

# behavior 가 아직 내지 않는 3 개. 재질별 (전기전도도 MS/m, 열팽창 1e-6/K, 파괴인성 MPa√m).
# 대략적인 교과서 범위이고 검증한 값이 아니다. 임시값이다.
MISSING_PROPERTY_PRIORS = {
    "carbon_steel":    ((5.0, 7.0),   (11.0, 13.0), (50.0, 120.0)),
    "stainless_steel": ((1.3, 1.5),   (16.0, 18.0), (100.0, 250.0)),
    "aluminium":       ((18.0, 38.0), (21.0, 24.0), (15.0, 45.0)),
    "copper":          ((55.0, 59.0), (16.5, 17.5), (30.0, 100.0)),
    "brass":           ((14.0, 16.0), (18.0, 21.0), (30.0, 80.0)),
    "bronze":          ((7.0, 10.0),  (17.0, 19.0), (30.0, 70.0)),
    "titanium":        ((0.5, 2.4),   (8.5, 9.5),   (50.0, 110.0)),
    "zinc_alloy":      ((15.0, 17.0), (26.0, 28.0), (20.0, 40.0)),
    "cast_iron":       ((1.0, 2.0),   (10.0, 12.0), (10.0, 25.0)),
}
_GENERIC_MISSING = ((1.0, 30.0), (10.0, 25.0), (20.0, 100.0))


def part_nodes_in_vlm_order(glb_path) -> list[str]:
    """behavior 가 partId 를 매기는 순서 그대로 노드 이름을 돌려준다. partId k -> [k-1].

    preprocess_glb_for_vlm.create_vlm_dataset_from_glb 의 순회를 그대로 따른다.
    로드 방식(force='scene')과 빈 geometry 를 건너뛰는 규칙까지 같아야 순서가 같다.
    """
    scene = trimesh.load(str(glb_path), force="scene")
    names = []
    for node_name in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph[node_name]
        if len(scene.geometry[geometry_name].vertices) == 0:
            continue
        names.append(node_name)
    return names


def _to_metadata(part: dict) -> dict:
    material = part.get("material", "")
    ranges = []
    for label in PROPERTY_LABELS[:_N_FROM_VLM]:
        v = part.get(_VLM_KEY.get(label, label))
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise RuntimeError(f"VLM 결과 partId={part.get('partId')} 에 {label} 범위가 없다: {v!r}")
        ranges.append({"label": label, "min": float(v[0]), "max": float(v[1])})
    for label, (lo, hi) in zip(PROPERTY_LABELS[_N_FROM_VLM:],
                               MISSING_PROPERTY_PRIORS.get(material, _GENERIC_MISSING)):
        ranges.append({"label": label, "min": lo, "max": hi})
    return {"material": material, "affordance": part.get("affordance", ""), "rangeProperties": ranges}


def properties_from_vlm(vlm_json, glb_path, n_parts: int) -> list[dict]:
    """vlm_result.json -> control_parts 인덱스 순서의 metadata 목록."""
    data = json.loads(Path(vlm_json).read_text(encoding="utf-8"))
    order = part_nodes_in_vlm_order(glb_path)
    expected = [f"{OBJECT_NAME_PREFIX}{i}" for i in range(n_parts)]
    if sorted(order) != sorted(expected):
        raise RuntimeError(f"합본 GLB 의 노드가 예상과 다르다: {order} vs {expected}")
    by_node = {}
    for p in data["parts"]:
        k = int(p["partId"])
        if not 1 <= k <= len(order):
            raise RuntimeError(f"partId {k} 가 노드 수 {len(order)} 를 벗어난다")
        by_node[order[k - 1]] = p
    missing = [n for n in expected if n not in by_node]
    if missing:
        raise RuntimeError(f"VLM 결과에 없는 파츠: {missing}")
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
            for spec, props in zip(specs, properties_from_vlm(vlm_json, glb, len(part_files))):
                spec["metadata"] = props
            print(f"[metaobj] 물성 {_N_FROM_VLM} 개는 VLM 추정, "
                  f"{PROPERTY_LABELS[_N_FROM_VLM:]} 는 재질별 임시값", flush=True)
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
    args = ap.parse_args()

    est = None
    if not args.no_behavior:
        sys.path.insert(0, str(_REPO / "modules"))
        from modules_behavior import BehaviorPropertyEstimator
        est = BehaviorPropertyEstimator()
    g, j, t = export_metaobj(args.parts, args.transforms, args.out, args.name, est)
    print(f"[metaobj] {g}\n[metaobj] {j}\n[metaobj] {t or '티저 없음'}")
