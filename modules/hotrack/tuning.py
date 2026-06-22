from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _center_scale(base: float, span: float, x01: float) -> float:
    return float(base + (float(x01) - 0.5) * span)


@dataclass(frozen=True)
class MetaTuningAxes:
    profile: str = "balanced"
    det_sensitivity: float = 0.5
    dup_strictness: float = 0.5
    size_strictness: float = 0.5
    temporal_stability: float = 0.5
    graph_event_sensitivity: float = 0.5


PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "precision": {
        "det_sensitivity": 0.75,
        "dup_strictness": 0.85,
        "size_strictness": 0.80,
        "temporal_stability": 0.75,
        "graph_event_sensitivity": 0.80,
    },
    "balanced": {
        "det_sensitivity": 0.50,
        "dup_strictness": 0.50,
        "size_strictness": 0.50,
        "temporal_stability": 0.50,
        "graph_event_sensitivity": 0.50,
    },
    "recall": {
        "det_sensitivity": 0.25,
        "dup_strictness": 0.25,
        "size_strictness": 0.25,
        "temporal_stability": 0.30,
        "graph_event_sensitivity": 0.30,
    },
}


def make_meta_tuning(
    *,
    profile: str = "balanced",
    det_sensitivity: Optional[float] = None,
    dup_strictness: Optional[float] = None,
    size_strictness: Optional[float] = None,
    temporal_stability: Optional[float] = None,
    graph_event_sensitivity: Optional[float] = None,
) -> MetaTuningAxes:
    key = str(profile).strip().lower()
    if key not in PROFILE_PRESETS:
        key = "balanced"
    base = dict(PROFILE_PRESETS[key])

    if det_sensitivity is not None:
        base["det_sensitivity"] = _clip01(det_sensitivity)
    if dup_strictness is not None:
        base["dup_strictness"] = _clip01(dup_strictness)
    if size_strictness is not None:
        base["size_strictness"] = _clip01(size_strictness)
    if temporal_stability is not None:
        base["temporal_stability"] = _clip01(temporal_stability)
    if graph_event_sensitivity is not None:
        base["graph_event_sensitivity"] = _clip01(graph_event_sensitivity)

    return MetaTuningAxes(
        profile=key,
        det_sensitivity=float(base["det_sensitivity"]),
        dup_strictness=float(base["dup_strictness"]),
        size_strictness=float(base["size_strictness"]),
        temporal_stability=float(base["temporal_stability"]),
        graph_event_sensitivity=float(base["graph_event_sensitivity"]),
    )


def derive_hotrack_params(meta: MetaTuningAxes) -> Dict[str, Any]:
    x_dup = _clip01(meta.dup_strictness)
    x_size = _clip01(meta.size_strictness)
    x_time = _clip01(meta.temporal_stability)

    p: Dict[str, Any] = {
        # Detector confidence gates
        # Keep aligned with the production YOLO detector threshold defaults.
        "ho_thresh_hand": 0.5,
        "ho_thresh_obj": 0.5,
        # Duplicate/overlap strictness
        "dup_ioa_lo": _clip01(_center_scale(0.70, 0.30, x_dup)),
        "new_mask_reject_ioa": _clip01(_center_scale(0.40, 0.30, x_dup)),
        "new_mask_inclusion_reject_ioa": _clip01(_center_scale(0.85, 0.20, x_dup)),
        "post_track_dup_ioa_threshold": _clip01(_center_scale(0.80, 0.20, x_dup)),
        "inclusion_ioa_threshold": _clip01(_center_scale(0.70, 0.20, x_dup)),
        "hand_overlap_reject": _clip01(_center_scale(0.50, 0.20, x_dup)),
        "ambiguous_mask_ioa_threshold": _clip01(_center_scale(0.85, 0.10, x_dup)),
        # Size/shape strictness
        "max_obj_area_ratio": max(0.15, min(0.75, _center_scale(0.40, -0.40, x_size))),
        "max_obj_vs_hand_ratio": max(1.25, min(6.0, _center_scale(3.0, -3.0, x_size))),
        "min_mask_area": int(round(max(10.0, min(140.0, _center_scale(50.0, 80.0, x_size))))),
        # Temporal stability
        "ambiguous_confirm_frames": int(round(max(1.0, min(8.0, _center_scale(3.0, 4.0, x_time))))),
        "hand_static_frames": int(round(max(2.0, min(12.0, _center_scale(5.0, 4.0, x_time))))),
        "post_dup_confirm_frames": int(round(max(2.0, min(6.0, _center_scale(3.0, 3.0, x_time))))),
    }

    # Logical coupling: inclusion reject should not be lower than base duplicate gate.
    p["new_mask_inclusion_reject_ioa"] = max(
        float(p["new_mask_inclusion_reject_ioa"]),
        min(0.98, float(p["dup_ioa_lo"]) + 0.10),
    )
    return p


def derive_hograph_assembly_params(meta: MetaTuningAxes) -> Dict[str, Any]:
    x_time = _clip01(meta.temporal_stability)
    x_graph = _clip01(meta.graph_event_sensitivity)

    return {
        "persist_window": int(round(max(3.0, min(16.0, _center_scale(8.0, 8.0, x_time))))),
        "persist_th": max(0.10, min(0.80, _center_scale(0.40, 0.30, x_graph))),
        "d_gate": max(8.0, min(40.0, _center_scale(20.0, -12.0, x_graph))),
        "attach_window": int(round(max(4.0, min(20.0, _center_scale(10.0, 8.0, x_time))))),
        "min_hold": int(round(max(2.0, min(20.0, _center_scale(8.0, 8.0, x_time))))),
        "static_iou_th": max(0.0, min(0.10, _center_scale(0.02, 0.04, x_graph))),
    }


def summarize_meta_tuning(meta: MetaTuningAxes) -> Dict[str, Any]:
    return {
        "meta_axes": asdict(meta),
        "derived_hotrack": derive_hotrack_params(meta),
        "derived_hograph": derive_hograph_assembly_params(meta),
    }
