"""YOLO hand/object detector adapter for hograph_plus."""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


CONTACT_STATE_TEXT = {
    0: "No Contact",
    1: "Self Contact",
    2: "Another Person",
    3: "Portable Object",
    4: "Stationary Object",
}
CONTACT_STATE_CODE = {0: "N", 1: "S", 2: "O", 3: "P", 4: "F"}
CONTACT_CODE_TO_INT = {"N": 0, "S": 1, "O": 2, "P": 3, "F": 4}
BACKEND_YOLO = "yolo"


def _calculate_center(bb_xyxy: np.ndarray) -> np.ndarray:
    return np.array(
        [(bb_xyxy[0] + bb_xyxy[2]) / 2.0, (bb_xyxy[1] + bb_xyxy[3]) / 2.0],
        dtype=np.float32,
    )


def _match_objects(obj_dets: Optional[np.ndarray], hand_dets: Optional[np.ndarray]) -> List[int]:
    if obj_dets is None or obj_dets.size == 0:
        return [-1] * (hand_dets.shape[0] if hand_dets is not None else 0)
    if hand_dets is None or hand_dets.size == 0:
        return []

    obj_centers = np.stack([_calculate_center(obj_dets[j, :4]) for j in range(obj_dets.shape[0])], axis=0)
    matches: List[int] = []
    for i in range(hand_dets.shape[0]):
        contact_state = float(hand_dets[i, 5])
        if contact_state <= 0:
            matches.append(-1)
            continue
        hand_center = _calculate_center(hand_dets[i, :4])
        offset_mag = float(hand_dets[i, 6])
        offset_dx = float(hand_dets[i, 7])
        offset_dy = float(hand_dets[i, 8])
        point_cc = np.array(
            [hand_center[0] + offset_mag * 10000.0 * offset_dx, hand_center[1] + offset_mag * 10000.0 * offset_dy],
            dtype=np.float32,
        )
        dist = np.sum((obj_centers - point_cc[None, :]) ** 2, axis=1)
        matches.append(int(np.argmin(dist)))
    return matches


def _clip_box_xyxy(box_xyxy: np.ndarray, width: int, height: int) -> List[int]:
    if box_xyxy is None or len(box_xyxy) < 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, 0.0), float(width - 1))
    y2 = min(max(y2, 0.0), float(height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _normalize_label_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _is_target_object_class(class_name: str) -> bool:
    token = _normalize_label_name(class_name)
    return token in {"targetobject", "targetobj", "object", "obj", "target"}


def _contact_code_from_hand_class(class_name: str) -> Optional[str]:
    raw = str(class_name or "").strip()
    token = _normalize_label_name(raw)
    if token == "hand":
        return "P"
    match = re.match(r"^hand([nsofp])$", token)
    if not match:
        return None
    code = str(match.group(1)).upper()
    return code if code in CONTACT_CODE_TO_INT else None


def _extract_yolo_name_map(raw_names: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if isinstance(raw_names, dict):
        for key, value in raw_names.items():
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
        return out
    if isinstance(raw_names, (list, tuple)):
        for idx, value in enumerate(raw_names):
            out[int(idx)] = str(value)
    return out


def build_detector(
    *,
    use_cuda: bool = True,
    yolo_model_path: str = "",
) -> Tuple[Any, np.ndarray]:
    print(f"[HO][Build] backend=yolo use_cuda={bool(use_cuda)}")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("YOLO backend requires ultralytics. Install with: pip install ultralytics") from exc

    ckpt_path = str(yolo_model_path or os.environ.get("HO_YOLO_MODEL_PATH", "")).strip()
    if not ckpt_path:
        raise RuntimeError("YOLO backend selected, but no model path provided.")
    ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"YOLO checkpoint not found: {ckpt_path}")

    model = YOLO(ckpt_path)
    model_names = _extract_yolo_name_map(getattr(model, "names", {}))
    state = {
        "backend": BACKEND_YOLO,
        "model": model,
        "names": model_names,
        "frame_idx": 0,
        "_runtime_log_once": False,
        "_yolo_checkpoint_path": ckpt_path,
    }
    print(f"[HO][Build] runtime_backend=yolo(ultralytics) checkpoint={ckpt_path}")
    return state, np.asarray(["targetobject", "hand"], dtype=object)


def _detect_image_bgr_yolo(
    model_state: Dict[str, Any],
    image_bgr: np.ndarray,
    *,
    thresh_hand: float,
    thresh_obj: float,
) -> Dict[str, Any]:
    checkpoint = str(model_state.get("_yolo_checkpoint_path", ""))
    if not bool(model_state.get("_runtime_log_once", False)):
        print(
            f"[HO][Infer] runtime_backend=yolo(ultralytics) checkpoint={checkpoint}"
        )
        model_state["_runtime_log_once"] = True

    yolo_model = model_state.get("model", None)
    if yolo_model is None:
        raise RuntimeError("Invalid YOLO model state: missing 'model'")

    names_map = _extract_yolo_name_map(model_state.get("names", {}))
    if not names_map:
        names_map = _extract_yolo_name_map(getattr(yolo_model, "names", {}))
        model_state["names"] = names_map

    height, width = image_bgr.shape[:2]
    device = 0 if torch.cuda.is_available() else "cpu"
    start = time.time()
    results = yolo_model.predict(source=image_bgr, verbose=False, device=device)
    elapsed = float(time.time() - start)

    objects: List[Dict[str, Any]] = []
    hands: List[Dict[str, Any]] = []
    if results:
        result0 = results[0]
        boxes = getattr(result0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
            for i in range(len(cls)):
                class_id = int(cls[i])
                score = float(conf[i])
                class_name = str(names_map.get(class_id, str(class_id)))
                bbox = _clip_box_xyxy(xyxy[i], width=width, height=height)

                contact_code = _contact_code_from_hand_class(class_name)
                if contact_code is not None:
                    if score < float(thresh_hand):
                        continue
                    contact = int(CONTACT_CODE_TO_INT.get(contact_code, 3))
                    hands.append(
                        {
                            "bbox_xyxy": bbox,
                            "score": score,
                            "contact": contact,
                            "contact_text": CONTACT_STATE_TEXT.get(contact, "Unknown"),
                            "contact_code": str(CONTACT_STATE_CODE.get(contact, contact_code)),
                            "offset_mag": 0.0,
                            "offset_dir": [0.0, 0.0],
                            "lr": -1,
                            "class_name": class_name,
                        }
                    )
                    continue

                if _is_target_object_class(class_name):
                    if score < float(thresh_obj):
                        continue
                    objects.append({"bbox_xyxy": bbox, "score": score, "class_name": class_name})

    obj_dets = None
    if objects:
        obj_dets = np.asarray([[*obj["bbox_xyxy"], float(obj["score"])] for obj in objects], dtype=np.float32)
    hand_dets = None
    if hands:
        hand_dets = np.asarray(
            [
                [
                    *hand["bbox_xyxy"],
                    float(hand["score"]),
                    float(hand.get("contact", 0)),
                    float(hand.get("offset_mag", 0.0)),
                    float((hand.get("offset_dir") or [0.0, 0.0])[0]),
                    float((hand.get("offset_dir") or [0.0, 0.0])[1]),
                    float(hand.get("lr", -1)),
                ]
                for hand in hands
            ],
            dtype=np.float32,
        )
    matches = _match_objects(obj_dets, hand_dets)

    match_entries: List[Dict[str, Any]] = []
    for hand_idx, hand in enumerate(hands):
        obj_idx = matches[hand_idx] if hand_idx < len(matches) else -1
        obj = objects[obj_idx] if 0 <= obj_idx < len(objects) else None
        match_entries.append(
            {
                "hand_idx": int(hand_idx),
                "hand_bbox_xyxy": hand["bbox_xyxy"],
                "hand_score": float(hand["score"]),
                "contact": hand.get("contact"),
                "contact_text": hand.get("contact_text"),
                "contact_code": hand.get("contact_code"),
                "object_idx": int(obj_idx),
                "object_bbox_xyxy": obj["bbox_xyxy"] if obj else None,
                "object_score": obj["score"] if obj else None,
            }
        )

    return {
        "time_sec": elapsed,
        "objects": objects,
        "hands": hands,
        "hand_to_object": [int(x) for x in matches],
        "matches": match_entries,
        "backend": BACKEND_YOLO,
    }


@torch.inference_mode()
def detect_image_bgr(
    model: Any,
    image_bgr: np.ndarray,
    *,
    use_cuda: bool = True,
    thresh_hand: float = 0.5,
    thresh_obj: float = 0.5,
) -> Dict[str, Any]:
    del use_cuda
    if isinstance(model, dict) and str(model.get("backend", "")).strip().lower() == BACKEND_YOLO:
        return _detect_image_bgr_yolo(
            model,
            image_bgr,
            thresh_hand=float(thresh_hand),
            thresh_obj=float(thresh_obj),
        )
    raise RuntimeError("invalid detector state: expected the hograph_plus YOLO adapter.")
