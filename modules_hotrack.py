import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import ultralytics


_REPO_ROOT = Path(__file__).resolve().parent
_SAM2_ROOT = _REPO_ROOT / "segmentor" / "sam2_realtime"
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))

from sam2.build_sam import build_sam2_realtime_predictor  # noqa: E402

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency
    mp = None


@dataclass
class TrackEntry:
    id: int
    kind: str
    class_id: int
    score: float
    box: List[int]
    prompt_frame: int
    last_seen_frame: int
    deleted: bool = False


class InteractiveHoTrackSegmentor:
    """Online stage-1 HoTrack module.

    This class only detects hand-object interactions and tracks the resulting
    masks. It intentionally does not merge/split masks, reassign ids, build
    graphs, or detect events.
    """

    CONFIGS = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    }

    def __init__(
        self,
        output_dir: str = "output/hotrack_stage1",
        video_name: str = "hl2_online",
        yolo_model_path: str = "segmentor/100DOH_small.pt",
        sam2_variant: str = "tiny",
        sam2_checkpoint: str = "",
        max_side: int = 960,
        detect_interval: int = 3,
        yolo_conf: float = 0.25,
        iou_threshold: float = 0.15,
        max_active_objects: int = 5,
        max_active_hands: int = 2,
        max_new_objects_per_frame: int = 2,
        object_class_ids: Sequence[int] = (1,),
        hand_class_ids: Sequence[int] = (0,),
        track_hands: bool = False,
        include_hands: bool = False,
        hand_backend: str = "auto",
        save_masks: bool = True,
        interactive_window: bool = True,
        window_name: str = "Interactive HoTrack",
        use_cuda: Optional[bool] = None,
        amp_dtype: str = "bfloat16",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        state_window: int = 9,
        keep_cond_frames_per_obj: int = 2,
        log_memory: bool = False,
    ):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = _REPO_ROOT / self.output_dir
        self.video_name = video_name
        self.run_dir = self.output_dir / video_name
        self.frames_dir = self.run_dir / "frames"
        self.masks_dir = self.run_dir / "masks_png"
        self.overlays_dir = self.run_dir / "overlays"
        self.save_masks = bool(save_masks)
        if self.save_masks:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            self.overlays_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if (torch.cuda.is_available() and use_cuda is not False) else "cpu"
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.max_side = int(max_side)
        self.detect_interval = max(1, int(detect_interval))
        self.yolo_conf = float(yolo_conf)
        self.iou_threshold = float(iou_threshold)
        self.max_active_objects = int(max_active_objects)
        self.max_active_hands = int(max_active_hands)
        self.max_new_objects_per_frame = int(max_new_objects_per_frame)
        self.object_class_ids = {int(v) for v in object_class_ids}
        self.hand_class_ids = {int(v) for v in hand_class_ids}
        self.track_hands = bool(track_hands)
        self.include_hands = bool(include_hands)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.state_window = max(1, int(state_window))
        self.keep_cond_frames_per_obj = max(1, int(keep_cond_frames_per_obj))
        self.log_memory = bool(log_memory)
        self.interactive_window = bool(interactive_window)
        self.window_name = window_name
        self.quit_requested = False

        self.model_cfg, checkpoint_path = self._resolve_sam2_paths(sam2_variant, sam2_checkpoint)
        self.tracker = build_sam2_realtime_predictor(self.model_cfg, checkpoint_path, device=self.device)
        self.yolo_model = ultralytics.YOLO(str(self._abs_path(yolo_model_path)), verbose=False)

        self.hand_backend = self._select_hand_backend(hand_backend)
        self.hands = None
        if self.hand_backend == "mediapipe":
            self.hands = mp.solutions.hands.Hands(max_num_hands=self.max_active_hands, model_complexity=0)

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.inference_state: Optional[dict] = None
        self.frame_idx = 0
        self.current_frame_idx = -1
        self.all_ids: List[int] = []
        self.tracks: Dict[int, TrackEntry] = {}
        self.masks: Dict[int, np.ndarray] = {}
        self.colors: Dict[int, Tuple[int, int, int]] = {}
        self.next_object_id = 100
        self.next_hand_id = 1
        self.selected_id: Optional[int] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_vis: Optional[np.ndarray] = None
        self.current_scale = 1.0
        self.orig_size: Tuple[int, int] = (0, 0)
        self.brush_radius = 18
        self.editing = False
        self.edit_mask: Optional[np.ndarray] = None
        self._drawing = False
        self._erase = False

        self._write_meta()
        if self.interactive_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(self.window_name, self._on_mouse)
            except cv2.error as exc:
                print(f"[HoTrack] Interactive window disabled: {exc}")
                self.interactive_window = False

    def _abs_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _resolve_sam2_paths(self, variant: str, checkpoint: str) -> Tuple[str, str]:
        variant = variant.lower()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown SAM2 variant '{variant}'. Choose one of {sorted(self.CONFIGS)}")
        cfg, default_ckpt = self.CONFIGS[variant]
        ckpt_path = Path(checkpoint) if checkpoint else _REPO_ROOT / "segmentor" / "sam2_realtime" / "checkpoints" / default_ckpt
        if not ckpt_path.is_absolute():
            ckpt_path = _REPO_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {ckpt_path}. "
                "Put the checkpoint there or set UVR_HOTRACK_SAM2_CHECKPOINT."
            )
        return cfg, str(ckpt_path)

    def _select_hand_backend(self, requested: str) -> str:
        requested = requested.lower()
        has_mp_solutions = mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if requested == "mediapipe" and not has_mp_solutions:
            raise RuntimeError("mediapipe.solutions.hands is unavailable; use hand_backend='yolo'.")
        if requested == "auto":
            return "mediapipe" if has_mp_solutions else "yolo"
        if requested not in {"mediapipe", "yolo", "none"}:
            raise ValueError("hand_backend must be one of auto, mediapipe, yolo, none")
        return requested

    def _write_meta(self) -> None:
        if not self.save_masks:
            return
        meta = {
            "pipeline": "interactive_hotrack_stage1",
            "video_name": self.video_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Stage-1 only: online hand-object tracking, no merge/split/re-id/graph/event logic.",
        }
        (self.run_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _resize_for_tracking(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self.orig_size = (h, w)
        if self.max_side <= 0 or max(h, w) <= self.max_side:
            self.current_scale = 1.0
            return frame_bgr
        self.current_scale = self.max_side / float(max(h, w))
        new_w = int(round(w * self.current_scale))
        new_h = int(round(h * self.current_scale))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _upscale_mask(self, mask: np.ndarray) -> np.ndarray:
        orig_h, orig_w = self.orig_size
        if mask.shape[:2] == (orig_h, orig_w):
            return mask.astype(np.uint8)
        return cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    @torch.inference_mode()
    def process_frame(self, frame_bgr: np.ndarray):
        """Process one BGR frame and return (overlay, object_masks, result_dict)."""
        self.current_frame = self._resize_for_tracking(frame_bgr.copy())
        frame = self.current_frame
        h, w = frame.shape[:2]
        self.current_frame_idx = self.frame_idx

        with self._maybe_amp():
            if self.inference_state is None:
                self.inference_state = self.tracker.init_state(
                    frame,
                    offload_video_to_cpu=self.offload_video_to_cpu,
                    offload_state_to_cpu=self.offload_state_to_cpu,
                )
            else:
                self.inference_state = self.tracker.add_frame(
                    self.inference_state,
                    frame,
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )

            self.masks = self._track_existing_masks(h, w)
            should_detect = self.frame_idx == 0 or self.frame_idx % self.detect_interval == 0 or not self.active_object_ids()
            detections = self._detect(frame) if should_detect else []
            hands, objects = self._split_detections(frame, detections)
            interactions = self._match_interactions(hands, objects)
            new_ids = self._add_interactions(interactions, h, w)
            self._prune_state()

        self.current_vis = self.render(frame)
        self._save_current_frame(new_ids, interactions)
        result = self._frame_result(new_ids, interactions)
        object_masks = [self._upscale_mask(self.masks[tid]) for tid in self.active_object_ids() if tid in self.masks]
        vis = self.current_vis
        if vis is not None and vis.shape[:2] != self.orig_size:
            vis = cv2.resize(vis, (self.orig_size[1], self.orig_size[0]), interpolation=cv2.INTER_LINEAR)

        if self.interactive_window and self.current_vis is not None:
            cv2.imshow(self.window_name, self.current_vis)

        if self.log_memory and self.device == "cuda":
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[HoTrack] frame={self.frame_idx} ids={self.all_ids} gpu={alloc:.2f}/{reserved:.2f}GB")

        self.frame_idx += 1
        return vis, object_masks, result

    def _maybe_amp(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return torch.autocast("cpu", enabled=False)

    def _track_existing_masks(self, height: int, width: int) -> Dict[int, np.ndarray]:
        masks: Dict[int, np.ndarray] = {}
        if not self.inference_state or not self.all_ids:
            return masks
        _, obj_ids, logits, self.inference_state = self.tracker.get_mask(self.inference_state, self.frame_idx)
        self.all_ids = [int(v) for v in obj_ids]
        if logits is None:
            return masks
        for idx, track_id in enumerate(self.all_ids):
            mask = self._logit_to_mask(logits, idx, height, width)
            masks[int(track_id)] = mask
            if int(track_id) in self.tracks:
                self.tracks[int(track_id)].last_seen_frame = self.frame_idx
        return masks

    @staticmethod
    def _logit_to_mask(mask_logits, idx: int, height: int, width: int) -> np.ndarray:
        mask = (mask_logits[idx] > 0).detach().cpu().numpy()
        mask = np.squeeze(mask).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _detect(self, frame: np.ndarray) -> List[dict]:
        results = self.yolo_model(frame, conf=self.yolo_conf, verbose=False)
        detections: List[dict] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].detach().cpu().numpy().tolist()]
                cls = int(box.cls[0].item())
                score = float(box.conf[0].item()) if hasattr(box, "conf") else 1.0
                detections.append({"class_id": cls, "score": score, "box": self._clip_box([x1, y1, x2, y2], frame.shape[:2])})
        return detections

    def _split_detections(self, frame: np.ndarray, detections: List[dict]) -> Tuple[List[dict], List[dict]]:
        objects = [d for d in detections if d["class_id"] in self.object_class_ids]
        if self.hand_backend == "mediapipe":
            hands = self._mediapipe_hand_boxes(frame)
            if hands:
                return hands[: self.max_active_hands], objects
        if self.hand_backend == "none":
            return [], objects
        hands = [d for d in detections if d["class_id"] in self.hand_class_ids]
        return hands[: self.max_active_hands], objects

    def _mediapipe_hand_boxes(self, frame: np.ndarray) -> List[dict]:
        if self.hands is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not getattr(results, "multi_hand_landmarks", None):
            return []
        h, w = frame.shape[:2]
        boxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.float32)
            x1, y1 = np.floor(pts.min(axis=0)).astype(int)
            x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
            pad = int(0.08 * max(x2 - x1, y2 - y1, 1))
            boxes.append({"class_id": 0, "score": 1.0, "box": self._clip_box([x1 - pad, y1 - pad, x2 + pad, y2 + pad], (h, w))})
        return boxes

    @staticmethod
    def _clip_box(box: Sequence[int], shape_hw: Tuple[int, int]) -> List[int]:
        h, w = shape_hw
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return [x1, y1, x2, y2]

    def _match_interactions(self, hands: List[dict], objects: List[dict]) -> List[dict]:
        if not hands or not objects:
            return []
        candidates = []
        for hand in hands:
            for obj in objects:
                iou = self._box_iou(hand["box"], obj["box"])
                distance_score = self._distance_score(hand["box"], obj["box"])
                score = max(iou, distance_score * 0.5)
                if iou >= self.iou_threshold or distance_score > 0:
                    candidates.append({"hand": hand, "object": obj, "score": float(score), "iou": float(iou)})
        candidates.sort(key=lambda v: v["score"], reverse=True)
        used_objects = set()
        interactions = []
        for c in candidates:
            obj_key = tuple(c["object"]["box"])
            if obj_key in used_objects:
                continue
            used_objects.add(obj_key)
            interactions.append(c)
            if len(interactions) >= self.max_new_objects_per_frame:
                break
        return interactions

    @staticmethod
    def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0 else inter / denom

    @staticmethod
    def _distance_score(a: Sequence[int], b: Sequence[int]) -> float:
        ax = (a[0] + a[2]) * 0.5
        ay = (a[1] + a[3]) * 0.5
        bx = (b[0] + b[2]) * 0.5
        by = (b[1] + b[3]) * 0.5
        diag = max(((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5, 1.0)
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return max(0.0, 1.0 - dist / (diag * 1.25))

    def _add_interactions(self, interactions: List[dict], height: int, width: int) -> List[int]:
        added: List[int] = []
        if self.inference_state is None:
            return added
        active_obj_count = len(self.active_object_ids())
        for item in interactions:
            if active_obj_count >= self.max_active_objects:
                break
            box = item["object"]["box"]
            if self._box_overlaps_existing_mask(box):
                continue
            track_id = self.next_object_id
            self.next_object_id += 1
            _, obj_ids, logits, self.inference_state = self.tracker.add_new_points_or_box(
                self.inference_state,
                frame_idx=self.frame_idx,
                obj_id=track_id,
                box=np.array(box, dtype=np.float32),
            )
            self.all_ids = [int(v) for v in obj_ids]
            if logits is not None:
                mask_idx = self.all_ids.index(track_id)
                self.masks[track_id] = self._logit_to_mask(logits, mask_idx, height, width)
            self.tracks[track_id] = TrackEntry(
                id=track_id,
                kind="object",
                class_id=int(item["object"].get("class_id", 1)),
                score=float(item["object"].get("score", item.get("score", 1.0))),
                box=[int(v) for v in box],
                prompt_frame=self.frame_idx,
                last_seen_frame=self.frame_idx,
            )
            added.append(track_id)
            active_obj_count += 1

        if self.track_hands:
            added.extend(self._add_hands([item["hand"] for item in interactions], height, width))
        return added

    def _add_hands(self, hands: List[dict], height: int, width: int) -> List[int]:
        added = []
        if self.inference_state is None:
            return added
        active_hands = len([tid for tid, t in self.tracks.items() if t.kind == "hand" and not t.deleted])
        for hand in hands:
            if active_hands >= self.max_active_hands:
                break
            box = hand["box"]
            track_id = self.next_hand_id
            self.next_hand_id += 1
            _, obj_ids, logits, self.inference_state = self.tracker.add_new_points_or_box(
                self.inference_state,
                frame_idx=self.frame_idx,
                obj_id=track_id,
                box=np.array(box, dtype=np.float32),
            )
            self.all_ids = [int(v) for v in obj_ids]
            if logits is not None:
                mask_idx = self.all_ids.index(track_id)
                self.masks[track_id] = self._logit_to_mask(logits, mask_idx, height, width)
            self.tracks[track_id] = TrackEntry(
                id=track_id,
                kind="hand",
                class_id=int(hand.get("class_id", 0)),
                score=float(hand.get("score", 1.0)),
                box=[int(v) for v in box],
                prompt_frame=self.frame_idx,
                last_seen_frame=self.frame_idx,
            )
            added.append(track_id)
            active_hands += 1
        return added

    def _box_overlaps_existing_mask(self, box: Sequence[int]) -> bool:
        x1, y1, x2, y2 = box
        for track_id in self.active_object_ids():
            mask = self.masks.get(track_id)
            if mask is None:
                continue
            crop = mask[y1:y2, x1:x2]
            if crop.size and float(crop.mean()) > 0.15:
                return True
        return False

    def active_object_ids(self) -> List[int]:
        return [tid for tid, t in self.tracks.items() if t.kind == "object" and not t.deleted]

    def _prune_state(self) -> None:
        if self.inference_state is None:
            return
        min_frame = max(0, self.frame_idx - self.state_window + 1)
        if hasattr(self.tracker, "clear_old_frames"):
            self.tracker.clear_old_frames(self.inference_state, min_frame)
        for obj_idx, out_dict in list(self.inference_state.get("output_dict_per_obj", {}).items()):
            cond = out_dict.get("cond_frame_outputs", {})
            if len(cond) <= self.keep_cond_frames_per_obj:
                continue
            keep = set(sorted(cond.keys())[-self.keep_cond_frames_per_obj:])
            for key in list(cond.keys()):
                if key not in keep:
                    cond.pop(key, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        if frame is None:
            frame = self.current_frame
        if frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        vis = frame.copy()
        for track_id in sorted(self.masks):
            track = self.tracks.get(track_id)
            if track is None or track.deleted:
                continue
            if track.kind == "hand" and not self.include_hands:
                continue
            mask = self.masks[track_id]
            color = self._color(track_id)
            overlay = vis.copy()
            overlay[mask.astype(bool)] = color
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
            if bw > 0 and bh > 0:
                thickness = 3 if track_id == self.selected_id else 2
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, thickness)
                cv2.putText(vis, f"{track.kind}:{track_id}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if self.editing and self.edit_mask is not None:
            edit_overlay = vis.copy()
            edit_overlay[self.edit_mask.astype(bool)] = (0, 255, 255)
            vis = cv2.addWeighted(edit_overlay, 0.35, vis, 0.65, 0)
        help_text = "q quit | click select | d delete | D remove-frame | e edit | a apply | z cancel | +/- brush | s save"
        cv2.putText(vis, help_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, help_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return vis

    def _color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self.colors:
            rng = np.random.default_rng(track_id * 9973)
            self.colors[track_id] = tuple(int(v) for v in rng.integers(40, 240, size=3))
        return self.colors[track_id]

    def handle_key(self, key: int) -> bool:
        if key in (-1, 255):
            return False
        key = key & 0xFF
        if key in (ord("q"), 27):
            self.quit_requested = True
            return True
        if key == ord("s"):
            self._save_current_frame([], [])
            return True
        if key == ord("d"):
            self.delete_active()
            return True
        if key == ord("D"):
            self.remove_active_current_frame()
            return True
        if key == ord("e"):
            self.start_edit()
            return True
        if key == ord("a"):
            self.apply_edit()
            return True
        if key == ord("z"):
            self.cancel_edit()
            return True
        if key in (ord("+"), ord("=")):
            self.brush_radius = min(100, self.brush_radius + 4)
            return True
        if key in (ord("-"), ord("_")):
            self.brush_radius = max(2, self.brush_radius - 4)
            return True
        if key == ord("["):
            self._cycle_selection(-1)
            return True
        if key == ord("]"):
            self._cycle_selection(1)
            return True
        return False

    def _cycle_selection(self, delta: int) -> None:
        ids = [tid for tid in sorted(self.masks) if self.tracks.get(tid) and not self.tracks[tid].deleted]
        if not ids:
            self.selected_id = None
            return
        if self.selected_id not in ids:
            self.selected_id = ids[0]
            return
        self.selected_id = ids[(ids.index(self.selected_id) + delta) % len(ids)]

    def delete_active(self) -> None:
        if self.selected_id is None or self.inference_state is None:
            return
        tid = int(self.selected_id)
        try:
            self.all_ids, _ = self.tracker.remove_object(self.inference_state, tid, strict=False)
        except Exception as exc:
            print(f"[HoTrack] remove_object failed for id={tid}: {exc}")
        self.masks.pop(tid, None)
        if tid in self.tracks:
            self.tracks[tid].deleted = True
        self._log_op({"op": "delete", "frame": self._current_output_frame_idx(), "id": tid})
        self.selected_id = None
        self.current_vis = self.render(self.current_frame)
        self._save_current_frame([], [])

    def remove_active_current_frame(self) -> None:
        if self.selected_id is None:
            return
        tid = int(self.selected_id)
        self.masks.pop(tid, None)
        self._log_op({"op": "remove_current_frame", "frame": self._current_output_frame_idx(), "id": tid})
        self.current_vis = self.render(self.current_frame)
        self._save_current_frame([], [])

    def start_edit(self) -> None:
        if self.selected_id is None or self.selected_id not in self.masks:
            return
        self.edit_mask = self.masks[self.selected_id].copy()
        self.editing = True

    def apply_edit(self) -> None:
        if not self.editing or self.edit_mask is None or self.selected_id is None or self.inference_state is None:
            return
        tid = int(self.selected_id)
        self.masks[tid] = self.edit_mask.astype(np.uint8)
        frame_idx = self._current_output_frame_idx()
        if hasattr(self.tracker, "add_new_mask"):
            _, obj_ids, logits = self.tracker.add_new_mask(
                self.inference_state,
                frame_idx=frame_idx,
                obj_id=tid,
                mask=self.edit_mask.astype(bool),
            )
            self.all_ids = [int(v) for v in obj_ids]
            if logits is not None and tid in self.all_ids:
                idx = self.all_ids.index(tid)
                h, w = self.current_frame.shape[:2]
                self.masks[tid] = self._logit_to_mask(logits, idx, h, w)
        else:
            points, labels = self._points_from_edit_mask(self.edit_mask)
            if points:
                _, obj_ids, logits, self.inference_state = self.tracker.add_new_points_or_box(
                    self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=tid,
                    points=np.asarray(points, dtype=np.float32),
                    labels=np.asarray(labels, dtype=np.int32),
                    clear_old_points=True,
                )
                self.all_ids = [int(v) for v in obj_ids]
                if logits is not None and tid in self.all_ids:
                    idx = self.all_ids.index(tid)
                    h, w = self.current_frame.shape[:2]
                    self.masks[tid] = self._logit_to_mask(logits, idx, h, w)
        self._log_op({"op": "edit_apply", "frame": frame_idx, "id": tid})
        self.cancel_edit()
        self.current_vis = self.render(self.current_frame)
        self._save_current_frame([], [])

    def cancel_edit(self) -> None:
        self.editing = False
        self.edit_mask = None
        self._drawing = False

    def _points_from_edit_mask(self, mask: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [], []
        points = []
        labels = []
        for qx, qy in [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]:
            idx = int(min(len(xs) - 1, max(0, round(qx * (len(xs) - 1)))))
            points.append([int(xs[idx]), int(ys[idx])])
            labels.append(1)
        return points, labels

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self.current_frame is None:
            return
        if self.editing and self.edit_mask is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawing = True
                self._erase = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
                self._paint(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
                self._paint(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._drawing = True
                self._erase = True
                self._paint(x, y)
            elif event == cv2.EVENT_RBUTTONUP:
                self._drawing = False
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_id = self._id_at(x, y)

    def _paint(self, x: int, y: int) -> None:
        if self.edit_mask is None:
            return
        value = 0 if self._erase else 1
        cv2.circle(self.edit_mask, (int(x), int(y)), self.brush_radius, value, -1)

    def _id_at(self, x: int, y: int) -> Optional[int]:
        for track_id in sorted(self.masks.keys(), reverse=True):
            mask = self.masks[track_id]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                return track_id
        return None


    def _current_output_frame_idx(self) -> int:
        return self.current_frame_idx if self.current_frame_idx >= 0 else self.frame_idx

    def _save_current_frame(self, new_ids: Sequence[int], interactions: Sequence[dict]) -> None:
        if not self.save_masks or self.current_frame is None:
            return
        tag = f"{self._current_output_frame_idx():06d}"
        frame_mask_dir = self.masks_dir / tag
        frame_mask_dir.mkdir(parents=True, exist_ok=True)
        for old_mask in frame_mask_dir.glob("id_*.png"):
            old_mask.unlink()
        mask_files = {}
        for track_id, mask in sorted(self.masks.items()):
            track = self.tracks.get(track_id)
            if track is None or track.deleted:
                continue
            if track.kind == "hand" and not self.include_hands:
                continue
            path = frame_mask_dir / f"id_{track_id:06d}.png"
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            mask_files[str(track_id)] = str(path)
        if self.current_vis is not None:
            cv2.imwrite(str(self.overlays_dir / f"{tag}.jpg"), self.current_vis)
        payload = self._frame_result(new_ids, interactions)
        payload["mask_files"] = mask_files
        (self.frames_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _frame_result(self, new_ids: Sequence[int], interactions: Sequence[dict]) -> dict:
        return {
            "frame_idx": int(self._current_output_frame_idx()),
            "active_ids": [int(v) for v in self.all_ids],
            "selected_id": self.selected_id,
            "new_ids": [int(v) for v in new_ids],
            "interactions": [
                {
                    "score": float(item.get("score", 0.0)),
                    "iou": float(item.get("iou", 0.0)),
                    "hand_box": [int(v) for v in item["hand"]["box"]],
                    "object_box": [int(v) for v in item["object"]["box"]],
                }
                for item in interactions
            ],
            "tracks": {str(k): asdict(v) for k, v in self.tracks.items() if not v.deleted},
        }

    def _log_op(self, payload: dict) -> None:
        if not self.save_masks:
            return
        with (self.run_dir / "ops_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.interactive_window:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
