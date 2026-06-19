import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracking.hotrack import Hotrack  # noqa: E402


CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def _abs_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else _REPO_ROOT / p


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _hograph_plus_fallback(rel_path: str | Path) -> Optional[Path]:
    root_raw = os.environ.get("HOGRAPH_PLUS_ROOT")
    roots = [Path(root_raw).expanduser()] if root_raw else [
        _REPO_ROOT.parent / "hograph_plus",
        _REPO_ROOT.parent.parent / "hograph_plus",
    ]
    for root in roots:
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    return None


def _sanitize_track_only_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    result["runtime_mode"] = "track_only"
    result["struct_events"] = []
    result["id_transitions"] = []
    result["id_remap"] = {}
    return result


def _sanitize_track_only_tracking_json(track_path: Path) -> None:
    if not track_path.is_file():
        return
    try:
        payload = json.loads(track_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    payload["runtime_mode"] = "track_only"
    payload["struct_events"] = []
    payload["id_transitions"] = []
    payload["id_remap"] = {}
    track_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_track_only_hotrack_runtime_flags(hotrack: Any) -> None:
    """Mirror hograph_plus HographPipelineRunner._apply_track_only_hotrack_runtime_flags."""
    hotrack.enable_structural_ops = False
    hotrack.track_hand_masks = False
    hotrack.skip_existing_object_box_prompts = True
    hotrack.use_hand_matched_det_boxes_only = True
    hotrack.use_all_object_boxes = False
    hotrack.enable_component_promote_from_det = True
    hotrack.replay_enable_structural_split_logic = False
    hotrack.attach_proxy_backfill_enabled = False
    hotrack.attach_proxy_signal_persist_frames = 0
    hotrack.merge_after_attach_cooldown_frames = 0
    hotrack.split_replay_debug = False
    hotrack.attach_replay_debug = False
    hotrack.max_new_per_frame = 0


class InteractiveHoTrackSegmentor:
    """Interactive online wrapper around hograph_plus track-only Hotrack.

    This intentionally uses Hotrack.process_frame_with_tracking() and disables
    structural graph/id-transition behavior the same way hograph_plus track-only
    mode does. The wrapper only adapts outputs for main_all.py and adds basic
    local interactive mask editing/deletion plus PNG overlay/mask saving.
    """

    def __init__(
        self,
        output_dir: str = "output/hotrack_stage1",
        video_name: str = "hl2_online",
        yolo_model_path: str = "weights/yolo_100doh_best.pt",
        sam2_variant: str = "tiny",
        sam2_checkpoint: str = "",
        image_dir: str = ".",
        max_side: int = 0,
        ho_thresh_hand: float = 0.55,
        ho_thresh_obj: float = 0.55,
        iou_threshold: float = 0.5,
        target_contact_code: str = "P",
        save_masks: bool = True,
        save_tracking_json: bool = True,
        interactive_window: bool = True,
        window_name: str = "Interactive HoTrack TrackOnly",
        use_cuda: Optional[bool] = None,
        sam2_amp: bool = True,
        sam2_amp_dtype: str = "bfloat16",
        sam2_backend: str = "online",
        backfill_window: int = 120,
        use_dino_id: bool = True,
        enable_id_recovery: bool = True,
        compute_dino_similarity: bool = False,
        dino_allow_download: bool = False,
        **_: Any,
    ):
        self.output_dir = _abs_repo_path(output_dir)
        self.video_name = str(video_name)
        self.run_dir = self.output_dir / self.video_name
        self.masks_dir = self.run_dir / "masks_png"
        self.overlays_dir = self.run_dir / "overlays"
        self.save_masks = bool(save_masks)
        self.save_tracking_json = bool(save_tracking_json)
        if self.save_masks:
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            self.overlays_dir.mkdir(parents=True, exist_ok=True)

        self.max_side = int(max_side or 0)
        self.iou_threshold = float(iou_threshold)
        self.target_contact_code = str(target_contact_code or "P")
        self.interactive_window = bool(interactive_window)
        self.window_name = str(window_name)
        self.quit_requested = False
        self.selected_id: Optional[int] = None
        self.editing = False
        self.edit_mask: Optional[np.ndarray] = None
        self.brush_radius = 18
        self._drawing = False
        self._erase = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_vis: Optional[np.ndarray] = None
        self.current_result: Dict[str, Any] = {}
        self.current_masks: Dict[int, np.ndarray] = {}
        self.current_frame_idx = -1
        self._orig_shape: Tuple[int, int] = (0, 0)

        sam2_cfg, default_ckpt = self._resolve_sam2_config(sam2_variant, sam2_checkpoint)
        yolo_path = _abs_repo_path(yolo_model_path)
        if not yolo_path.exists():
            fallback = _hograph_plus_fallback(Path("weights") / Path(yolo_model_path).name)
            if fallback is not None:
                yolo_path = fallback
        if not yolo_path.exists():
            raise FileNotFoundError(
                f"HoTrack YOLO checkpoint not found: {yolo_path}. "
                "Use weights/yolo_100doh_best.pt, set UVR_HOTRACK_YOLO_MODEL, or set HOGRAPH_PLUS_ROOT."
            )

        ckpt_path = Path(default_ckpt)
        if not ckpt_path.exists():
            fallback = _hograph_plus_fallback(Path("third_party") / "sam2_realtime" / "checkpoints" / ckpt_path.name)
            if fallback is not None:
                ckpt_path = fallback
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {ckpt_path}. "
                "Set UVR_HOTRACK_SAM2_CHECKPOINT, place the file under segmentor/sam2_realtime/checkpoints/, or set HOGRAPH_PLUS_ROOT."
            )

        self.hotrack = Hotrack(
            image_dir=str(_abs_repo_path(image_dir)),
            output_root=str(self.output_dir),
            backfill_window=max(1, int(backfill_window)),
            ho_thresh_hand=float(ho_thresh_hand),
            ho_thresh_obj=float(ho_thresh_obj),
            ho_yolo_model_path=str(yolo_path),
            sam2_checkpoint=str(ckpt_path),
            sam2_model_cfg=str(sam2_cfg),
            sam2_amp=bool(sam2_amp),
            sam2_amp_dtype=str(sam2_amp_dtype),
            sam2_backend=str(sam2_backend),
            use_cuda=(use_cuda is not False),
            use_dino_id=bool(use_dino_id),
            enable_id_recovery=bool(enable_id_recovery),
            compute_dino_similarity=bool(compute_dino_similarity),
            dino_allow_download=bool(dino_allow_download),
        )
        self.hotrack.save_tracking_json = bool(self.save_tracking_json)
        apply_track_only_hotrack_runtime_flags(self.hotrack)
        self.hotrack.set_output_dirs(str(self.output_dir), self.video_name, image_dir=str(_abs_repo_path(image_dir)))
        self.hotrack.reset_tracker_state()

        self._write_meta()
        if self.interactive_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(self.window_name, self._on_mouse)
            except cv2.error as exc:
                print(f"[HoTrackTrackOnly] Interactive window disabled: {exc}")
                self.interactive_window = False

    def _resolve_sam2_config(self, variant: str, checkpoint: str) -> Tuple[str, str]:
        key = str(variant or "tiny").strip().lower()
        if key not in CONFIGS:
            raise ValueError(f"Unknown SAM2 variant '{variant}'. Choose one of {sorted(CONFIGS)}")
        cfg, ckpt_name = CONFIGS[key]
        ckpt_path = Path(checkpoint).expanduser() if checkpoint else _REPO_ROOT / "segmentor" / "sam2_realtime" / "checkpoints" / ckpt_name
        if not ckpt_path.is_absolute():
            ckpt_path = _REPO_ROOT / ckpt_path
        return cfg, str(ckpt_path)

    def _write_meta(self) -> None:
        if not self.save_masks:
            return
        payload = {
            "pipeline": "hograph_plus_hotrack_track_only",
            "runtime_mode": "track_only",
            "video_name": self.video_name,
            "track_only_flags": self.track_only_flags(),
            "disabled": {
                "hograph": True,
                "external_structural_events": True,
                "struct_events": True,
                "id_transitions": True,
                "id_remap": True,
            },
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "track_only_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def track_only_flags(self) -> Dict[str, bool]:
        return {
            "enable_structural_ops": bool(getattr(self.hotrack, "enable_structural_ops", False)),
            "track_hand_masks": bool(getattr(self.hotrack, "track_hand_masks", True)),
            "skip_existing_object_box_prompts": bool(getattr(self.hotrack, "skip_existing_object_box_prompts", False)),
            "use_dino_id": bool(getattr(self.hotrack, "use_dino_id", False)),
            "enable_id_recovery": bool(getattr(self.hotrack, "enable_id_recovery", False)),
            "compute_dino_similarity": bool(getattr(self.hotrack, "compute_dino_similarity", False)),
            "use_hand_matched_det_boxes_only": bool(getattr(self.hotrack, "use_hand_matched_det_boxes_only", False)),
            "use_all_object_boxes": bool(getattr(self.hotrack, "use_all_object_boxes", False)),
            "enable_component_promote_from_det": bool(getattr(self.hotrack, "enable_component_promote_from_det", False)),
            "replay_enable_structural_split_logic": bool(getattr(self.hotrack, "replay_enable_structural_split_logic", False)),
        }

    def _resize_input(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._orig_shape = (h, w)
        if self.max_side <= 0 or max(h, w) <= self.max_side:
            return frame_bgr
        scale = self.max_side / float(max(h, w))
        return cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    def _upscale_mask(self, mask: np.ndarray) -> np.ndarray:
        oh, ow = self._orig_shape
        if mask.shape[:2] == (oh, ow):
            return mask.astype(np.uint8)
        return cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)

    def process_frame(self, frame_bgr: np.ndarray):
        self.current_frame = self._resize_input(frame_bgr.copy())
        frame_tag = f"{int(getattr(self.hotrack, 'frame_idx', 0)):06d}_online"
        result = self.hotrack.process_frame_with_tracking(
            self.current_frame,
            target_contact_code=self.target_contact_code,
            iou_threshold=self.iou_threshold,
            frame_tag=frame_tag,
        )
        _sanitize_track_only_result(result)
        self.current_result = result
        self.current_frame_idx = int(result.get("frame_idx", -1))
        self.current_masks = {
            int(k): np.asarray(v).astype(np.uint8)
            for k, v in (result.get("tracked_masks", {}) or {}).items()
            if v is not None and np.any(v)
        }
        if self.save_tracking_json:
            _sanitize_track_only_tracking_json(
                Path(self.hotrack.tracking_json_dir) / f"{result.get('frame_tag', frame_tag)}_track.json"
            )
        self.current_vis = self.render(self.current_frame)
        self._save_current_outputs()

        object_masks = [
            self._upscale_mask(self.current_masks[int(oid)])
            for oid in list(result.get("object_ids", []) or [])
            if int(oid) in self.current_masks
        ]
        vis = self.current_vis
        if vis is not None and vis.shape[:2] != self._orig_shape:
            vis = cv2.resize(vis, (self._orig_shape[1], self._orig_shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.interactive_window and self.current_vis is not None:
            cv2.imshow(self.window_name, self.current_vis)
        return vis, object_masks, result

    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        if frame is None:
            frame = self.current_frame
        if frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        vis = frame.copy()
        target_matches = ((self.current_result.get("detections") or {}).get("target_matches") or [])
        for match in target_matches:
            box = match.get("object_bbox_xyxy") if isinstance(match, dict) else None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

        new_ids = {int(x) for x in list(self.current_result.get("new_objects", []) or [])}
        for oid, mask in sorted(self.current_masks.items()):
            if int(oid) < 100:
                continue
            color = (0, 0, 255) if oid in new_ids else self._color_for_id(oid)
            overlay = vis.copy()
            overlay[mask.astype(bool)] = color
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2 if oid == self.selected_id else 1)
            x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
            if bw > 0 and bh > 0:
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2 if oid == self.selected_id else 1)
                cv2.putText(vis, f"O{oid - 100}", (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if self.editing and self.edit_mask is not None:
            edit = vis.copy()
            edit[self.edit_mask.astype(bool)] = (0, 255, 255)
            vis = cv2.addWeighted(edit, 0.35, vis, 0.65, 0)
        text = "track_only | click select | d delete | D frame-remove | e edit | a apply | z cancel | +/- brush | s save | q quit"
        cv2.putText(vis, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(vis, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
        return vis

    @staticmethod
    def _color_for_id(oid: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(int(oid) * 9973)
        return tuple(int(v) for v in rng.integers(60, 230, size=3))

    def handle_key(self, key: int) -> bool:
        if key in (-1, 255):
            return False
        key = key & 0xFF
        if key in (ord("q"), 27):
            self.quit_requested = True
            return True
        if key == ord("s"):
            self._save_current_outputs()
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
        ids = [int(x) for x in sorted(self.current_masks) if int(x) >= 100]
        if not ids:
            self.selected_id = None
            return
        if self.selected_id not in ids:
            self.selected_id = ids[0]
        else:
            self.selected_id = ids[(ids.index(int(self.selected_id)) + int(delta)) % len(ids)]
        self.current_vis = self.render(self.current_frame)

    def delete_active(self) -> None:
        if self.selected_id is None or self.hotrack.inference_state is None:
            return
        oid = int(self.selected_id)
        try:
            ret = self.hotrack.sam2_tracker.remove_object(self.hotrack.inference_state, oid, strict=False, need_output=False)
            if isinstance(ret, tuple) and ret:
                self.hotrack.all_ids = [int(x) for x in list(ret[0])]
        except Exception as exc:
            print(f"[HoTrackTrackOnly] remove_object failed id={oid}: {exc}")
        self.current_masks.pop(oid, None)
        self.selected_id = None
        self.current_vis = self.render(self.current_frame)
        self._save_current_outputs()

    def remove_active_current_frame(self) -> None:
        if self.selected_id is None:
            return
        self.current_masks.pop(int(self.selected_id), None)
        self.current_vis = self.render(self.current_frame)
        self._save_current_outputs()

    def start_edit(self) -> None:
        if self.selected_id is None or int(self.selected_id) not in self.current_masks:
            return
        self.edit_mask = self.current_masks[int(self.selected_id)].copy()
        self.editing = True

    def apply_edit(self) -> None:
        if not self.editing or self.edit_mask is None or self.selected_id is None or self.hotrack.inference_state is None:
            return
        oid = int(self.selected_id)
        self.current_masks[oid] = self.edit_mask.astype(np.uint8)
        try:
            frame_idx = int(self.current_result.get("frame_idx", self.current_frame_idx))
            ret = self.hotrack.sam2_tracker.add_new_mask(
                self.hotrack.inference_state,
                frame_idx=frame_idx,
                obj_id=oid,
                mask=self.edit_mask.astype(bool),
            )
            if isinstance(ret, tuple) and len(ret) >= 3:
                _, obj_ids, logits = ret
                self.hotrack.all_ids = [int(x) for x in list(obj_ids)]
                if logits is not None and oid in self.hotrack.all_ids:
                    idx = self.hotrack.all_ids.index(oid)
                    mask = (logits[idx] > 0).detach().cpu().numpy()
                    mask = np.squeeze(mask).astype(np.uint8)
                    if self.current_frame is not None and mask.shape != self.current_frame.shape[:2]:
                        mask = cv2.resize(mask, (self.current_frame.shape[1], self.current_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    self.current_masks[oid] = mask
        except Exception as exc:
            print(f"[HoTrackTrackOnly] add_new_mask failed id={oid}: {exc}")
        self.cancel_edit()
        self.current_vis = self.render(self.current_frame)
        self._save_current_outputs()

    def cancel_edit(self) -> None:
        self.editing = False
        self.edit_mask = None
        self._drawing = False

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
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
            self.current_vis = self.render(self.current_frame)

    def _paint(self, x: int, y: int) -> None:
        if self.edit_mask is None:
            return
        cv2.circle(self.edit_mask, (int(x), int(y)), int(self.brush_radius), 0 if self._erase else 1, -1)
        self.current_vis = self.render(self.current_frame)

    def _id_at(self, x: int, y: int) -> Optional[int]:
        for oid in sorted(self.current_masks.keys(), reverse=True):
            if int(oid) < 100:
                continue
            mask = self.current_masks[oid]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                return int(oid)
        return None

    def _save_current_outputs(self) -> None:
        if not self.save_masks or self.current_frame is None:
            return
        frame_idx = int(self.current_result.get("frame_idx", self.current_frame_idx if self.current_frame_idx >= 0 else 0))
        tag = str(self.current_result.get("frame_tag") or f"{frame_idx:06d}_online")
        mask_dir = self.masks_dir / tag
        mask_dir.mkdir(parents=True, exist_ok=True)
        for old in mask_dir.glob("id_*.png"):
            old.unlink()
        mask_files = {}
        for oid, mask in sorted(self.current_masks.items()):
            if int(oid) < 100:
                continue
            path = mask_dir / f"id_{int(oid):06d}.png"
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            mask_files[str(int(oid))] = str(path)
        if self.current_vis is not None:
            cv2.imwrite(str(self.overlays_dir / f"{tag}.jpg"), self.current_vis)
        summary = {
            "runtime_mode": "track_only",
            "frame_idx": frame_idx,
            "frame_tag": tag,
            "object_ids": [int(x) for x in list(self.current_result.get("object_ids", []) or [])],
            "new_objects": [int(x) for x in list(self.current_result.get("new_objects", []) or [])],
            "mask_files": mask_files,
            "struct_events": [],
            "id_transitions": [],
            "id_remap": {},
        }
        (self.run_dir / f"{tag}_masks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self.interactive_window:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
