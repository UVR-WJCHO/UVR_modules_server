import os
import contextlib
import hashlib
import json
import re
import sys
import time
import copy
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import torch

HOTRACKER_PATH = "modules/segmentor/sam2_realtime"
SAM2_PACKAGE_ROOT = "modules/segmentor"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAM2_ROOT = os.path.join(PROJECT_ROOT, HOTRACKER_PATH)
SAM2_IMPORT_ROOT = os.path.join(PROJECT_ROOT, SAM2_PACKAGE_ROOT)


def _resolve_under(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    norm = path.replace("\\", "/")
    if (
        norm.startswith(HOTRACKER_PATH + "/")
        or norm.startswith("sam2_realtime/")
    ):
        return os.path.join(PROJECT_ROOT, path)
    return os.path.join(root, path)


def _to_bool_mask(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.size == 0:
        return None
    return m.astype(bool)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


# sam2_realtime and local hograph_plus imports.
for p in [SAM2_IMPORT_ROOT, SAM2_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hotrack.detector import build_detector, detect_image_bgr

# SAM2 모듈 import
from sam2_realtime.sam2.build_sam import build_sam2_realtime_predictor, build_sam2_video_predictor

# Utils 모듈 import
from hotrack.utils import (
    visualize_segmentation,
    calculate_iou,
    calculate_ioa_bidirectional,
    is_duplicate_mask,
    bbox_iou,
    get_image_paths,
    overlay_motion_debug,
)
from hotrack.tuning import MetaTuningAxes, derive_hotrack_params


_SAM2_TIMED_METHODS = {
    "init_state",
    "add_frame",
    "get_mask",
    "add_new_points_or_box",
    "add_new_mask",
    "remove_object",
    "reset_state",
}


class _Sam2RuntimeProxy:
    """Adds per-call timing and optional autocast without changing call sites."""

    def __init__(self, tracker: Any, owner: "Hotrack") -> None:
        self._tracker = tracker
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tracker, name)
        if name not in _SAM2_TIMED_METHODS or not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            owner = self._owner
            owner._sync_if_profile_timing()
            start = time.time()
            with owner._sam2_autocast_context():
                out = attr(*args, **kwargs)
            owner._sync_if_profile_timing()
            owner._record_sam2_call_timing(name, time.time() - start)
            return out

        return wrapped


class _OfflineSam2VideoAdapter:
    """Expose the official SAM2 video predictor through Hotrack's streaming API.

    The main sequence is backed by SAM2VideoPredictor over a prebuilt JPEG folder.
    Hotrack also uses the same tracker object for short replay sessions built from
    in-memory numpy frames; those sessions fall back to the realtime predictor so
    existing split/attach verification logic remains unchanged.
    """

    def __init__(
        self,
        *,
        image_dir: str,
        cache_root: str,
        model_cfg: str,
        checkpoint: str,
        device: str,
        hydra_overrides_extra: Optional[List[str]],
        apply_postprocessing: bool,
        jpeg_quality: int = 95,
        max_frames: int = 0,
    ) -> None:
        self.image_dir = os.path.abspath(image_dir)
        self.cache_root = os.path.abspath(cache_root)
        self.model_cfg = str(model_cfg)
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self.hydra_overrides_extra = list(hydra_overrides_extra or [])
        self.apply_postprocessing = bool(apply_postprocessing)
        self.jpeg_quality = int(jpeg_quality)
        self.max_frames = max(0, int(max_frames))
        self.jpeg_dir = self._prepare_jpeg_cache()
        self.video_predictor = build_sam2_video_predictor(
            self.model_cfg,
            self.checkpoint,
            device=self.device,
            hydra_overrides_extra=self.hydra_overrides_extra,
            apply_postprocessing=self.apply_postprocessing,
            vos_optimized=False,
        )
        self.realtime_fallback = None
        self._main_state_claimed = False

    def _prepare_jpeg_cache(self) -> str:
        frame_paths = [str(p) for p in get_image_paths(self.image_dir)]
        if self.max_frames > 0:
            frame_paths = frame_paths[: self.max_frames]
        if not frame_paths:
            raise FileNotFoundError(f"offline SAM2 backend found no frames under {self.image_dir}")
        digest = hashlib.sha1(os.path.abspath(self.image_dir).encode("utf-8")).hexdigest()[:12]
        cache_dir = os.path.join(
            self.cache_root,
            "_sam2_offline_cache",
            f"{os.path.basename(self.image_dir)}_{digest}_{len(frame_paths)}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        signature = []
        for path in frame_paths:
            st = os.stat(path)
            signature.append({"path": os.path.abspath(path), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)})

        expected_files = [os.path.join(cache_dir, f"{idx:06d}.jpg") for idx in range(len(frame_paths))]
        cache_ok = False
        if os.path.isfile(manifest_path) and all(os.path.isfile(p) for p in expected_files):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                cache_ok = list(manifest.get("frames") or []) == signature
            except Exception:
                cache_ok = False
        if cache_ok:
            return cache_dir

        for idx, path in enumerate(frame_paths):
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"failed to read frame for offline SAM2 cache: {path}")
            out_path = expected_files[idx]
            ok = cv2.imwrite(out_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            if not ok:
                raise RuntimeError(f"failed to write offline SAM2 JPEG cache frame: {out_path}")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"source_dir": self.image_dir, "frames": signature}, f, ensure_ascii=False, indent=2)
        return cache_dir

    def _is_main_state(self, state: Any) -> bool:
        return isinstance(state, dict) and state.get("_hograph_sam2_backend") == "offline_video"

    def _build_realtime_fallback(self) -> Any:
        if self.realtime_fallback is None:
            self.realtime_fallback = build_sam2_realtime_predictor(
                self.model_cfg,
                self.checkpoint,
                device=self.device,
                hydra_overrides_extra=self.hydra_overrides_extra,
                apply_postprocessing=self.apply_postprocessing,
            )
            print("[Hotrack] SAM2 offline backend loaded realtime fallback for replay sessions")
        return self.realtime_fallback

    @staticmethod
    def _with_state(ret: Any, state: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        if isinstance(ret, tuple) and len(ret) == 4:
            return ret
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret[0], ret[1], ret[2], state
        raise RuntimeError(f"unexpected SAM2 return value: {type(ret).__name__}")

    def init_state(
        self,
        img_cv: np.ndarray,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Dict[str, Any]:
        if not self._main_state_claimed:
            state = self.video_predictor.init_state(
                video_path=self.jpeg_dir,
                offload_video_to_cpu=bool(offload_video_to_cpu),
                offload_state_to_cpu=bool(offload_state_to_cpu),
                async_loading_frames=bool(async_loading_frames),
            )
            state["_hograph_sam2_backend"] = "offline_video"
            state["_hograph_current_frame_idx"] = 0
            self._main_state_claimed = True
            return state

        state = self._build_realtime_fallback().init_state(
            img_cv,
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=bool(offload_state_to_cpu),
            async_loading_frames=bool(async_loading_frames),
        )
        if isinstance(state, dict):
            state["_hograph_sam2_backend"] = "realtime_fallback"
        return state

    def add_frame(self, state: Dict[str, Any], img_cv: np.ndarray, offload_video_to_cpu: bool = False) -> Dict[str, Any]:
        if self._is_main_state(state):
            cur = int(state.get("_hograph_current_frame_idx", 0))
            state["_hograph_current_frame_idx"] = min(cur + 1, int(state.get("num_frames", cur + 2)) - 1)
            return state
        return self._build_realtime_fallback().add_frame(state, img_cv, offload_video_to_cpu=bool(offload_video_to_cpu))

    def get_mask(self, state: Dict[str, Any], frame_idx: int) -> Tuple[int, List[int], Any, Dict[str, Any]]:
        if not self._is_main_state(state):
            return self._build_realtime_fallback().get_mask(state, frame_idx)

        frame_idx = int(frame_idx)
        self.video_predictor.propagate_in_video_preflight(state)
        obj_ids = state["obj_ids"]
        batch_size = self.video_predictor._get_obj_num(state)
        if batch_size <= 0:
            return frame_idx, obj_ids, None, state

        pred_masks_per_obj = [None] * batch_size
        for obj_idx in range(batch_size):
            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                pred_masks = current_out["pred_masks"].to(state["device"], non_blocking=True)
                if self.video_predictor.clear_non_cond_mem_around_input:
                    self.video_predictor._clear_obj_non_cond_mem_around_input(state, frame_idx, obj_idx)
            else:
                current_out, pred_masks = self.video_predictor._run_single_frame_inference(
                    inference_state=state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
            pred_masks_per_obj[obj_idx] = pred_masks

        all_pred_masks = torch.cat(pred_masks_per_obj, dim=0) if len(pred_masks_per_obj) > 1 else pred_masks_per_obj[0]
        _, video_res_masks = self.video_predictor._get_orig_video_res_output(state, all_pred_masks)
        return frame_idx, obj_ids, video_res_masks, state

    def add_new_points_or_box(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        if self._is_main_state(state):
            return self._with_state(self.video_predictor.add_new_points_or_box(state, *args, **kwargs), state)
        return self._build_realtime_fallback().add_new_points_or_box(state, *args, **kwargs)

    def add_new_mask(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        if self._is_main_state(state):
            return self._with_state(self.video_predictor.add_new_mask(state, *args, **kwargs), state)
        return self._build_realtime_fallback().add_new_mask(state, *args, **kwargs)

    def remove_object(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self._is_main_state(state):
            return self.video_predictor.remove_object(state, *args, **kwargs)
        return self._build_realtime_fallback().remove_object(state, *args, **kwargs)

    def reset_state(self, state: Dict[str, Any]) -> Any:
        if self._is_main_state(state):
            out = self.video_predictor.reset_state(state)
            self._main_state_claimed = False
            return out
        return self._build_realtime_fallback().reset_state(state)


class Hotrack:
    """Hand-Object Tracker with EXTREME precision focus (avoid false positives)
    
    Design philosophy:
    - Aggressive duplicate removal (if slightly duplicate, remove it)
    - Multiple hard rejection gates (size, shape, hand overlap)
    - Near-hand gating (only track objects near hands)
    - Pending confirmation (NEW objects need 2+ frames to be accepted)
    - Conservative thresholds (prefer reject over accept)
    """
    
    def __init__(
        self,
        image_dir: str,
        output_root: Optional[str] = None,
        backfill_window: int = 15,
        # Hand-object YOLO detector settings.
        ho_thresh_hand: float = 0.5,
        ho_thresh_obj: float = 0.5,
        ho_yolo_model_path: str = "",
        # SAM2 Tracker 설정
        sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_amp: bool = False,
        sam2_amp_dtype: str = "bfloat16",
        sam2_backend: str = "online",
        sam2_compile_image_encoder: bool = False,
        sam2_disable_postprocessing: bool = False,
        sam2_offline_max_frames: int = 0,
        profile_timing: bool = False,
        # 공통 설정
        use_cuda: bool = True,
        # DINO ID recovery (optional mode)
        use_dino_id: bool = False,
        dino_model_name: str = "facebook/dinov2-base",
        dino_checkpoint: Optional[str] = None,
        dino_device: str = "cuda",
        dino_allow_download: bool = True,
        # Similarity reporting (no ID switch prevention)
        enable_id_recovery: bool = False,
        compute_dino_similarity: bool = True,
        # Size gating
        max_obj_area_ratio: float = 0.4,
        max_obj_vs_hand_ratio: float = 4.0,
        # Meta-parameter tuning (5 axes)
        meta_tuning: Optional[MetaTuningAxes] = None,
        threshold_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.image_dir = _resolve_under(PROJECT_ROOT, image_dir)
        self.output_root = self._resolve_output_root(output_root)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        # Hand-object YOLO detector settings.
        self.ho_thresh_hand = ho_thresh_hand
        self.ho_thresh_obj = ho_thresh_obj
        self.ho_yolo_model_path = str(ho_yolo_model_path or "").strip()
        # Distance transform point sampling for SAM2 refinement
        self.dt_point_count: int = 3
        self.dt_point_min_dist: int = 3
        # Optional ablation: skip the second point-prompt refinement after box seeding.
        # Default stays enabled to preserve the paper reproduction path.
        self.enable_prompt_refinement: bool = True

        # SAM2 Tracker Settings
        self.sam2_checkpoint = _resolve_under(SAM2_ROOT, sam2_checkpoint)
        self.sam2_model_cfg = sam2_model_cfg
        self.sam2_amp = bool(sam2_amp)
        self.sam2_amp_dtype = str(sam2_amp_dtype or "bfloat16").strip().lower()
        self.sam2_backend = str(sam2_backend or "online").strip().lower()
        if self.sam2_backend not in {"online", "offline"}:
            raise ValueError("sam2_backend must be 'online' or 'offline'")
        self.sam2_compile_image_encoder = bool(sam2_compile_image_encoder)
        self.sam2_disable_postprocessing = bool(sam2_disable_postprocessing)
        self.sam2_offline_max_frames = max(0, int(sam2_offline_max_frames))
        self.profile_timing = bool(profile_timing)
        self._sam2_timing_frame: Dict[str, float] = {}

        self.ho_model = None
        self.ho_class_names = None
        self.sam2_tracker = None
        
        # SAM2 트래킹 상태
        self.inference_state: Optional[Dict] = None
        self.all_ids: List[int] = []
        self.frame_idx: int = 0
        
        # Stable ID allocation counters (monotonic, never reuse)
        self.next_hand_id: int = 0  # Hand IDs: 0-99
        self.next_obj_id: int = 100  # Object IDs: 100+
        
        # Events (minimal - graph code handles semantic reasoning)
        self.events: List[Dict[str, Any]] = []
        # Structural operation mode (SAM2 state-coupled split/merge).
        self.enable_structural_ops: bool = True
        self.id_recovery_block_after_struct_frames: int = 15
        self._struct_id_recovery_block_until: Dict[int, int] = {}
        self._struct_recovery_block_until_frame: int = -1
        self._struct_event_last_frame: Dict[int, int] = {}
        self._struct_events_frame: List[Dict[str, Any]] = []
        self._id_transitions_frame: List[Dict[str, Any]] = []
        # Structurally retired IDs must never be recovered again.
        self._retired_struct_ids: Set[int] = set()
        
        # =================================================================
        # EXTREME PRECISION THRESHOLDS (intentionally conservative)
        # =================================================================
        
        # DUPLICATE removal - more aggressive (lower thresholds = easier to reject)
        # Rationale: if it looks even slightly duplicate, remove it
        self.dup_ioa_lo: float = 0.7  # IoA (both directions) >= this => duplicate
        # Reject new masks when overlap is moderately high
        self.new_mask_reject_ioa: float = 0.4
        # Reject when a new mask is mostly contained within an existing mask
        self.new_mask_inclusion_reject_ioa: float = 0.85
        # Force-new override: if detector box is isolated from existing objects,
        # allow newborn candidate even when SAM2 mask leaks and triggers overlap gate.
        self.force_new_when_det_box_isolated: bool = True
        self.force_new_det_box_isolation_iou_th: float = 0.05
        self.force_new_det_box_isolation_mask_in_th: float = 0.05
        self.force_new_det_box_min_candidate_in_box_ratio: float = 0.01
        # HAND overlap rejection (strict to avoid hand mistaken as object)
        self.hand_overlap_reject: float = 0.5  # IoA(hand->obj) or IoA(obj->hand) >= this => reject
        self.hand_dilation_kernel: int = 20  # pixels for dilated hand mask
        
        # Detection mask overlaps a component of a multi-component tracked mask.
        # Disabled by default because it tends to create false-positive newborn IDs
        # from inclusion-rejected detector masks.
        self.enable_component_promote_from_det: bool = False
        self.detect_component_add_ioa_threshold: float = 0.6
        # Post-tracking duplicate removal pass (overlap-based ID pruning).
        self.enable_post_track_duplicate_removal: bool = True
        # Seed SAM2 with all detected object boxes, not only hand-matched ones.
        self.use_all_object_boxes: bool = False
        # Strict mode: downstream logic uses only hand-matched detector boxes.
        self.use_hand_matched_det_boxes_only: bool = True
        # Track hands as SAM2 mask IDs. Paper mode keeps this enabled for
        # mask-level hand/object reasoning; track-only can use detector hand
        # boxes as lightweight proxy masks instead.
        self.track_hand_masks: bool = True
        # Skip expensive SAM2 prompt calls when a detector target box already
        # matches an active tracked object. This is intended for fast track-only
        # operation; paper mode keeps the legacy per-frame prompt behavior.
        self.skip_existing_object_box_prompts: bool = False
        self.existing_object_box_iou_skip_th: float = 0.25
        self.existing_object_box_cover_skip_th: float = 0.55
        
        # Quality scoring (select best if multiple candidates)
        # 0 or negative => unlimited new objects accepted per frame.
        self.max_new_per_frame: int = 0
        # Candidate scorer: replace stacked hard gates with a single soft score.
        self.enable_candidate_scorer: bool = True
        self.candidate_score_threshold: float = 0.45
        self.candidate_backfill_top_p: int = 2
        self.candidate_backfill_max_steps: int = 4
        self.candidate_model_path: str = str(os.environ.get("HOTRACK_CANDIDATE_MODEL", "") or "").strip()
        self.candidate_log_enabled: bool = True
        self.candidate_log_path: str = ""
        self._candidate_model: Optional[Dict[str, Any]] = None

        # Temporal ID reassignment (prevent shrink-and-replace ID switch)
        self.enable_temporal_id_reassign: bool = True
        self.temporal_reassign_shrink_ratio: float = 0.45
        self.temporal_reassign_prev_to_new_ioa: float = 0.60
        self.temporal_reassign_new_to_prev_ioa: float = 0.60
        self.temporal_reassign_new_vs_cur_ratio: float = 1.6
        self.temporal_reassign_min_prev_area: int = 80
        
        # Unified confusion memory (includes lost objects as a subset).
        self.confusion_objects_history: Dict[int, Dict[str, Any]] = {}  # obj_id -> {"frame":..., "embed":..., "mask":..., "reason":...}
        self.min_mask_area: int = 50  # Minimum mask area (pixels) to consider valid
        self.max_obj_area_ratio = float(max_obj_area_ratio)
        self.max_obj_vs_hand_ratio = float(max_obj_vs_hand_ratio)
        self.dino_recovery_sim_threshold: float = 0.7
        # Relaxed recovery for active-overlap confusion snapshots.
        self.confusion_overlap_relaxed_ioa_th: float = 0.2
        self.confusion_overlap_min_ioa_for_dino: float = 0.05
        self.confusion_overlap_max_age_frames: int = 60
        self.enable_id_recovery = bool(enable_id_recovery)
        self.compute_dino_similarity = bool(compute_dino_similarity)
        # Per-frame selective DINO EMA freeze for duplicate-overlap IDs only.
        self._freeze_dino_ema_obj_ids: Set[int] = set()
        
        # Mask history for temporal consistency
        self.mask_history: Dict[int, deque] = {}  # obj_id -> deque of (frame_idx, mask)
        self.mask_history_length: int = 5  # Keep last N frames for temporal ID consistency
        self._prev_tracked_masks: Dict[int, np.ndarray] = {}
        # Multi-component handling (objects only)
        self.detect_add_ioa_threshold: float = 0.35
        self._current_target_boxes: List[List[int]] = []
        self._current_target_points: List[List[Tuple[int, int]]] = []
        self._current_split_target_masks: List[np.ndarray] = []
        # Pending split confirmation (T consecutive frames)
        # Split only after consistent detector-backed evidence.
        self.split_confirm_frames: int = 1
        self.split_min_area_ratio: float = 0.12
        # If both split/keep components are mostly inside the same detector bbox,
        # treat as occlusion and do not split.
        self.split_same_box_in_ratio_th: float = 0.3
        # Component->detector-mask duplicate gate for multicomponent split.
        # Require near-duplicate overlap (bi-directional IoA).
        self.split_component_duplicate_ioa_th: float = 0.85
        # When False, component association must come from mask overlap only
        # (det->point->mask), not detector-box fallback.
        self.split_component_allow_box_fallback: bool = False
        # Strict split evidence policy:
        # - fallback evidence off: no split when only one component is detector-backed
        # - require keep/split components to map to two distinct detector boxes
        self.split_allow_fallback_evidence: bool = False
        self.split_require_two_det_boxes: bool = True
        # Reject split when candidate component still overlaps another tracked object.
        self.split_other_overlap_reject_ioa: float = self.new_mask_reject_ioa
        # Seed split candidates from any newly created object (det/new or multicomponent/new).
        self.split_register_all_new_objects: bool = True
        self.split_new_parent_min_child_in_prev: float = 0.08
        self.split_new_parent_min_overlap_now: float = 0.05
        self.split_new_parent_score_th: float = 0.10
        # When True, `new_object_from_multicomponent` is emitted immediately
        # once detector-backed split evidence appears in the current frame.
        self.split_multicomponent_instant_new: bool = True
        self._pending_splits: Dict[int, Dict[str, Any]] = {}
        # Split candidates are tracked by newly created child IDs.
        self._pending_split_from_new: Dict[int, Dict[str, Any]] = {}
        self._emitted_struct_split_pairs: Set[Tuple[int, int]] = set()
        # Split replay session cache:
        # key=(parent_id, child_id, split_start_frame) -> verify result.
        # Reuse for the same candidate session to avoid re-running heavy replay each frame.
        self._split_verify_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.split_candidate_max_age_frames: int = 30
        # On repeated verification failure/expiry, drop split-candidate only (no rollback by default).
        self.split_candidate_fail_drop_frames: int = 3
        self.split_candidate_revert_on_expire: bool = False
        self.split_candidate_revert_on_fail: bool = False
        # New-object split gate: child must be contained by
        # the same parent mask for consecutive frames in replay window.
        self.split_new_det_pair_in_th: float = 0.3
        self.split_new_det_pair_persist_frames: int = 3
        # Split replay window for new-object validation.
        # Confirmation still requires `split_new_det_pair_persist_frames`,
        # but replay is evaluated over this longer forward window.
        self.split_new_replay_window_frames: int = 30
        # When a new object is born but strict parent pick fails, run split
        # verification with a relaxed parent pick so debug evidence is still produced.
        self.split_seed_relaxed_when_parent_missing: bool = True
        # Reverse attach replay mirrors forward new-object decision flow.
        self.replay_enable_structural_split_logic: bool = True
        # Ambiguous hand/object regions (detector label flips or high overlap)
        self.ambiguous_confirm_frames: int = 3
        self.ambiguous_bbox_iou_threshold: float = 0.6
        self.ambiguous_mask_ioa_threshold: float = 0.85
        self.ambiguous_max_age: int = 6
        self._ambiguous_regions: Dict[int, Dict[str, Any]] = {}
        self._ambiguous_next_id: int = 0
        self._prev_hand_boxes: List[List[int]] = []
        self._prev_object_boxes: List[List[int]] = []

        # DINO appearance matcher (optional)
        self.use_dino_id = bool(use_dino_id)
        self.dino_model_name = str(dino_model_name)
        self.dino_checkpoint = dino_checkpoint
        self.dino_device = str(dino_device)
        self.dino_allow_download = bool(dino_allow_download)
        self._dino_embedder = None
        self._dino_warned = False
        self._dino_error_msg: Optional[str] = None
        self._dino_model_id_resolved: Optional[str] = None
        self._dino_embedding_history: Dict[int, np.ndarray] = {}
        self._id_last_seen_frame: Dict[int, int] = {}
        self._id_birth_frame: Dict[int, int] = {}
        # DINO history update policy: EMA over per-object embeddings.
        # new = alpha * current + (1-alpha) * previous
        self.dino_embed_ema_alpha: float = 0.3

        # Backfill tracking (reverse-window, separate state)
        self.backfill_window: int = max(int(backfill_window), 0)
        self._frame_history: deque = deque(maxlen=self.backfill_window)
        self._frame_idx_history: deque = deque(maxlen=self.backfill_window)
        self._tracked_masks_history: deque = deque(maxlen=self.backfill_window)
        self._target_boxes_history: deque = deque(maxlen=self.backfill_window)
        self._target_masks_history: deque = deque(maxlen=self.backfill_window)
        self._tracked_boxes_history: deque = deque(maxlen=self.backfill_window)
        self._geom_history: deque = deque(maxlen=self.backfill_window)
        self._bg_history: deque = deque(maxlen=self.backfill_window)
        self.backfill_debug: bool = False
        self.attach_proxy_backfill_enabled: bool = True
        self.attach_proxy_backfill_max_boxes: int = 3
        self.attach_mask_in_th: float = 0.45
        self.attach_proxy_signal_in_ratio_th: float = 0.8
        self.attach_proxy_signal_pair_union_in_th: float = 0.7
        self.attach_proxy_signal_persist_frames: int = 3
        self.attach_reverse_y_window: int = 15
        self.attach_reverse_xy_match_th: float = 0.25
        # After Y is first created in reverse replay, keep this many
        # additional reverse frames and allow XY<->AB pair match in that window.
        self.attach_reverse_xy_match_post_y_frames: int = 2
        self.attach_forward_window: int = 15
        self.attach_forward_count_on_th: int = 5
        # Cooldown after attach-tail start before allowing structural merge.
        # Set to 0 to bypass this gate.
        self.merge_after_attach_cooldown_frames: int = 0
        # Size gate for merge candidates sourced from attach-session seed boxes.
        # Prevents full-frame (or near full-frame) detector boxes from triggering merge.
        self.merge_seed_box_size_gate: bool = True
        self.merge_seed_box_max_area_ratio: float = self.max_obj_area_ratio
        # Reverse replay starts a bit after candidate-start to avoid early unstable contact.
        self.attach_replay_start_offset_frames: int = 2
        self.attach_seed_box_topk: int = 1
        self.attach_replay_all_candidates: bool = True
        # Replay init uses the seed box as-is.
        self.attach_session_ttl_frames: int = 240
        replay_debug = _env_flag("HOGRAPH_SAVE_REPLAY_DEBUG", _env_flag("HOGRAPH_SAVE_HEAVY_DEBUG", False))
        self.attach_replay_debug: bool = replay_debug
        self.attach_replay_debug_only_passed: bool = True
        self.split_replay_debug: bool = replay_debug
        self._attach_pair_prev_qualified: Dict[Tuple[int, int], bool] = {}
        self._attach_pair_last_qualified: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._attach_pair_locked_start: Dict[Tuple[int, int], int] = {}
        self._attach_replay_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._attach_sessions: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._attach_proxy_id_counter: int = 0
        self._split_replay_debug_counter: int = 0
        self.video_name: str = os.path.basename(os.path.abspath(image_dir.rstrip("/")))
        self.backfill_debug_dir: str = os.path.join(self.output_root, "backfill", self.video_name)
        self.tracking_dir: str = os.path.join(self.output_root, self.video_name, "tracking")
        self.tracking_json_dir: str = os.path.join(self.tracking_dir, "json")
        self.tracking_png_dir: str = os.path.join(self.tracking_dir, "png")
        self.attach_replay_debug_dir: str = os.path.join(self.tracking_dir, "attach_replay_debug")
        self.split_replay_debug_dir: str = os.path.join(self.tracking_dir, "split_replay_debug")
        self.candidate_log_path = os.path.join(self.tracking_json_dir, "candidate_features.jsonl")
        self.save_tracking_json: bool = _env_flag("HOGRAPH_SAVE_TRACKING_JSON", False)

        # Background motion estimation cache
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_fg_mask: Optional[np.ndarray] = None

        # Post-tracking duplicate removal (ID switch prevention)
        self.post_track_dup_ioa_threshold: float = 0.8  # IoA threshold for post-tracking duplicate check
        self.post_dup_history: Dict[Tuple[int, int], int] = {}
        self.post_dup_locked_decisions: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Require consecutive overlap frames before removing as duplicate.
        self.post_dup_confirm_frames: int = 3
        # Optional fast mode: compare/remove immediately at overlap start (streak=1).
        # Default keeps temporal confirmation and only locks keep/remove at start.
        self.post_dup_immediate_compare: bool = False
        # If a duplicate pair includes a newly born ID, remove duplicate immediately.
        self.post_dup_immediate_if_newborn: bool = True
        self.post_dup_newborn_max_age_frames: int = 6
        # Cooldown to prevent immediate re-creation of just-removed duplicate IDs.
        self.dup_readd_cooldown_frames: int = 3
        self.dup_readd_cooldown_ioa: float = 0.9
        self._recent_post_duplicates: Dict[int, Dict[str, Any]] = {}

        self.hand_inclusion_ioa_threshold: float = 0.8

        # Static hand cleanup (likely misclassified object)
        self.hand_static_frames: int = 5
        self.hand_static_threshold: float = 1.0
        self.hand_static_counts: Dict[int, int] = {}
        self.prev_hand_centroids: Dict[int, Tuple[float, float]] = {}

        # Optional: collapse many low-level thresholds from a compact 5-axis config.
        self._meta_tuning = meta_tuning
        if meta_tuning is not None:
            self._apply_threshold_overrides(derive_hotrack_params(meta_tuning))
        if threshold_overrides:
            self._apply_threshold_overrides(threshold_overrides)
        self._candidate_model = self._load_candidate_model()

        
        # GPU Optimization
        if self.use_cuda and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Model Initialization
        self._init_yolo_detector()
        self._init_sam2_tracker()

    def _resolve_output_root(self, output_root: Optional[str]) -> str:
        if output_root:
            return os.path.abspath(output_root)
        return os.path.join(PROJECT_ROOT, "static", "outputs")

    def _apply_threshold_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        if not overrides:
            return
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                setattr(self, key, bool(value))
                continue
            if isinstance(current, int):
                try:
                    setattr(self, key, max(0, int(round(float(value)))))
                except Exception:
                    continue
                continue
            if isinstance(current, float):
                try:
                    setattr(self, key, float(value))
                except Exception:
                    continue
                continue
            setattr(self, key, value)

    def set_output_dirs(
        self,
        output_root: str,
        video_name: str,
        *,
        image_dir: Optional[str] = None,
    ) -> None:
        """Reconfigure output directories for a new sequence."""
        resolved_root = self._resolve_output_root(output_root)
        self.video_name = str(video_name)
        # If caller passed a per-video output folder, normalize to root.
        if os.path.basename(resolved_root) == self.video_name:
            base_dir = resolved_root
            self.output_root = os.path.dirname(resolved_root)
        else:
            self.output_root = resolved_root
            base_dir = os.path.join(self.output_root, self.video_name)
        if image_dir is not None:
            self.image_dir = _resolve_under(PROJECT_ROOT, image_dir)
        self.backfill_debug_dir = os.path.join(self.output_root, "backfill", self.video_name)
        self.tracking_dir = os.path.join(base_dir, "tracking")
        self.tracking_json_dir = os.path.join(self.tracking_dir, "json")
        self.tracking_png_dir = os.path.join(self.tracking_dir, "png")
        self.attach_replay_debug_dir = os.path.join(self.tracking_dir, "attach_replay_debug")
        self.split_replay_debug_dir = os.path.join(self.tracking_dir, "split_replay_debug")
        self.candidate_log_path = os.path.join(self.tracking_json_dir, "candidate_features.jsonl")
        self.candidate_log_enabled = bool(self.save_tracking_json)
        if self.backfill_debug:
            os.makedirs(self.backfill_debug_dir, exist_ok=True)
        os.makedirs(self.tracking_dir, exist_ok=True)
        if self.save_tracking_json:
            os.makedirs(self.tracking_json_dir, exist_ok=True)
        if _env_flag("HOGRAPH_SAVE_TRACKING_PNG", False):
            os.makedirs(self.tracking_png_dir, exist_ok=True)
        if self.attach_replay_debug:
            os.makedirs(self.attach_replay_debug_dir, exist_ok=True)
        if self.split_replay_debug:
            os.makedirs(self.split_replay_debug_dir, exist_ok=True)

    def _init_yolo_detector(self):
        self.ho_model, self.ho_class_names = build_detector(
            use_cuda=self.use_cuda,
            yolo_model_path=self.ho_yolo_model_path,
        )
        if isinstance(self.ho_model, dict):
            runtime_backend = str(self.ho_model.get("backend", "unknown"))
            ckpt_path = str(self.ho_model.get("_yolo_checkpoint_path", "")).strip()
        else:
            runtime_backend = str(getattr(self.ho_model, "backend", "unknown"))
            ckpt_path = str(getattr(self.ho_model, "_yolo_checkpoint_path", "")).strip()
        print(
            f"[Hotrack] Hand-Object Detector loaded on {self.device} "
            f"(runtime={runtime_backend})"
        )
        if ckpt_path:
            print(f"[Hotrack] HO checkpoint={ckpt_path}")

    def _init_sam2_tracker(self):
        hydra_overrides_extra: List[str] = []
        if bool(self.sam2_compile_image_encoder):
            hydra_overrides_extra.append("++model.compile_image_encoder=True")
        if self.sam2_backend == "offline":
            tracker = _OfflineSam2VideoAdapter(
                image_dir=self.image_dir,
                cache_root=self.output_root,
                model_cfg=self.sam2_model_cfg,
                checkpoint=self.sam2_checkpoint,
                device=self.device,
                hydra_overrides_extra=hydra_overrides_extra,
                apply_postprocessing=not bool(self.sam2_disable_postprocessing),
                max_frames=int(self.sam2_offline_max_frames),
            )
        else:
            tracker = build_sam2_realtime_predictor(
                self.sam2_model_cfg,
                self.sam2_checkpoint,
                device=self.device,
                hydra_overrides_extra=hydra_overrides_extra,
                apply_postprocessing=not bool(self.sam2_disable_postprocessing),
            )
        self.sam2_tracker = _Sam2RuntimeProxy(tracker, self)
        print(f"[Hotrack] SAM2 Tracker loaded on {self.device} (backend={self.sam2_backend})")
        if self.sam2_amp and self.use_cuda:
            print(f"[Hotrack] SAM2 autocast enabled dtype={self.sam2_amp_dtype}")
        if self.sam2_compile_image_encoder:
            print("[Hotrack] SAM2 compile_image_encoder enabled")
        if self.sam2_disable_postprocessing:
            print("[Hotrack] SAM2 postprocessing disabled")
        if self.backfill_debug:
            os.makedirs(self.backfill_debug_dir, exist_ok=True)
        os.makedirs(self.tracking_dir, exist_ok=True)
        if self.save_tracking_json:
            os.makedirs(self.tracking_json_dir, exist_ok=True)
        if _env_flag("HOGRAPH_SAVE_TRACKING_PNG", False):
            os.makedirs(self.tracking_png_dir, exist_ok=True)
        if self.attach_replay_debug:
            os.makedirs(self.attach_replay_debug_dir, exist_ok=True)
        if self.split_replay_debug:
            os.makedirs(self.split_replay_debug_dir, exist_ok=True)

    def reset_tracker_state(self):
        """SAM2 트래커 상태 초기화"""
        if self.inference_state is not None:
            self.sam2_tracker.reset_state(self.inference_state)
        self.inference_state = None
        self.all_ids = []
        self.frame_idx = 0
        self.next_hand_id = 0
        self.next_obj_id = 100
        self.confusion_objects_history = {}
        self.mask_history = {}
        self.events = []
        self._struct_events_frame = []
        self._id_transitions_frame = []
        self._struct_id_recovery_block_until = {}
        self._struct_recovery_block_until_frame = -1
        self._struct_event_last_frame = {}
        self._retired_struct_ids = set()
        self._ambiguous_regions = {}
        self._ambiguous_next_id = 0
        self._prev_hand_boxes = []
        self._prev_object_boxes = []
        self.hand_static_counts = {}
        self.prev_hand_centroids = {}
        self._current_target_boxes = []
        self._pending_splits = {}
        self._pending_split_from_new = {}
        self._emitted_struct_split_pairs = set()
        self._split_verify_cache = {}
        self._frame_history.clear()
        self._frame_idx_history.clear()
        self._tracked_masks_history.clear()
        self._target_boxes_history.clear()
        self._target_masks_history.clear()
        self._tracked_boxes_history.clear()
        self._geom_history.clear()
        self._bg_history.clear()
        self._prev_gray = None
        self._prev_fg_mask = None
        self._prev_tracked_masks = {}
        self._frame_history = deque(maxlen=self.backfill_window)
        self._frame_idx_history = deque(maxlen=self.backfill_window)
        self._tracked_masks_history = deque(maxlen=self.backfill_window)
        self._target_boxes_history = deque(maxlen=self.backfill_window)
        self._target_masks_history = deque(maxlen=self.backfill_window)
        self._tracked_boxes_history = deque(maxlen=self.backfill_window)
        self._geom_history = deque(maxlen=self.backfill_window)
        self._bg_history = deque(maxlen=self.backfill_window)
        # Keep backfill debug directory ready
        if self.backfill_debug:
            os.makedirs(self.backfill_debug_dir, exist_ok=True)
        self.post_dup_history = {}
        self.post_dup_locked_decisions = {}
        self._dino_embedding_history = {}
        self._id_last_seen_frame = {}
        self._id_birth_frame = {}
        self._attach_pair_prev_qualified = {}
        self._attach_pair_last_qualified = {}
        self._attach_pair_locked_start = {}
        self._attach_replay_cache = {}
        self._attach_sessions = {}
        self._attach_proxy_id_counter = 0
        self._split_replay_debug_counter = 0
        self._freeze_dino_ema_obj_ids = set()
        if isinstance(self.ho_model, dict) and "frame_idx" in self.ho_model:
            self.ho_model["frame_idx"] = 0

    def _sam2_autocast_context(self):
        if not (bool(self.sam2_amp) and bool(self.use_cuda)):
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if str(self.sam2_amp_dtype) == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _sync_if_profile_timing(self) -> None:
        if bool(self.profile_timing) and bool(self.use_cuda) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record_sam2_call_timing(self, name: str, elapsed: float) -> None:
        if not isinstance(self._sam2_timing_frame, dict):
            self._sam2_timing_frame = {}
        key = f"sam2_call_{str(name)}"
        self._sam2_timing_frame[key] = float(self._sam2_timing_frame.get(key, 0.0) + max(0.0, float(elapsed)))
    
    def _get_mask_bbox(self, mask: np.ndarray) -> List[int]:
        """Get bbox from mask using nonzero (more reliable than contours[0])"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return [x1, y1, x2, y2]

    @staticmethod
    def _expand_box_xyxy(
        box: List[int],
        image_w: int,
        image_h: int,
        *,
        expand_ratio: float,
        min_pad_px: int,
    ) -> List[int]:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return [0, 0, 0, 0]
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            return [0, 0, 0, 0]
        if image_w <= 1 or image_h <= 1:
            return [0, 0, 0, 0]

        x1 = max(0, min(int(image_w - 1), int(x1)))
        x2 = max(0, min(int(image_w - 1), int(x2)))
        y1 = max(0, min(int(image_h - 1), int(y1)))
        y2 = max(0, min(int(image_h - 1), int(y2)))
        if x2 <= x1 or y2 <= y1:
            return [int(x1), int(y1), int(x2), int(y2)]

        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        ratio = max(0.0, float(expand_ratio))
        pad_min = max(0, int(min_pad_px))
        pad_x = max(pad_min, int(round(w * ratio)))
        pad_y = max(pad_min, int(round(h * ratio)))

        ex1 = max(0, int(x1 - pad_x))
        ey1 = max(0, int(y1 - pad_y))
        ex2 = min(int(image_w - 1), int(x2 + pad_x))
        ey2 = min(int(image_h - 1), int(y2 + pad_y))
        if ex2 <= ex1 or ey2 <= ey1:
            return [int(x1), int(y1), int(x2), int(y2)]
        return [int(ex1), int(ey1), int(ex2), int(ey2)]

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> List[int]:
        """Encode binary mask as simple RLE counts (row-major, starts with zeros)."""
        if mask is None:
            return []
        flat = mask.astype(np.uint8).reshape(-1)
        counts: List[int] = []
        prev = 0
        run = 0
        for v in flat:
            if v == prev:
                run += 1
            else:
                counts.append(run)
                run = 1
                prev = int(v)
        counts.append(run)
        return counts

    def _affine_to_tr(self, A: np.ndarray) -> Tuple[np.ndarray, float]:
        t = np.array([float(A[0, 2]), float(A[1, 2])], dtype=np.float32)
        theta = float(np.arctan2(A[1, 0], A[0, 0]))
        return t, theta

    def _estimate_component_motion(
        self,
        prev_gray: Optional[np.ndarray],
        gray: Optional[np.ndarray],
        comp_mask: np.ndarray,
        A_bg: Optional[np.ndarray],
        bg_ok: bool,
    ) -> Optional[Tuple[np.ndarray, float]]:
        if prev_gray is None or gray is None:
            return None
        if prev_gray.shape != gray.shape or comp_mask.shape != prev_gray.shape:
            return None
        mask_u8 = comp_mask.astype(np.uint8) * 255
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=mask_u8,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=5,
        )
        if pts_prev is None or len(pts_prev) < 6:
            return None
        pts_curr, st, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if pts_curr is None or st is None:
            return None
        good_prev = pts_prev[st.flatten() == 1]
        good_curr = pts_curr[st.flatten() == 1]
        if len(good_prev) < 6:
            return None
        A_obj, _inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if A_obj is None:
            flow = (good_curr - good_prev).reshape(-1, 2)
            t = np.median(flow, axis=0).astype(np.float32)
            theta = 0.0
        else:
            t, theta = self._affine_to_tr(A_obj)
        if bg_ok and A_bg is not None:
            t_bg, theta_bg = self._affine_to_tr(A_bg)
            t = t - t_bg
            theta = float(theta - theta_bg)
        return t, theta

    def _masked_crop_bgr(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        pad: int = 6,
    ) -> Optional[np.ndarray]:
        out = self._masked_crop_with_mask(image_bgr, mask, pad=pad)
        if out is None:
            return None
        crop, crop_mask = out
        if crop.size == 0:
            return None
        bg = np.full_like(crop, 127, dtype=np.uint8)
        crop[~crop_mask] = bg[~crop_mask]
        return crop

    def _masked_crop_with_mask(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        pad: int = 6,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if image_bgr is None or mask is None:
            return None
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        h, w = mask.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        crop = image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
        crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        return crop, crop_mask

    def _resolve_dino_model_id(self, model_id: str) -> str:
        if not model_id:
            return model_id
        mid = str(model_id)
        if mid in ("dinov2_vits14", "dinov2-small", "dinov2_small"):
            return "facebook/dinov2-small"
        if mid in ("dinov2_vitb14", "dinov2-base", "dinov2_base"):
            return "facebook/dinov2-base"
        if mid in ("dinov2_vitl14", "dinov2-large", "dinov2_large"):
            return "facebook/dinov2-large"
        if mid in ("dinov2_vitg14", "dinov2-giant", "dinov2_giant"):
            return "facebook/dinov2-giant"
        if "/" not in mid and (mid.startswith("dinov3-") or mid.startswith("dinov3_")):
            return "facebook/" + mid.replace("_", "-")
        return mid

    def _get_dino_embedder(self):
        if self._dino_embedder is not None:
            return self._dino_embedder
        if not self.use_dino_id:
            self._dino_error_msg = "use_dino_id=False"
            return None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            from PIL import Image

            dev = self.dino_device
            if str(dev).startswith("cuda") and not torch.cuda.is_available():
                dev = "cpu"
            model_id = self._resolve_dino_model_id(self.dino_checkpoint or self.dino_model_name)
            self._dino_model_id_resolved = model_id
            local_only = not self.dino_allow_download
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            model = AutoModel.from_pretrained(
                model_id,
                local_files_only=local_only,
                trust_remote_code=True,
                token=token,
            ).to(dev).eval()
            processor_kwargs = {
                "local_files_only": local_only,
                "trust_remote_code": True,
                "token": token,
            }
            try:
                # Prefer slow processor when available for backward consistency.
                processor = AutoImageProcessor.from_pretrained(
                    model_id,
                    use_fast=False,
                    **processor_kwargs,
                )
            except Exception as proc_exc:
                msg = str(proc_exc)
                if ("does not have a slow version" in msg) or ("use_fast=True" in msg):
                    # DINOv3 fast-only processors require use_fast=True.
                    processor = AutoImageProcessor.from_pretrained(
                        model_id,
                        use_fast=True,
                        **processor_kwargs,
                    )
                else:
                    raise

            self._dino_embedder = {
                "torch": torch,
                "model": model,
                "processor": processor,
                "Image": Image,
                "device": dev,
            }
            self._dino_error_msg = None
            return self._dino_embedder
        except Exception as exc:
            if not self._dino_warned:
                print("[Hotrack] DINO unavailable or not cached; DINO ID recovery unavailable")
                self._dino_warned = True
            self._dino_error_msg = f"{type(exc).__name__}: {exc}"
            return None

    def _dino_status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.use_dino_id),
            "embedder_ready": self._dino_embedder is not None,
            "model_id": self._dino_model_id_resolved or (self.dino_checkpoint or self.dino_model_name),
            "allow_download": bool(self.dino_allow_download),
            "device": str(self.dino_device),
            "history_count": int(len(self._dino_embedding_history)),
            "error": self._dino_error_msg,
        }

    def _dino_encode(self, image_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        embedder = self._get_dino_embedder()
        if embedder is None:
            return None
        out = self._masked_crop_with_mask(image_bgr, mask, pad=8)
        if out is None:
            return None
        crop, crop_mask = out
        if crop is None or crop_mask is None:
            return None
        # Zero out background to reduce leakage.
        crop_bg = crop.copy()
        crop_bg[~crop_mask] = 0
        rgb = cv2.cvtColor(crop_bg, cv2.COLOR_BGR2RGB)
        pil = embedder["Image"].fromarray(rgb)
        inputs = embedder["processor"](images=pil, return_tensors="pt")
        inputs = {k: v.to(embedder["device"]) for k, v in inputs.items()}
        with embedder["torch"].inference_mode():
            out = embedder["model"](**inputs)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                tokens = out.last_hidden_state
                if tokens.dim() == 3 and tokens.shape[1] > 1:
                    # Build patch mask weights aligned to token grid.
                    try:
                        patch_size = int(getattr(embedder["model"].config, "patch_size", 14))
                    except Exception:
                        patch_size = 14
                    _, _, h, w = inputs["pixel_values"].shape
                    gh = max(1, int(h // patch_size))
                    gw = max(1, int(w // patch_size))
                    mask_resized = cv2.resize(
                        crop_mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_t = embedder["torch"].from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(tokens.device)
                    mask_grid = embedder["torch"].nn.functional.interpolate(
                        mask_t, size=(gh, gw), mode="area"
                    ).reshape(-1)
                    tok = tokens[:, 1:, :]
                    if mask_grid.numel() == tok.shape[1] and float(mask_grid.sum()) > 1e-6:
                        wts = mask_grid / (mask_grid.sum() + 1e-8)
                        vec = (tok * wts.view(1, -1, 1)).sum(dim=1)
                    else:
                        vec = tokens[:, 0]
                else:
                    vec = tokens[:, 0]
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                return None
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        v = vec.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        return v

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an <= 1e-8 or bn <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def _normalize_embedding(self, emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if emb is None:
            return None
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n <= 1e-8:
            return None
        return (v / n).astype(np.float32)

    def _update_dino_embedding_history(self, obj_id: int, emb: Optional[np.ndarray]) -> bool:
        """Update per-object DINO history using EMA."""
        if emb is None:
            return False
        cur = self._normalize_embedding(emb)
        if cur is None:
            return False

        oid = int(obj_id)
        prev = self._dino_embedding_history.get(oid)
        # Duplicate-overlap IDs are frozen for this frame; allow initial registration only.
        if oid in set(getattr(self, "_freeze_dino_ema_obj_ids", set()) or set()):
            if prev is None:
                self._dino_embedding_history[oid] = cur
                return True
            return False
        if prev is None:
            self._dino_embedding_history[oid] = cur
            return True

        prev_n = self._normalize_embedding(prev)
        if prev_n is None:
            self._dino_embedding_history[oid] = cur
            return True

        alpha = float(np.clip(getattr(self, "dino_embed_ema_alpha", 0.3), 0.0, 1.0))
        mixed = alpha * cur + (1.0 - alpha) * prev_n
        mixed_n = self._normalize_embedding(mixed)
        self._dino_embedding_history[oid] = mixed_n if mixed_n is not None else prev_n
        return True

    def _prune_ambiguous_regions(self):
        stale_ids = [
            rid for rid, reg in self._ambiguous_regions.items()
            if self.frame_idx - reg["last_seen"] > self.ambiguous_max_age
        ]
        for rid in stale_ids:
            del self._ambiguous_regions[rid]

    def _update_ambiguous_region(self, box: List[int], kind: str) -> Optional[str]:
        """Track ambiguous regions and decide label after N confirmations."""
        match_id = None
        for rid, reg in self._ambiguous_regions.items():
            if bbox_iou(reg["box"], box) >= self.ambiguous_bbox_iou_threshold:
                match_id = rid
                break
        if match_id is None:
            match_id = self._ambiguous_next_id
            self._ambiguous_next_id += 1
            self._ambiguous_regions[match_id] = {
                "box": box,
                "hand_hits": 0,
                "obj_hits": 0,
                "created": self.frame_idx,
                "last_seen": self.frame_idx,
            }
        reg = self._ambiguous_regions[match_id]
        reg["box"] = box
        reg["last_seen"] = self.frame_idx
        if kind == "hand":
            reg["hand_hits"] += 1
        else:
            reg["obj_hits"] += 1
        if reg["hand_hits"] >= self.ambiguous_confirm_frames and reg["hand_hits"] >= reg["obj_hits"] + 1:
            return "hand"
        if reg["obj_hits"] >= self.ambiguous_confirm_frames and reg["obj_hits"] >= reg["hand_hits"] + 1:
            return "object"
        return None

    def _suppress_non_target_matches_for_same_object(
        self,
        detections: Dict[str, Any],
        target_contact_code: str,
    ) -> Dict[str, Any]:
        matches_raw = detections.get("matches", [])
        if not isinstance(matches_raw, list) or not matches_raw:
            detections["target_contact_code"] = str(target_contact_code)
            detections["suppressed_non_target_matches"] = 0
            detections["suppressed_conflicting_matches"] = 0
            detections["removed_conflicting_objects"] = 0
            detections["suppressed_conflicting_object_idxs"] = []
            return detections

        target_code = str(target_contact_code)
        objects_raw = detections.get("objects", [])
        if not isinstance(objects_raw, list):
            objects_raw = []

        obj_codes: Dict[int, Set[str]] = {}
        for match in matches_raw:
            if not isinstance(match, dict):
                continue
            obj_idx_val = match.get("object_idx", -1)
            try:
                obj_idx = int(obj_idx_val)
            except (TypeError, ValueError):
                continue
            if obj_idx < 0:
                continue
            code = str(match.get("contact_code") or "")
            if not code:
                continue
            obj_codes.setdefault(int(obj_idx), set()).add(code)

        conflicted_obj_idxs: Set[int] = set()
        for obj_idx, codes in obj_codes.items():
            has_target = target_code in codes
            has_non_target = any((code != target_code) for code in codes)
            if has_target and has_non_target:
                conflicted_obj_idxs.add(int(obj_idx))

        n_objects = len(objects_raw)
        if n_objects >= 2:
            for i in range(n_objects):
                box_i = objects_raw[i].get("bbox_xyxy") if isinstance(objects_raw[i], dict) else None
                if not box_i:
                    continue
                codes_i = obj_codes.get(int(i), set())
                has_i_target = target_code in codes_i
                has_i_non_target = any((code != target_code) for code in codes_i)
                for j in range(i + 1, n_objects):
                    box_j = objects_raw[j].get("bbox_xyxy") if isinstance(objects_raw[j], dict) else None
                    if not box_j:
                        continue
                    if bbox_iou(box_i, box_j) < self.ambiguous_bbox_iou_threshold:
                        continue
                    codes_j = obj_codes.get(int(j), set())
                    has_j_target = target_code in codes_j
                    has_j_non_target = any((code != target_code) for code in codes_j)
                    if (has_i_target and has_j_non_target) or (has_j_target and has_i_non_target):
                        conflicted_obj_idxs.add(int(i))
                        conflicted_obj_idxs.add(int(j))

        obj_idx_map: Dict[int, int] = {}
        filtered_objects: List[Dict[str, Any]] = []
        for old_idx, obj in enumerate(objects_raw):
            if int(old_idx) in conflicted_obj_idxs:
                continue
            obj_idx_map[int(old_idx)] = int(len(filtered_objects))
            filtered_objects.append(obj if isinstance(obj, dict) else {})

        suppressed_conflicting_matches = 0
        matches_after_conflict: List[Dict[str, Any]] = []
        for match in matches_raw:
            if not isinstance(match, dict):
                continue
            out_match = dict(match)
            obj_idx_val = out_match.get("object_idx", -1)
            try:
                obj_idx = int(obj_idx_val)
            except (TypeError, ValueError):
                obj_idx = -1

            if obj_idx >= 0 and obj_idx in conflicted_obj_idxs:
                out_match["object_idx"] = -1
                out_match["object_bbox_xyxy"] = None
                out_match["object_score"] = None
                out_match["suppressed_by_conflicting_contact"] = True
                suppressed_conflicting_matches += 1
            elif obj_idx >= 0:
                new_idx = obj_idx_map.get(int(obj_idx))
                if new_idx is None:
                    out_match["object_idx"] = -1
                    out_match["object_bbox_xyxy"] = None
                    out_match["object_score"] = None
                else:
                    out_match["object_idx"] = int(new_idx)
                    mapped_obj = filtered_objects[int(new_idx)] if int(new_idx) < len(filtered_objects) else None
                    if isinstance(mapped_obj, dict):
                        out_match["object_bbox_xyxy"] = mapped_obj.get("bbox_xyxy")
                        out_match["object_score"] = mapped_obj.get("score")
            matches_after_conflict.append(out_match)

        preferred_obj_idxs: Set[int] = set()
        for match in matches_after_conflict:
            obj_idx_val = match.get("object_idx", -1)
            try:
                obj_idx = int(obj_idx_val)
            except (TypeError, ValueError):
                continue
            if obj_idx < 0:
                continue
            if str(match.get("contact_code") or "") == target_code:
                preferred_obj_idxs.add(int(obj_idx))

        suppressed_non_target_matches = 0
        filtered_matches: List[Dict[str, Any]] = []
        for match in matches_after_conflict:
            out_match = dict(match)
            obj_idx_val = out_match.get("object_idx", -1)
            try:
                obj_idx = int(obj_idx_val)
            except (TypeError, ValueError):
                obj_idx = -1
            if (
                obj_idx >= 0
                and obj_idx in preferred_obj_idxs
                and str(out_match.get("contact_code") or "") != target_code
            ):
                out_match["object_idx"] = -1
                out_match["object_bbox_xyxy"] = None
                out_match["object_score"] = None
                out_match["suppressed_by_target_contact"] = True
                suppressed_non_target_matches += 1
            filtered_matches.append(out_match)

        detections["objects"] = filtered_objects
        detections["matches"] = filtered_matches
        detections["target_contact_code"] = target_code
        detections["suppressed_non_target_matches"] = int(suppressed_non_target_matches)
        detections["suppressed_conflicting_matches"] = int(suppressed_conflicting_matches)
        detections["removed_conflicting_objects"] = int(len(conflicted_obj_idxs))
        detections["suppressed_conflicting_object_idxs"] = sorted([int(x) for x in conflicted_obj_idxs])

        hand_to_object = detections.get("hand_to_object")
        if isinstance(hand_to_object, list):
            remapped = list(hand_to_object)
            for match in filtered_matches:
                hand_idx_val = match.get("hand_idx", -1)
                obj_idx_val = match.get("object_idx", -1)
                try:
                    hand_idx = int(hand_idx_val)
                except (TypeError, ValueError):
                    continue
                if hand_idx < 0 or hand_idx >= len(remapped):
                    continue
                try:
                    remapped[hand_idx] = int(obj_idx_val)
                except (TypeError, ValueError):
                    remapped[hand_idx] = -1
            detections["hand_to_object"] = remapped

        return detections
    
    def _keep_best_matching_component(self, mask: np.ndarray, obj_id: int) -> np.ndarray:
        """Keep the component that best matches historical masks (prevents drift)
        
        Args:
            mask: Boolean mask that may contain multiple disconnected components
            obj_id: Object ID to look up history
            
        Returns:
            Boolean mask with only the best matching component
        """
        if not np.any(mask):
            return mask

        # Find all connected components
        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        if num_labels <= 1:  # Only background, no components
            return mask
        
        # If no history, fall back to largest component
        if obj_id not in self.mask_history or len(self.mask_history[obj_id]) == 0:
            largest_label = 1
            largest_area = stats[1, cv2.CC_STAT_AREA]
            
            for label in range(2, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = label
            
            filtered_mask = (labels == largest_label)
            if num_labels > 2:
                removed_count = num_labels - 2
                print(f"[Hotrack] Removed {removed_count} component(s) (no history, kept largest {largest_area} px)")
            return filtered_mask
        
        # Search through history to find best matching component (use most recent frame)
        component_info = []
        hist_frame, hist_mask = self.mask_history[obj_id][-1]
        if hist_mask.shape != mask.shape:
            hist_mask = cv2.resize(
                hist_mask.astype(np.uint8),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_area = stats[label, cv2.CC_STAT_AREA]
            overlap = np.sum(np.logical_and(component_mask, hist_mask))
            ioa = overlap / max(component_area, 1)
            component_info.append({
                "label": label,
                "area": component_area,
                "ioa": float(ioa),
                "frame": hist_frame,
            })

        # Detector overlap: if a new detection box overlaps multiple components, keep full mask
        if obj_id >= 100 and self._current_target_boxes:
            comp_hits = {c["label"]: 0 for c in component_info}
            for box in self._current_target_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                box_mask = np.zeros_like(mask_uint8, dtype=bool)
                box_mask[y1 : y2 + 1, x1 : x2 + 1] = True
                for c in component_info:
                    comp_mask = (labels == c["label"])
                    _, ioa_c_to_box, ioa_box_to_c = calculate_ioa_bidirectional(comp_mask, box_mask)
                    if (
                        ioa_c_to_box >= self.detect_keep_ioa_threshold
                        and ioa_box_to_c >= self.detect_keep_ioa_threshold
                    ):
                        comp_hits[c["label"]] += 1
            if sum(1 for v in comp_hits.values() if v > 0) >= 2:
                print(f"[Hotrack] F{self.frame_idx} OBJ_DET_KEEP: obj-{obj_id-100} keep full mask (det overlap)")
                return mask
        
        # Find best component: filter out noise (too small), then choose by overlap pixels
        # Filter out components smaller than min_mask_area (likely noise)
        valid_components = [c for c in component_info if c["area"] >= self.min_mask_area]
        
        if len(valid_components) == 0:
            # All components are too small, fall back to largest area
            best_component = max(component_info, key=lambda x: x["area"])
            print(f"[Hotrack] WARNING: All components < {self.min_mask_area}px, using largest")
        else:
            if obj_id < 100:
                # Hands: favor area more than IoA (avoid tiny fragments winning)
                max_area = max(c["area"] for c in valid_components)
                def _hand_score(c):
                    area_norm = c["area"] / max(max_area, 1)
                    return 0.3 * c["ioa"] + 0.7 * area_norm
                best_component = max(valid_components, key=_hand_score)
            else:
                # Objects: prefer IoA, then area
                best_component = max(valid_components, key=lambda x: (x["ioa"], x["area"]))
        
        best_label = best_component["label"]
        
        # Create mask with only best matching component
        filtered_mask = (labels == best_label)
        
        # Log if we removed fragments
        if num_labels > 2:
            obj_type = "hand" if obj_id < 100 else "obj"
            obj_num = obj_id if obj_id < 100 else obj_id - 100
            print(f"[Hotrack] Component analysis for {obj_type}-{obj_num} ({num_labels-1} components):")
            
            # Sort by IoA descending, then area descending (for display)
            component_info.sort(key=lambda x: (x["ioa"], x["area"]), reverse=True)
            
            for i, comp in enumerate(component_info):
                status = "KEPT" if comp["label"] == best_label else "REMOVED"
                frames_back = self.frame_idx - comp["frame"] if comp["frame"] is not None else 0
                hist_desc = f"{frames_back}F back" if frames_back > 0 else "prev"
                print(f"  - Component {comp['label']}: area={comp['area']} px, IoA={comp['ioa']:.2f} (match {hist_desc}) → {status}")
        
        return filtered_mask

    def _is_near_hand_dilated(self, mask: np.ndarray, hand_masks: List[np.ndarray]) -> bool:
        """Check if mask overlaps with dilated hand region"""
        if len(hand_masks) == 0:
            return False
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.hand_dilation_kernel, self.hand_dilation_kernel)
        )
        
        for hand_mask in hand_masks:
            hand_uint8 = hand_mask.astype(np.uint8) * 255
            dilated = cv2.dilate(hand_uint8, kernel, iterations=1)
            dilated_bool = dilated > 0
            
            if np.any(np.logical_and(mask, dilated_bool)):
                return True
        
        return False
    
    
    def _compute_quality_score(self, mask: np.ndarray, bbox: List[int], 
                               hand_masks: List[np.ndarray]) -> float:
        """Compute quality score for candidate (higher = better)"""
        score = 0.0
        
        mask_area = np.sum(mask)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = mask.shape[0] * mask.shape[1]
        
        # Prefer reasonable size
        score += (mask_area / image_area) * 0.3
        
        # Prefer compact (tight) masks
        if bbox_area > 0:
            compactness = mask_area / bbox_area
            score += compactness * 0.4
        
        # Penalize hand overlap
        max_hand_overlap = 0.0
        for hand_mask in hand_masks:
            intersection = np.logical_and(mask, hand_mask)
            overlap_ratio = np.sum(intersection) / max(mask_area, 1)
            max_hand_overlap = max(max_hand_overlap, overlap_ratio)
        score -= max_hand_overlap * 0.3
        
        return score
    
    def _check_hand_overlap(self, mask: np.ndarray, hand_masks: List[np.ndarray]) -> Tuple[bool, str]:
        """Check if mask is likely hand (strong overlap)
        
        Returns:
            (is_hand, reason) - (True, reason) if likely hand
        """
        for hand_mask in hand_masks:
            iou, ioa_hand_to_mask, ioa_mask_to_hand = calculate_ioa_bidirectional(hand_mask, mask)
            
            # Either direction high overlap => likely hand
            if ioa_hand_to_mask >= self.hand_overlap_reject:
                return True, f"hand_overlap_h2m={ioa_hand_to_mask:.2f}"
            if ioa_mask_to_hand >= self.hand_overlap_reject:
                return True, f"hand_overlap_m2h={ioa_mask_to_hand:.2f}"
        
        return False, ""
    
    def _check_duplicate(
        self,
        mask: np.ndarray,
        bbox: List[int],
        existing_masks: Dict[int, np.ndarray],
        existing_boxes: Dict[int, List[int]],
        *,
        ioa_threshold: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """AGGRESSIVE duplicate check (if slightly duplicate, reject)
        
        Returns:
            (is_duplicate, reason) - (True, reason) if duplicate
        """
        dup_id = self._find_duplicate_object(mask, existing_masks, ioa_threshold=ioa_threshold)
        if dup_id is None:
            return False, ""
        exist_mask = existing_masks.get(dup_id)
        if exist_mask is None:
            return True, f"duplicate_with_{dup_id}"
        _, ioa_exist_to_new, ioa_new_to_exist = calculate_ioa_bidirectional(exist_mask, mask)
        return True, f"ioa_e2n={ioa_exist_to_new:.2f}_ioa_n2e={ioa_new_to_exist:.2f}_with_{dup_id}"

    def _find_duplicate_object(
        self,
        mask: np.ndarray,
        existing_masks: Dict[int, np.ndarray],
        *,
        ioa_threshold: Optional[float] = None,
    ) -> Optional[int]:
        if mask is None or np.count_nonzero(mask) == 0:
            return None
        threshold = self.dup_ioa_lo
        if ioa_threshold is not None:
            threshold = float(np.clip(ioa_threshold, 0.0, 1.0))
        for obj_id, exist_mask in existing_masks.items():
            if exist_mask is None or np.count_nonzero(exist_mask) == 0:
                continue
            # IoA-based overlap (both directions)
            _, ioa_exist_to_new, ioa_new_to_exist = calculate_ioa_bidirectional(exist_mask, mask)
            # Duplicate only when both directions are high
            if ioa_exist_to_new >= threshold and ioa_new_to_exist >= threshold:
                return int(obj_id)
        return None

    def _find_temporal_reassign_target(
        self,
        new_mask: np.ndarray,
        current_object_masks: Dict[int, np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_temporal_id_reassign:
            return None
        if new_mask is None or np.count_nonzero(new_mask) == 0:
            return None

        new_area = float(np.sum(new_mask))
        min_prev_area = int(max(self.temporal_reassign_min_prev_area, self.min_mask_area))
        best_match: Optional[Dict[str, Any]] = None

        for obj_id, cur_mask in current_object_masks.items():
            oid = int(obj_id)
            if oid < 100:
                continue
            prev_mask = self._prev_tracked_masks.get(oid)
            if prev_mask is None or np.count_nonzero(prev_mask) == 0:
                continue
            if prev_mask.shape != new_mask.shape:
                prev_mask = cv2.resize(
                    prev_mask.astype(np.uint8),
                    (new_mask.shape[1], new_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if cur_mask is None:
                continue
            if cur_mask.shape != new_mask.shape:
                cur_mask = cv2.resize(
                    cur_mask.astype(np.uint8),
                    (new_mask.shape[1], new_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            prev_area = float(np.sum(prev_mask))
            cur_area = float(np.sum(cur_mask))
            if prev_area < float(min_prev_area):
                continue
            if cur_area <= 0.0:
                continue
            if cur_area > prev_area * float(self.temporal_reassign_shrink_ratio):
                continue
            if new_area < cur_area * float(self.temporal_reassign_new_vs_cur_ratio):
                continue

            _, ioa_prev_to_new, ioa_new_to_prev = calculate_ioa_bidirectional(prev_mask, new_mask)
            if ioa_prev_to_new < float(self.temporal_reassign_prev_to_new_ioa):
                continue
            if ioa_new_to_prev < float(self.temporal_reassign_new_to_prev_ioa):
                continue

            score = float(min(ioa_prev_to_new, ioa_new_to_prev))
            if best_match is None or score > float(best_match.get("score", -1.0)):
                best_match = {
                    "obj_id": int(oid),
                    "score": float(score),
                    "prev_area": float(prev_area),
                    "cur_area": float(cur_area),
                    "new_area": float(new_area),
                    "ioa_prev_to_new": float(ioa_prev_to_new),
                    "ioa_new_to_prev": float(ioa_new_to_prev),
                }

        return best_match

    def _apply_temporal_reassign(
        self,
        *,
        target_id: int,
        replacement_mask: np.ndarray,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Optional[Dict[int, np.ndarray]] = None,
        object_boxes: Optional[Dict[int, List[int]]] = None,
        accepted_in_frame: Optional[Dict[int, np.ndarray]] = None,
        temp_obj_id: Optional[int] = None,
        source: str = "temporal_reassign",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        target_id = int(target_id)
        if replacement_mask is None or np.count_nonzero(replacement_mask) == 0:
            return

        if temp_obj_id is not None:
            temp_obj_id = int(temp_obj_id)
            if temp_obj_id >= 100:
                try:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state,
                        temp_obj_id,
                        strict=False,
                        need_output=False,
                    )
                except Exception:
                    pass
                tracked_masks.pop(temp_obj_id, None)
                tracked_boxes.pop(temp_obj_id, None)
                self.mask_history.pop(temp_obj_id, None)
                if object_masks is not None:
                    object_masks.pop(temp_obj_id, None)
                if object_boxes is not None:
                    object_boxes.pop(temp_obj_id, None)

        ret = self.sam2_tracker.add_new_mask(
            self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=target_id,
            mask=replacement_mask,
        )
        if isinstance(ret, (tuple, list)) and len(ret) >= 2:
            self.all_ids = ret[1]

        bbox = self._get_mask_bbox(replacement_mask)
        tracked_masks[target_id] = replacement_mask
        tracked_boxes[target_id] = bbox
        self.mask_history[target_id] = deque(
            [(self.frame_idx, replacement_mask.copy())],
            maxlen=self.mask_history_length,
        )
        if object_masks is not None:
            object_masks[target_id] = replacement_mask
        if object_boxes is not None:
            object_boxes[target_id] = bbox
        if accepted_in_frame is not None:
            accepted_in_frame[target_id] = replacement_mask

        evt: Dict[str, Any] = {
            "type": "temporal_reassign",
            "frame": int(self.frame_idx),
            "obj_id": int(target_id),
            "temp_id": None if temp_obj_id is None else int(temp_obj_id),
            "source": str(source),
            "area": int(np.sum(replacement_mask)),
        }
        if isinstance(metrics, dict):
            evt.update({k: v for k, v in metrics.items()})
        self.events.append(evt)
        print(
            f"[Hotrack] F{self.frame_idx} EVENT_TEMPORAL_REASSIGN: "
            f"obj-{target_id-100} source={source}"
        )

    @staticmethod
    def _mask_in_box_ratio(mask: np.ndarray, box: List[int]) -> float:
        if mask is None or box is None:
            return 0.0
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        area = float(np.sum(mask))
        if area <= 0.0:
            return 0.0
        crop = mask[y1:y2, x1:x2]
        return float(np.sum(crop)) / area

    def _best_det_box_match_for_component(
        self,
        component_mask: Optional[np.ndarray],
        det_boxes: List[List[int]],
    ) -> Dict[str, Any]:
        """Find best detector box for a component using strict containment only."""
        cm = _to_bool_mask(component_mask)
        if cm is None or np.count_nonzero(cm) <= 0:
            return {
                "matched": False,
                "det_idx": -1,
                "strict_in_ratio": 0.0,
                "expanded_in_ratio": 0.0,
                "score": 0.0,
                "source": "none",
            }
        if not det_boxes:
            return {
                "matched": False,
                "det_idx": -1,
                "strict_in_ratio": 0.0,
                "expanded_in_ratio": 0.0,
                "score": 0.0,
                "source": "no_boxes",
            }

        best_idx = -1
        best_strict = 0.0
        for idx, box in enumerate(list(det_boxes or [])):
            strict_in = float(self._mask_in_box_ratio(cm, box))
            if float(strict_in) > float(best_strict):
                best_idx = int(idx)
                best_strict = float(strict_in)

        matched_strict = bool(best_strict >= float(self.split_same_box_in_ratio_th))
        matched = bool(matched_strict)
        if matched_strict:
            source = "strict_box"
            score = float(best_strict)
        else:
            source = "none"
            score = float(best_strict)
        return {
            "matched": bool(matched),
            "det_idx": int(best_idx if matched else -1),
            "strict_in_ratio": float(best_strict),
            "expanded_in_ratio": 0.0,
            "score": float(score),
            "source": str(source),
        }

    @staticmethod
    def _mask_in_mask_ratio(src_mask: Optional[np.ndarray], dst_mask: Optional[np.ndarray]) -> float:
        if src_mask is None or dst_mask is None:
            return 0.0
        src = np.asarray(src_mask).astype(bool)
        dst = np.asarray(dst_mask).astype(bool)
        if src.shape != dst.shape:
            return 0.0
        src_area = float(np.count_nonzero(src))
        if src_area <= 0.0:
            return 0.0
        inter = float(np.count_nonzero(np.logical_and(src, dst)))
        return float(inter / src_area)

    def _is_id_recovery_blocked(self, obj_id: int, frame_idx: Optional[int] = None) -> bool:
        fi = int(self.frame_idx if frame_idx is None else frame_idx)
        until = self._struct_id_recovery_block_until.get(int(obj_id))
        if until is None:
            return False
        return bool(fi <= int(until))

    def _set_struct_id_recovery_block(self, obj_ids: List[int], frame_idx: int) -> None:
        ttl = int(max(0, int(self.id_recovery_block_after_struct_frames)))
        until = int(frame_idx) + int(ttl)
        self._struct_recovery_block_until_frame = int(max(int(self._struct_recovery_block_until_frame), int(until)))
        for oid in list(obj_ids or []):
            oid_i = int(oid)
            prev = self._struct_id_recovery_block_until.get(oid_i)
            if prev is None:
                self._struct_id_recovery_block_until[oid_i] = int(until)
            else:
                self._struct_id_recovery_block_until[oid_i] = int(max(int(prev), int(until)))
            prev_evt = self._struct_event_last_frame.get(oid_i)
            if prev_evt is None:
                self._struct_event_last_frame[oid_i] = int(frame_idx)
            else:
                self._struct_event_last_frame[oid_i] = int(max(int(prev_evt), int(frame_idx)))

    @staticmethod
    def _connected_components_count(mask: Optional[np.ndarray]) -> int:
        if mask is None:
            return 0
        m = np.asarray(mask).astype(np.uint8)
        if m.size == 0 or np.count_nonzero(m) <= 0:
            return 0
        try:
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        except Exception:
            return 0
        return int(max(0, int(num_labels) - 1))

    def _apply_split_transaction(
        self,
        *,
        parent_id: int,
        child_masks: List[np.ndarray],
        reason: str,
        score: float,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not bool(self.enable_structural_ops):
            return None
        parent_id = int(parent_id)
        valid_masks: List[np.ndarray] = []
        for m in list(child_masks or []):
            mm = _to_bool_mask(m)
            if mm is None or np.count_nonzero(mm) <= 0:
                continue
            valid_masks.append(mm.astype(bool))
        if len(valid_masks) != 2:
            return None

        child_ids: List[int] = []
        for cmask in valid_masks:
            new_id = int(self.next_obj_id)
            self.next_obj_id += 1
            try:
                ret = self.sam2_tracker.add_new_mask(
                    self.inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(new_id),
                    mask=cmask,
                )
                if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                    self.all_ids = ret[1]
                elif int(new_id) not in self.all_ids:
                    self.all_ids.append(int(new_id))
            except Exception:
                return None
            cbox = self._get_mask_bbox(cmask)
            tracked_masks[int(new_id)] = cmask
            tracked_boxes[int(new_id)] = cbox
            object_masks[int(new_id)] = cmask
            object_boxes[int(new_id)] = cbox
            self.mask_history[int(new_id)] = deque(
                [(int(frame_idx), cmask.copy())],
                maxlen=self.mask_history_length,
            )
            new_objects.append({
                "obj_id": int(new_id),
                "bbox": cbox,
                "source": "split_reid",
            })
            child_ids.append(int(new_id))

        if int(parent_id) in self.all_ids:
            try:
                self.all_ids, _ = self.sam2_tracker.remove_object(
                    self.inference_state,
                    int(parent_id),
                    strict=False,
                    need_output=False,
                )
            except Exception:
                pass
        if int(parent_id) in self.all_ids:
            self.all_ids = [int(x) for x in self.all_ids if int(x) != int(parent_id)]
        tracked_masks.pop(int(parent_id), None)
        tracked_boxes.pop(int(parent_id), None)
        object_masks.pop(int(parent_id), None)
        object_boxes.pop(int(parent_id), None)
        self.mask_history.pop(int(parent_id), None)
        self.confusion_objects_history.pop(int(parent_id), None)
        self._retired_struct_ids.add(int(parent_id))

        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        struct_event = {
            "type": "split_apply",
            "frame": int(frame_idx),
            "parents": [int(parent_id)],
            "children": [int(x) for x in child_ids],
            "reason": str(reason),
            "score": float(score_f),
        }
        self._struct_events_frame.append(dict(struct_event))
        self.events.append(dict(struct_event))
        self._id_transitions_frame.append({
            "type": "split",
            "frame": int(frame_idx),
            "from": [int(parent_id)],
            "to": [int(x) for x in child_ids],
            "reason": str(reason),
            "score": float(score_f),
        })
        self._set_struct_id_recovery_block([int(parent_id)] + [int(x) for x in child_ids], int(frame_idx))

        return struct_event

    def _emit_struct_split_keep_parent(
        self,
        *,
        parent_id: int,
        child_id: int,
        reason: str,
        score: float,
        frame_idx: int,
        confirmed_tail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parent_i = int(parent_id)
        child_i = int(child_id)
        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        struct_event = {
            "type": "split_apply",
            "frame": int(frame_idx),
            "parents": [int(parent_i)],
            "children": [int(parent_i), int(child_i)],
            "reason": str(reason),
            "score": float(score_f),
        }
        if isinstance(confirmed_tail, dict):
            struct_event["confirmed_tail"] = dict(confirmed_tail)
        self._struct_events_frame.append(dict(struct_event))
        self.events.append(dict(struct_event))
        self._id_transitions_frame.append({
            "type": "split",
            "frame": int(frame_idx),
            "from": [int(parent_i)],
            "to": [int(parent_i), int(child_i)],
            "reason": str(reason),
            "score": float(score_f),
        })
        self._set_struct_id_recovery_block([int(parent_i), int(child_i)], int(frame_idx))

        return struct_event

    def _apply_split_reid_from_candidate(
        self,
        *,
        parent_id: int,
        child_id: int,
        reason: str,
        score: float,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
        confirmed_tail: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Confirm split and reassign fresh IDs to both split components."""
        if not bool(self.enable_structural_ops):
            return None
        parent_i = int(parent_id)
        child_i = int(child_id)
        if int(parent_i) == int(child_i):
            return None

        parent_mask = _to_bool_mask(tracked_masks.get(int(parent_i)))
        child_mask = _to_bool_mask(tracked_masks.get(int(child_i)))
        if (
            parent_mask is None
            or child_mask is None
            or np.count_nonzero(parent_mask) <= 0
            or np.count_nonzero(child_mask) <= 0
        ):
            return None

        keep_mask = np.logical_and(parent_mask.astype(bool), np.logical_not(child_mask.astype(bool)))
        split_mask = child_mask.astype(bool)
        if np.count_nonzero(keep_mask) <= 0 or np.count_nonzero(split_mask) <= 0:
            return None

        child_ids: List[int] = []
        child_masks_apply = [keep_mask.astype(bool), split_mask.astype(bool)]
        for cmask in child_masks_apply:
            new_id = int(self.next_obj_id)
            self.next_obj_id += 1
            try:
                ret = self.sam2_tracker.add_new_mask(
                    self.inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(new_id),
                    mask=cmask,
                )
                if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                    self.all_ids = ret[1]
                elif int(new_id) not in self.all_ids:
                    self.all_ids.append(int(new_id))
            except Exception:
                return None

            cbox = self._get_mask_bbox(cmask)
            tracked_masks[int(new_id)] = cmask
            tracked_boxes[int(new_id)] = cbox
            object_masks[int(new_id)] = cmask
            object_boxes[int(new_id)] = cbox
            self.mask_history[int(new_id)] = deque(
                [(int(frame_idx), cmask.copy())],
                maxlen=self.mask_history_length,
            )
            new_objects.append({
                "obj_id": int(new_id),
                "bbox": cbox,
                "source": "split_reid",
            })
            child_ids.append(int(new_id))

        for oid in [int(parent_i), int(child_i)]:
            if int(oid) in self.all_ids:
                try:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state,
                        int(oid),
                        strict=False,
                        need_output=False,
                    )
                except Exception:
                    pass
            if int(oid) in self.all_ids:
                self.all_ids = [int(x) for x in self.all_ids if int(x) != int(oid)]
            tracked_masks.pop(int(oid), None)
            tracked_boxes.pop(int(oid), None)
            object_masks.pop(int(oid), None)
            object_boxes.pop(int(oid), None)
            self.mask_history.pop(int(oid), None)
            self.confusion_objects_history.pop(int(oid), None)
            self._retired_struct_ids.add(int(oid))

        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        struct_event = {
            "type": "split_apply",
            "frame": int(frame_idx),
            "parents": [int(parent_i)],
            "children": [int(x) for x in child_ids],
            "reason": str(reason),
            "score": float(score_f),
        }
        if isinstance(confirmed_tail, dict):
            struct_event["confirmed_tail"] = dict(confirmed_tail)
        self._struct_events_frame.append(dict(struct_event))
        self.events.append(dict(struct_event))
        self._id_transitions_frame.append({
            "type": "split",
            "frame": int(frame_idx),
            "from": [int(parent_i)],
            "to": [int(x) for x in child_ids],
            "reason": str(reason),
            "score": float(score_f),
        })
        self._set_struct_id_recovery_block(
            [int(parent_i), int(child_i)] + [int(x) for x in child_ids],
            int(frame_idx),
        )

        return struct_event

    def _drop_object_id(
        self,
        *,
        obj_id: int,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: Optional[List[Dict[str, Any]]] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Remove object ID from tracker/state and local frame outputs."""
        oid = int(obj_id)
        if int(oid) < 100:
            return False

        removed_any = False
        if int(oid) in self.all_ids:
            try:
                self.all_ids, _ = self.sam2_tracker.remove_object(
                    self.inference_state,
                    int(oid),
                    strict=False,
                    need_output=False,
                )
                removed_any = True
            except Exception:
                pass
        if int(oid) in self.all_ids:
            self.all_ids = [int(x) for x in self.all_ids if int(x) != int(oid)]

        if int(oid) in tracked_masks:
            tracked_masks.pop(int(oid), None)
            removed_any = True
        tracked_boxes.pop(int(oid), None)
        object_masks.pop(int(oid), None)
        object_boxes.pop(int(oid), None)
        self.mask_history.pop(int(oid), None)
        self.confusion_objects_history.pop(int(oid), None)
        self._pending_split_from_new.pop(int(oid), None)

        if isinstance(new_objects, list):
            new_objects[:] = [
                row for row in list(new_objects)
                if not (isinstance(row, dict) and int(row.get("obj_id", -1)) == int(oid))
            ]

        if removed_any:
            self._retired_struct_ids.add(int(oid))
            if isinstance(reason, str) and reason:
                self.events.append({
                    "type": "split_cluster_drop_child",
                    "frame": int(frame_idx),
                    "obj_id": int(oid),
                    "reason": str(reason),
                })
        return bool(removed_any)

    def _apply_split_confirm_existing_children(
        self,
        *,
        parent_id: int,
        child_ids: List[int],
        reason: str,
        score: float,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
        confirmed_tail: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Confirm split using already-created child IDs (atomic parent removal)."""
        parent_i = int(parent_id)
        if int(parent_i) < 100:
            return None
        parent_mask = _to_bool_mask(tracked_masks.get(int(parent_i)))
        if parent_mask is None or np.count_nonzero(parent_mask) <= 0:
            return None

        valid_children: List[int] = []
        seen: Set[int] = set()
        for cid_raw in list(child_ids or []):
            try:
                cid = int(cid_raw)
            except Exception:
                continue
            if int(cid) < 100 or int(cid) == int(parent_i) or int(cid) in seen:
                continue
            cm = _to_bool_mask(tracked_masks.get(int(cid)))
            if cm is None or np.count_nonzero(cm) <= 0:
                continue
            valid_children.append(int(cid))
            seen.add(int(cid))
        if len(valid_children) < 2:
            return None

        self._drop_object_id(
            obj_id=int(parent_i),
            frame_idx=int(frame_idx),
            tracked_masks=tracked_masks,
            tracked_boxes=tracked_boxes,
            object_masks=object_masks,
            object_boxes=object_boxes,
            new_objects=new_objects,
            reason="split_cluster_parent_remove",
        )

        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        out_children = [int(x) for x in sorted(valid_children)]
        struct_event = {
            "type": "split_apply",
            "frame": int(frame_idx),
            "parents": [int(parent_i)],
            "children": out_children,
            "reason": str(reason),
            "score": float(score_f),
        }
        if isinstance(confirmed_tail, dict):
            struct_event["confirmed_tail"] = dict(confirmed_tail)
        self._struct_events_frame.append(dict(struct_event))
        self.events.append(dict(struct_event))
        self._id_transitions_frame.append({
            "type": "split",
            "frame": int(frame_idx),
            "from": [int(parent_i)],
            "to": out_children,
            "reason": str(reason),
            "score": float(score_f),
        })
        self._set_struct_id_recovery_block([int(parent_i)] + out_children, int(frame_idx))

        return struct_event

    def _revert_unconfirmed_split_candidate(
        self,
        *,
        parent_id: int,
        child_id: int,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
        reason: str,
    ) -> bool:
        """Rollback provisional child when split confirmation fails."""
        parent_i = int(parent_id)
        child_i = int(child_id)

        parent_mask = _to_bool_mask(tracked_masks.get(int(parent_i)))
        child_mask = _to_bool_mask(tracked_masks.get(int(child_i)))
        if child_mask is None or np.count_nonzero(child_mask) <= 0:
            return False

        if parent_mask is not None and parent_mask.shape == child_mask.shape and np.count_nonzero(parent_mask) > 0:
            union_mask = np.logical_or(parent_mask.astype(bool), child_mask.astype(bool))
        elif parent_mask is not None and np.count_nonzero(parent_mask) > 0:
            union_mask = parent_mask.astype(bool)
        else:
            union_mask = child_mask.astype(bool)

        if np.count_nonzero(union_mask) > 0:
            try:
                ret = self.sam2_tracker.add_new_mask(
                    self.inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(parent_i),
                    mask=union_mask.astype(bool),
                )
                if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                    self.all_ids = ret[1]
                elif int(parent_i) not in self.all_ids:
                    self.all_ids.append(int(parent_i))
            except Exception:
                pass
            pbox = self._get_mask_bbox(union_mask)
            tracked_masks[int(parent_i)] = union_mask.astype(bool)
            tracked_boxes[int(parent_i)] = pbox
            object_masks[int(parent_i)] = union_mask.astype(bool)
            object_boxes[int(parent_i)] = pbox
            self.mask_history[int(parent_i)] = deque(
                [(int(frame_idx), union_mask.astype(bool).copy())],
                maxlen=self.mask_history_length,
            )

        if int(child_i) in self.all_ids:
            try:
                self.all_ids, _ = self.sam2_tracker.remove_object(
                    self.inference_state,
                    int(child_i),
                    strict=False,
                    need_output=False,
                )
            except Exception:
                pass
        if int(child_i) in self.all_ids:
            self.all_ids = [int(x) for x in self.all_ids if int(x) != int(child_i)]

        tracked_masks.pop(int(child_i), None)
        tracked_boxes.pop(int(child_i), None)
        object_masks.pop(int(child_i), None)
        object_boxes.pop(int(child_i), None)
        self.mask_history.pop(int(child_i), None)
        self.confusion_objects_history.pop(int(child_i), None)
        if isinstance(new_objects, list):
            new_objects[:] = [
                row for row in list(new_objects)
                if not (isinstance(row, dict) and int(row.get("obj_id", -1)) == int(child_i))
            ]

        self.events.append({
            "type": "split_candidate_revert",
            "frame": int(frame_idx),
            "parent": int(parent_i),
            "child": int(child_i),
            "reason": str(reason),
        })
        print(
            f"[Hotrack] F{frame_idx} OBJ_SPLIT_CANDIDATE_REVERT: "
            f"parent=obj-{parent_i-100} child=obj-{child_i-100} reason={reason}"
        )
        return True

    def _pick_split_parent_for_new_object(
        self,
        *,
        child_id: int,
        child_mask: np.ndarray,
        tracked_masks: Dict[int, np.ndarray],
        relaxed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        child_i = int(child_id)
        child_b = _to_bool_mask(child_mask)
        if child_b is None or np.count_nonzero(child_b) <= 0:
            return None

        min_child_in_prev = float(np.clip(self.split_new_parent_min_child_in_prev, 0.0, 1.0))
        min_overlap_now = float(np.clip(self.split_new_parent_min_overlap_now, 0.0, 1.0))
        min_score = float(np.clip(self.split_new_parent_score_th, 0.0, 1.0))
        best: Optional[Dict[str, Any]] = None

        for parent_id_raw, parent_mask_raw in list((tracked_masks or {}).items()):
            try:
                parent_i = int(parent_id_raw)
            except Exception:
                continue
            if int(parent_i) < 100 or int(parent_i) == int(child_i):
                continue
            if int(self._id_birth_frame.get(int(parent_i), -1)) >= int(self.frame_idx):
                continue

            parent_b = _to_bool_mask(parent_mask_raw)
            if parent_b is None or np.count_nonzero(parent_b) <= 0:
                continue
            if parent_b.shape != child_b.shape:
                parent_b = cv2.resize(
                    parent_b.astype(np.uint8),
                    (child_b.shape[1], child_b.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if np.count_nonzero(parent_b) <= 0:
                continue

            _, ioa_p_to_c, ioa_c_to_p = calculate_ioa_bidirectional(parent_b, child_b)
            overlap_now = float(max(ioa_p_to_c, ioa_c_to_p))

            child_in_prev = 0.0
            prev_parent = _to_bool_mask(self._prev_tracked_masks.get(int(parent_i)))
            if prev_parent is not None and np.count_nonzero(prev_parent) > 0:
                if prev_parent.shape != child_b.shape:
                    prev_parent = cv2.resize(
                        prev_parent.astype(np.uint8),
                        (child_b.shape[1], child_b.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                if np.count_nonzero(prev_parent) > 0:
                    child_in_prev = float(self._mask_in_mask_ratio(child_b, prev_parent))

            if (not bool(relaxed)) and overlap_now < min_overlap_now and child_in_prev < min_child_in_prev:
                continue

            union_mask = np.logical_or(parent_b, child_b).astype(bool)
            union_components = int(self._connected_components_count(union_mask))
            union_single = 1.0 if int(union_components) <= 1 else 0.0
            score = float(
                np.clip(
                    0.60 * float(child_in_prev)
                    + 0.30 * float(overlap_now)
                    + 0.10 * float(union_single),
                    0.0,
                    1.0,
                )
            )
            if (not bool(relaxed)) and score < min_score:
                continue

            row = {
                "parent_id": int(parent_i),
                "score": float(score),
                "child_in_prev": float(child_in_prev),
                "overlap_now": float(overlap_now),
                "union_components": int(union_components),
            }
            if best is None:
                best = row
                continue
            rank_new = (
                float(row.get("score", 0.0)),
                float(row.get("child_in_prev", 0.0)),
                -int(self._id_birth_frame.get(int(parent_i), self.frame_idx)),
            )
            rank_old = (
                float(best.get("score", 0.0)),
                float(best.get("child_in_prev", 0.0)),
                -int(self._id_birth_frame.get(int(best.get("parent_id", -1)), self.frame_idx)),
            )
            if rank_new > rank_old:
                best = row
        return best

    def _seed_pending_split_from_new_objects(
        self,
        *,
        new_objects: List[Dict[str, Any]],
        tracked_masks: Dict[int, np.ndarray],
    ) -> None:
        if not bool(self.enable_structural_ops):
            return
        if not bool(self.split_register_all_new_objects):
            return
        if not isinstance(new_objects, list) or not new_objects:
            return

        for row in list(new_objects):
            if not isinstance(row, dict):
                continue
            try:
                child_i = int(row.get("obj_id", -1))
            except Exception:
                continue
            if int(child_i) < 100:
                continue
            if int(child_i) in self._pending_split_from_new:
                continue
            if int(child_i) in self._retired_struct_ids:
                continue

            source = str(row.get("source") or "").strip().lower()
            # All detector-originated new objects should be validated as
            # split candidates (user policy): plain new_object, det, and
            # multicomponent_split.
            if source not in {"new_object", "det", "multicomponent_split"}:
                continue

            child_mask = _to_bool_mask(tracked_masks.get(int(child_i)))
            if child_mask is None or np.count_nonzero(child_mask) <= 0:
                continue

            parent_pick = self._pick_split_parent_for_new_object(
                child_id=int(child_i),
                child_mask=child_mask,
                tracked_masks=tracked_masks,
            )
            seed_reason = "split_seed_from_new_object"
            parent_pick_relaxed = False
            if (
                not isinstance(parent_pick, dict)
                and bool(getattr(self, "split_seed_relaxed_when_parent_missing", True))
            ):
                parent_pick = self._pick_split_parent_for_new_object(
                    child_id=int(child_i),
                    child_mask=child_mask,
                    tracked_masks=tracked_masks,
                    relaxed=True,
                )
                if isinstance(parent_pick, dict):
                    parent_pick_relaxed = True
                    seed_reason = "split_seed_from_new_object_relaxed"
            parent_i = -1
            score_hint = 0.0
            parent_select_score = 0.0
            parent_select_child_in_prev = 0.0
            parent_select_overlap_now = 0.0
            parent_select_union_components = 0
            if isinstance(parent_pick, dict):
                parent_i = int(parent_pick.get("parent_id", -1))
                if int(parent_i) >= 100 and int(parent_i) != int(child_i):
                    pair_key = (int(min(parent_i, child_i)), int(max(parent_i, child_i)))
                    if pair_key in self._emitted_struct_split_pairs:
                        continue
                    score_hint = float(parent_pick.get("score", 0.0) or 0.0)
                    parent_select_score = float(parent_pick.get("score", 0.0) or 0.0)
                    parent_select_child_in_prev = float(parent_pick.get("child_in_prev", 0.0) or 0.0)
                    parent_select_overlap_now = float(parent_pick.get("overlap_now", 0.0) or 0.0)
                    parent_select_union_components = int(parent_pick.get("union_components", 0) or 0)
                else:
                    parent_i = -1
            if int(parent_i) < 100:
                seed_reason = "split_seed_from_new_object_all_parents"
            seed_parent_i = int(parent_i)
            try:
                component_parent_i = int(row.get("component_parent", -1))
            except Exception:
                component_parent_i = -1
            if int(component_parent_i) >= 100:
                seed_parent_i = int(component_parent_i)

            self._pending_split_from_new[int(child_i)] = {
                "parent_id": int(parent_i),
                "seed_parent_id": int(seed_parent_i),
                "start_frame": int(self.frame_idx),
                "last_frame": int(self.frame_idx),
                "score": float(score_hint),
                "reason": str(seed_reason),
                "source": "new_object",
                "parent_select_score": float(parent_select_score),
                "parent_select_child_in_prev": float(parent_select_child_in_prev),
                "parent_select_overlap_now": float(parent_select_overlap_now),
                "parent_select_union_components": int(parent_select_union_components),
                "parent_select_relaxed": bool(parent_pick_relaxed),
            }
            self.events.append({
                "type": "split_candidate_seed",
                "frame": int(self.frame_idx),
                "parent": int(parent_i) if int(parent_i) >= 100 else None,
                "child": int(child_i),
                "reason": str(seed_reason),
                "score": float(score_hint),
                "parent_select_relaxed": bool(parent_pick_relaxed),
            })
            parent_txt = f"obj-{parent_i-100}" if int(parent_i) >= 100 else "all-parents"
            print(
                f"[Hotrack] F{self.frame_idx} OBJ_SPLIT_CANDIDATE_SEED: "
                f"parent={parent_txt} child=obj-{child_i-100} "
                f"score={score_hint:.2f} relaxed={bool(parent_pick_relaxed)}"
            )

    def _apply_keep_and_new_transaction(
        self,
        *,
        parent_id: int,
        keep_mask: np.ndarray,
        new_mask: np.ndarray,
        reason: str,
        score: float,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
        new_obj_meta: Optional[Dict[str, Any]] = None,
        emit_struct_split: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Keep parent on main component and spawn one new child object.

        When `emit_struct_split` is True, also emit structural split metadata
        without replacing the parent ID.
        """
        parent_id = int(parent_id)
        keep_b = _to_bool_mask(keep_mask)
        new_b = _to_bool_mask(new_mask)
        if keep_b is None or new_b is None:
            return None
        keep_b = keep_b.astype(bool)
        new_b = new_b.astype(bool)
        if np.count_nonzero(keep_b) <= 0 or np.count_nonzero(new_b) <= 0:
            return None

        # Parent keeps old ID on the component that best explains previous frame.
        try:
            ret_keep = self.sam2_tracker.add_new_mask(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(parent_id),
                mask=keep_b,
            )
            if isinstance(ret_keep, (tuple, list)) and len(ret_keep) >= 2:
                self.all_ids = ret_keep[1]
            elif int(parent_id) not in self.all_ids:
                self.all_ids.append(int(parent_id))
        except Exception:
            return None

        keep_box = self._get_mask_bbox(keep_b)
        tracked_masks[int(parent_id)] = keep_b
        tracked_boxes[int(parent_id)] = keep_box
        object_masks[int(parent_id)] = keep_b
        object_boxes[int(parent_id)] = keep_box
        self.mask_history[int(parent_id)] = deque(
            [(int(frame_idx), keep_b.copy())],
            maxlen=self.mask_history_length,
        )

        # Non-main component becomes a new object ID.
        new_id = int(self.next_obj_id)
        self.next_obj_id += 1
        try:
            ret_new = self.sam2_tracker.add_new_mask(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(new_id),
                mask=new_b,
            )
            if isinstance(ret_new, (tuple, list)) and len(ret_new) >= 2:
                self.all_ids = ret_new[1]
            elif int(new_id) not in self.all_ids:
                self.all_ids.append(int(new_id))
        except Exception:
            return None

        new_box = self._get_mask_bbox(new_b)
        tracked_masks[int(new_id)] = new_b
        tracked_boxes[int(new_id)] = new_box
        object_masks[int(new_id)] = new_b
        object_boxes[int(new_id)] = new_box
        self.mask_history[int(new_id)] = deque(
            [(int(frame_idx), new_b.copy())],
            maxlen=self.mask_history_length,
        )
        new_row = {"obj_id": int(new_id), "bbox": new_box}
        if isinstance(new_obj_meta, dict):
            for k, v in new_obj_meta.items():
                new_row[k] = v
        new_objects.append(new_row)

        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        evt = {
            "type": "new_object_from_multicomponent",
            "frame": int(frame_idx),
            "parent": int(parent_id),
            "child": int(new_id),
            "reason": str(reason),
            "score": float(score_f),
        }
        self.events.append(dict(evt))
        # This is an actual split-like structural state change (parent + new child).
        # Record event frame for per-ID structural cooldown.
        self._struct_event_last_frame[int(parent_id)] = int(
            max(int(self._struct_event_last_frame.get(int(parent_id), -10**9)), int(frame_idx))
        )
        self._struct_event_last_frame[int(new_id)] = int(
            max(int(self._struct_event_last_frame.get(int(new_id), -10**9)), int(frame_idx))
        )
        if bool(emit_struct_split):
            self._emit_struct_split_keep_parent(
                parent_id=int(parent_id),
                child_id=int(new_id),
                reason=str(reason),
                score=float(score_f),
                frame_idx=int(frame_idx),
            )
        return evt

    def _apply_merge_transaction(
        self,
        *,
        a_id: int,
        b_id: int,
        merged_mask: np.ndarray,
        reason: str,
        score: float,
        frame_idx: int,
        tracked_masks: Dict[int, np.ndarray],
        tracked_boxes: Dict[int, List[int]],
        object_masks: Dict[int, np.ndarray],
        object_boxes: Dict[int, List[int]],
        new_objects: List[Dict[str, Any]],
        confirmed_tail: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not bool(self.enable_structural_ops):
            return None
        a_id = int(a_id)
        b_id = int(b_id)
        if int(a_id) == int(b_id):
            return None
        mm = _to_bool_mask(merged_mask)
        if mm is None or np.count_nonzero(mm) <= 0:
            return None
        mm = mm.astype(bool)

        new_id = int(self.next_obj_id)
        self.next_obj_id += 1
        try:
            ret = self.sam2_tracker.add_new_mask(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(new_id),
                mask=mm,
            )
            if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                self.all_ids = ret[1]
            elif int(new_id) not in self.all_ids:
                self.all_ids.append(int(new_id))
        except Exception:
            return None

        mbox = self._get_mask_bbox(mm)
        tracked_masks[int(new_id)] = mm
        tracked_boxes[int(new_id)] = mbox
        object_masks[int(new_id)] = mm
        object_boxes[int(new_id)] = mbox
        self.mask_history[int(new_id)] = deque(
            [(int(frame_idx), mm.copy())],
            maxlen=self.mask_history_length,
        )
        new_objects.append({
            "obj_id": int(new_id),
            "bbox": mbox,
            "source": "merge",
        })

        old_ids = [int(a_id), int(b_id)]
        for oid in old_ids:
            if int(oid) in self.all_ids:
                try:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state,
                        int(oid),
                        strict=False,
                        need_output=False,
                    )
                except Exception:
                    pass
            if int(oid) in self.all_ids:
                self.all_ids = [int(x) for x in self.all_ids if int(x) != int(oid)]
            tracked_masks.pop(int(oid), None)
            tracked_boxes.pop(int(oid), None)
            object_masks.pop(int(oid), None)
            object_boxes.pop(int(oid), None)
            self.mask_history.pop(int(oid), None)
            self.confusion_objects_history.pop(int(oid), None)
            self._retired_struct_ids.add(int(oid))

        score_f = float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0
        parents_sorted = [int(min(a_id, b_id)), int(max(a_id, b_id))]
        struct_event = {
            "type": "merge_apply",
            "frame": int(frame_idx),
            "parents": parents_sorted,
            "children": [int(new_id)],
            "reason": str(reason),
            "score": float(score_f),
        }
        if isinstance(confirmed_tail, dict):
            struct_event["confirmed_tail"] = dict(confirmed_tail)
        self._struct_events_frame.append(dict(struct_event))
        self.events.append(dict(struct_event))
        self._id_transitions_frame.append({
            "type": "merge",
            "frame": int(frame_idx),
            "from": parents_sorted,
            "to": [int(new_id)],
            "reason": str(reason),
            "score": float(score_f),
        })
        self._set_struct_id_recovery_block(parents_sorted + [int(new_id)], int(frame_idx))

        return struct_event

    def _select_struct_merge_mask(
        self,
        a_mask: np.ndarray,
        b_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        a = _to_bool_mask(a_mask)
        b = _to_bool_mask(b_mask)
        if a is None or b is None or a.shape != b.shape:
            return None, {"reason": "invalid_input"}
        union_mask = np.logical_or(a, b).astype(bool)
        if np.count_nonzero(union_mask) <= 0:
            return None, {"reason": "empty_union"}

        best_row: Optional[Dict[str, Any]] = None
        for det_idx, det_mask in enumerate(list(self._current_target_masks or [])):
            dm = _to_bool_mask(det_mask)
            if dm is None or dm.shape != union_mask.shape or np.count_nonzero(dm) <= 0:
                continue
            a_in = float(self._mask_in_mask_ratio(a, dm))
            b_in = float(self._mask_in_mask_ratio(b, dm))
            union_cover = float(self._mask_in_mask_ratio(union_mask, dm))
            components = int(self._connected_components_count(dm))
            mask_pass = bool(
                min(a_in, b_in) >= float(self.attach_mask_in_th)
                and union_cover >= float(self.attach_proxy_signal_pair_union_in_th)
            )
            if not bool(mask_pass):
                continue
            if int(components) > 1:
                continue
            score = float(min(a_in, b_in, union_cover))
            row = {
                "source": "detector_mask",
                "det_idx": int(det_idx),
                "mask": dm,
                "a_in": float(a_in),
                "b_in": float(b_in),
                "union_cover": float(union_cover),
                "components": int(components),
                "score": float(score),
            }
            if best_row is None or float(row["score"]) > float(best_row["score"]):
                best_row = row

        if best_row is None:
            return None, {"reason": "strict_gate_failed"}
        return best_row.get("mask"), {
            "source": str(best_row.get("source")),
            "det_idx": int(best_row.get("det_idx", -1)),
            "a_in": float(best_row.get("a_in", 0.0)),
            "b_in": float(best_row.get("b_in", 0.0)),
            "union_cover": float(best_row.get("union_cover", 0.0)),
            "components": int(best_row.get("components", 0)),
            "score": float(best_row.get("score", 0.0)),
        }

    def apply_external_structural_events(
        self,
        *,
        struct_events: Optional[List[Dict[str, Any]]],
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply externally-decided structural events to live SAM2 state.

        Current scope:
        - split only (pair event style): parent->child(ren)
        - parent ID is retired from SAM2 state immediately
        - if parent still has remaining area after removing child masks, promote it to a fresh ID
        """
        if not bool(self.enable_structural_ops):
            return {"applied_count": 0, "applied": []}
        ev_rows = list(struct_events or [])
        if not ev_rows:
            return {"applied_count": 0, "applied": []}

        fi = int(self.frame_idx - 1 if frame_idx is None else frame_idx)
        split_children_by_parent: Dict[int, Set[int]] = {}
        for ev in ev_rows:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("type") or "") != "split":
                continue
            parent_ids: List[int] = []
            child_ids: List[int] = []
            try:
                a_val = ev.get("a")
                if a_val is not None:
                    parent_ids.append(int(a_val))
            except Exception:
                pass
            for p in list(ev.get("parents", []) or []):
                try:
                    parent_ids.append(int(p))
                except Exception:
                    continue
            try:
                b_val = ev.get("b")
                if b_val is not None:
                    child_ids.append(int(b_val))
            except Exception:
                pass
            for c in list(ev.get("children", []) or []):
                try:
                    child_ids.append(int(c))
                except Exception:
                    continue
            parent_ids = [int(x) for x in parent_ids if int(x) >= 100]
            child_ids = [int(x) for x in child_ids if int(x) >= 100]
            if not parent_ids:
                continue
            parent_id = int(parent_ids[0])
            if not child_ids:
                continue
            split_children_by_parent.setdefault(parent_id, set()).update(
                int(x) for x in child_ids if int(x) != int(parent_id)
            )

        if not split_children_by_parent:
            return {"applied_count": 0, "applied": []}

        applied_rows: List[Dict[str, Any]] = []
        for parent_id in sorted(split_children_by_parent.keys()):
            parent_id_i = int(parent_id)
            if parent_id_i < 100:
                continue
            if parent_id_i in self._retired_struct_ids:
                continue
            if parent_id_i not in self.all_ids:
                continue

            parent_mask = _to_bool_mask(self._prev_tracked_masks.get(parent_id_i))
            if parent_mask is None or np.count_nonzero(parent_mask) <= 0:
                continue

            active_children: List[int] = []
            child_union = np.zeros_like(parent_mask, dtype=bool)
            for child_id in sorted(split_children_by_parent.get(parent_id_i, set())):
                cid = int(child_id)
                if cid < 100 or cid not in self.all_ids:
                    continue
                cm = _to_bool_mask(self._prev_tracked_masks.get(cid))
                if cm is None or cm.shape != parent_mask.shape or np.count_nonzero(cm) <= 0:
                    continue
                active_children.append(int(cid))
                child_union = np.logical_or(child_union, cm.astype(bool))
            if not active_children:
                continue

            remaining_mask = np.logical_and(parent_mask.astype(bool), np.logical_not(child_union))
            parent_area = int(np.count_nonzero(parent_mask))
            remain_area = int(np.count_nonzero(remaining_mask))
            replacement_id: Optional[int] = None

            if remain_area > 0:
                replacement_id = int(self.next_obj_id)
                self.next_obj_id += 1
                try:
                    ret = self.sam2_tracker.add_new_mask(
                        self.inference_state,
                        frame_idx=int(fi),
                        obj_id=int(replacement_id),
                        mask=remaining_mask,
                    )
                    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                        self.all_ids = ret[1]
                    elif int(replacement_id) not in self.all_ids:
                        self.all_ids.append(int(replacement_id))
                except Exception:
                    replacement_id = None

            if parent_id_i in self.all_ids:
                try:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state,
                        int(parent_id_i),
                        strict=False,
                        need_output=False,
                    )
                except Exception:
                    pass
            if parent_id_i in self.all_ids:
                self.all_ids = [int(x) for x in self.all_ids if int(x) != int(parent_id_i)]

            self._prev_tracked_masks.pop(int(parent_id_i), None)
            if replacement_id is not None and remain_area > 0:
                self._prev_tracked_masks[int(replacement_id)] = remaining_mask.astype(bool).copy()

            if len(self._tracked_masks_history) > 0:
                try:
                    last_masks = dict(self._tracked_masks_history[-1] or {})
                except Exception:
                    last_masks = {}
                last_masks.pop(int(parent_id_i), None)
                if replacement_id is not None and remain_area > 0:
                    last_masks[int(replacement_id)] = remaining_mask.astype(bool).copy()
                try:
                    self._tracked_masks_history[-1] = last_masks
                except Exception:
                    pass

            self.mask_history.pop(int(parent_id_i), None)
            self.confusion_objects_history.pop(int(parent_id_i), None)
            self._retired_struct_ids.add(int(parent_id_i))

            if replacement_id is not None:
                self.mask_history[int(replacement_id)] = deque(
                    [(int(fi), remaining_mask.astype(bool).copy())],
                    maxlen=self.mask_history_length,
                )
                self._id_birth_frame[int(replacement_id)] = int(fi)
                self._id_last_seen_frame[int(replacement_id)] = int(fi)

            block_ids = [int(parent_id_i)] + [int(x) for x in active_children]
            if replacement_id is not None:
                block_ids.append(int(replacement_id))
            self._set_struct_id_recovery_block(block_ids, int(fi))

            row = {
                "type": "split_apply_external",
                "frame": int(fi),
                "parent": int(parent_id_i),
                "children": [int(x) for x in sorted(active_children)],
                "replacement_id": None if replacement_id is None else int(replacement_id),
                "parent_area": int(parent_area),
                "replacement_area": int(remain_area),
            }
            applied_rows.append(row)
            print(
                f"[Hotrack] F{fi} EXTERNAL_SPLIT_APPLY: "
                f"parent={int(parent_id_i)} -> children={sorted(active_children)} "
                f"replacement={int(replacement_id) if replacement_id is not None else None}"
            )

        return {
            "applied_count": int(len(applied_rows)),
            "applied": applied_rows,
        }

    def _attach_proxy_signal_from_boxes(
        self,
        target_boxes: List[List[int]],
        tracked_masks: Dict[int, np.ndarray],
        object_ids: List[int],
        target_masks: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Attach proxy trigger signal.

        Preferred mode:
            detector-box seeded masks (target_masks) are reused and pair(A,B) is accepted
            when union(A,B) is sufficiently included in det-mask.
        Fallback mode:
            raw detector boxes with per-object in-box ratio.
        """
        in_th = float(np.clip(self.attach_proxy_signal_in_ratio_th, 0.0, 1.0))
        pair_union_th = float(np.clip(self.attach_proxy_signal_pair_union_in_th, 0.0, 1.0))
        # Pair-temporal logic: candidate is determined by pair continuity.
        min_objs = 2
        signal_boxes: List[List[int]] = []
        signal_pairs: Set[Tuple[int, int]] = set()
        box_rows: List[Dict[str, Any]] = []
        obj_masks: List[Tuple[int, np.ndarray]] = []
        for oid in list(object_ids or []):
            oid_i = int(oid)
            m = _to_bool_mask(tracked_masks.get(int(oid_i)))
            if m is None or np.count_nonzero(m) == 0:
                continue
            obj_masks.append((int(oid_i), m))

        cleaned_masks: List[np.ndarray] = []
        for dm in list(target_masks or []):
            mm = _to_bool_mask(dm)
            if mm is None or np.count_nonzero(mm) == 0:
                continue
            cleaned_masks.append(mm)
        cleaned_boxes_gate: List[List[int]] = []
        for box in list(target_boxes or []):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                cleaned_boxes_gate.append([int(v) for v in box])
            except Exception:
                continue

        require_single_box = True
        if cleaned_masks:
            single_box_mode_ok = bool(len(cleaned_masks) == 1)
            for det_idx, det_mask in enumerate(cleaned_masks):
                det_box = self._get_mask_bbox(det_mask)
                scores: List[Dict[str, Any]] = []
                in_ratio_by_id: Dict[int, float] = {}
                for oid_i, m in obj_masks:
                    in_ratio = float(self._mask_in_mask_ratio(m, det_mask))
                    in_ratio_by_id[int(oid_i)] = float(in_ratio)
                    scores.append({
                        "obj_id": int(oid_i),
                        "in_ratio": float(in_ratio),
                        "pass": bool(in_ratio >= in_th),
                    })

                pair_candidates: List[Dict[str, Any]] = []
                pair_rows_tmp: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
                pass_pair_scores: Dict[Tuple[int, int], float] = {}
                inside_by_pair: Set[int] = set()
                for ii, (a_id, a_mask) in enumerate(obj_masks):
                    for b_id, b_mask in obj_masks[ii + 1:]:
                        union_mask = np.logical_or(a_mask, b_mask)
                        pair_union_in = float(self._mask_in_mask_ratio(union_mask, det_mask))
                        a_in = float(in_ratio_by_id.get(int(a_id), 0.0))
                        b_in = float(in_ratio_by_id.get(int(b_id), 0.0))
                        # Enforce "same detector box contains both masks" even in det-mask mode.
                        # Without this gate, union-only pass can create false pair tails.
                        same_box_pass = False
                        same_box_idx: Optional[int] = None
                        for bidx, box_gate in enumerate(cleaned_boxes_gate):
                            a_in_box = float(self._mask_in_box_ratio(a_mask, box_gate))
                            b_in_box = float(self._mask_in_box_ratio(b_mask, box_gate))
                            if a_in_box >= in_th and b_in_box >= in_th:
                                same_box_pass = True
                                same_box_idx = int(bidx)
                                break
                        pair_pass = bool(
                            a_in >= in_th
                            and b_in >= in_th
                            and pair_union_in >= pair_union_th
                            and same_box_pass
                        )
                        pair_key = (min(int(a_id), int(b_id)), max(int(a_id), int(b_id)))
                        row_pair = {
                            "pair": [int(min(a_id, b_id)), int(max(a_id, b_id))],
                            "a_in_ratio": float(a_in),
                            "b_in_ratio": float(b_in),
                            "union_in_ratio": float(pair_union_in),
                            "same_box_pass": bool(same_box_pass),
                            "same_box_idx": None if same_box_idx is None else int(same_box_idx),
                            "pass": bool(pair_pass),
                            "selected": False,
                        }
                        pair_rows_tmp.append((pair_key, row_pair))
                        if pair_pass:
                            prev = float(pass_pair_scores.get(pair_key, 0.0))
                            pass_pair_scores[pair_key] = float(max(prev, pair_union_in))

                selected_pairs: Set[Tuple[int, int]] = set()
                if pass_pair_scores:
                    selected_pairs = set(pass_pair_scores.keys())
                for pkey, prow in pair_rows_tmp:
                    if pkey in selected_pairs:
                        prow["selected"] = True
                    pair_candidates.append(prow)
                for pkey in selected_pairs:
                    inside_by_pair.add(int(pkey[0]))
                    inside_by_pair.add(int(pkey[1]))
                    signal_pairs.add((int(pkey[0]), int(pkey[1])))

                inside_ids = sorted(int(x) for x in inside_by_pair)
                has_pair = bool(len(inside_ids) >= min_objs)
                trigger = bool((not require_single_box or single_box_mode_ok) and has_pair)
                row = {
                    "mode": "det_mask",
                    "box": [int(v) for v in det_box],
                    "box_idx": int(det_idx),
                    "inside_ids": [int(x) for x in inside_ids],
                    "inside_count": int(len(inside_ids)),
                    "pair_candidates": pair_candidates[:12],
                    "single_box_mode_ok": bool(single_box_mode_ok),
                    "trigger": bool(trigger),
                    "scores": sorted(scores, key=lambda r: float(r.get("in_ratio", 0.0)), reverse=True)[:8],
                }
                box_rows.append(row)
                if trigger:
                    signal_boxes.append([int(v) for v in det_box])

            return {
                "signal": bool(len(signal_boxes) > 0),
                "mode": "det_mask_reuse",
                "in_ratio_th": float(in_th),
                "pair_union_in_th": float(pair_union_th),
                "min_objs": int(min_objs),
                "require_single_box": bool(require_single_box),
                "num_boxes": int(len(cleaned_masks)),
                "single_box_mode_ok": bool(single_box_mode_ok),
                "signal_boxes": [[int(v) for v in b] for b in signal_boxes],
                "signal_pairs": [[int(a), int(b)] for a, b in sorted(signal_pairs)],
                "box_rows": box_rows,
            }

        # Fallback: raw detector boxes.
        cleaned_boxes: List[List[int]] = []
        for box in list(target_boxes or []):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                b = [int(v) for v in box]
            except Exception:
                continue
            cleaned_boxes.append(b)
        single_box_mode_ok = bool(len(cleaned_boxes) == 1)
        for box_idx, b in enumerate(cleaned_boxes):
            inside_ids: List[int] = []
            scores: List[Dict[str, Any]] = []
            in_ratio_by_id: Dict[int, float] = {}
            for oid_i, m in obj_masks:
                in_ratio = float(self._mask_in_box_ratio(m, b))
                scores.append({
                    "obj_id": int(oid_i),
                    "in_ratio": float(in_ratio),
                    "pass": bool(in_ratio >= in_th),
                })
                if in_ratio >= in_th:
                    inside_ids.append(int(oid_i))
                    in_ratio_by_id[int(oid_i)] = float(in_ratio)
            inside_ids = sorted(set(inside_ids))
            pass_pair_scores: Dict[Tuple[int, int], float] = {}
            for i, a in enumerate(inside_ids):
                for bb in inside_ids[i + 1:]:
                    key = (min(int(a), int(bb)), max(int(a), int(bb)))
                    s = float(min(in_ratio_by_id.get(int(a), 0.0), in_ratio_by_id.get(int(bb), 0.0)))
                    pass_pair_scores[key] = float(max(float(pass_pair_scores.get(key, 0.0)), s))
            selected_pairs = set(pass_pair_scores.keys()) if pass_pair_scores else set()
            pair_candidates = [[int(a), int(bb)] for a, bb in sorted(selected_pairs)]
            has_pair = bool(len(pair_candidates) > 0)
            row = {
                "mode": "det_box",
                "box": [int(v) for v in b],
                "box_idx": int(box_idx),
                "inside_ids": [int(x) for x in inside_ids],
                "inside_count": int(len(inside_ids)),
                "pair_candidates": pair_candidates,
                "single_box_mode_ok": bool(single_box_mode_ok),
                "trigger": bool((not require_single_box or single_box_mode_ok) and has_pair),
                "scores": sorted(scores, key=lambda r: float(r.get("in_ratio", 0.0)), reverse=True)[:8],
            }
            box_rows.append(row)
            if (not require_single_box or single_box_mode_ok) and has_pair:
                signal_boxes.append([int(v) for v in b])
                for a, bb in selected_pairs:
                    signal_pairs.add((int(min(a, bb)), int(max(a, bb))))

        return {
            "signal": bool(len(signal_boxes) > 0),
            "mode": "det_box_fallback",
            "in_ratio_th": float(in_th),
            "pair_union_in_th": float(pair_union_th),
            "min_objs": int(min_objs),
            "require_single_box": bool(require_single_box),
            "num_boxes": int(len(cleaned_boxes)),
            "single_box_mode_ok": bool(single_box_mode_ok),
            "signal_boxes": [[int(v) for v in b] for b in signal_boxes],
            "signal_pairs": [[int(a), int(b)] for a, b in sorted(signal_pairs)],
            "box_rows": box_rows,
        }

    @staticmethod
    def _mask_center_point_dt(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        if mask is None or np.count_nonzero(mask) == 0:
            return None
        mask_u8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if dist is None or dist.size == 0:
            return None
        max_val = float(dist.max())
        if max_val <= 0.0:
            return None
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        return int(x), int(y)

    @staticmethod
    def _mask_center_points_dt(
        mask: np.ndarray,
        *,
        max_points: int,
        min_dist: int,
    ) -> List[Tuple[int, int]]:
        if mask is None or np.count_nonzero(mask) == 0:
            return []
        mask_u8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if dist is None or dist.size == 0:
            return []
        pts: List[Tuple[int, int]] = []
        dist_work = dist.copy()
        h, w = dist_work.shape[:2]
        for _ in range(int(max_points)):
            max_val = float(dist_work.max())
            if max_val <= 0.0:
                break
            y, x = np.unravel_index(int(dist_work.argmax()), dist_work.shape)
            pts.append((int(x), int(y)))
            r = max(int(min_dist), int(max_val * 0.5))
            x1 = max(0, int(x) - r)
            y1 = max(0, int(y) - r)
            x2 = min(w - 1, int(x) + r)
            y2 = min(h - 1, int(y) + r)
            dist_work[y1:y2 + 1, x1:x2 + 1] = 0.0
        return pts

    @staticmethod
    def _mask_uniform_points(
        mask: np.ndarray,
        *,
        max_points: int,
        min_dist: int,
    ) -> List[Tuple[int, int]]:
        if mask is None or np.count_nonzero(mask) == 0:
            return []
        max_points = max(0, int(max_points))
        if max_points <= 0:
            return []

        obj_mask = mask.astype(bool)
        dist_obj = cv2.distanceTransform(obj_mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Prefer interior pixels first; fallback to all mask pixels when mask is thin.
        candidates = obj_mask & (dist_obj >= float(max(1, int(min_dist))))
        if not candidates.any():
            candidates = obj_mask
        ys, xs = np.where(candidates)
        if len(xs) == 0:
            return []

        coords = np.column_stack((ys, xs)).astype(np.float32)  # (N,2): y,x

        # Cap candidate count for speed while preserving spatial coverage.
        max_candidates = 5000
        if coords.shape[0] > max_candidates:
            pick = np.linspace(0, coords.shape[0] - 1, num=max_candidates, dtype=np.int32)
            coords = coords[pick]

        centroid = np.mean(coords, axis=0)
        d0 = np.sum((coords - centroid) ** 2, axis=1)
        first_idx = int(np.argmin(d0))
        selected: List[np.ndarray] = [coords[first_idx]]

        min_dist_sq = float(max(1, int(min_dist)) ** 2)
        while len(selected) < max_points:
            sel = np.stack(selected, axis=0)  # (M,2)
            diff = coords[:, None, :] - sel[None, :, :]
            d2 = np.sum(diff * diff, axis=2)  # (N,M)
            min_d2 = np.min(d2, axis=1)      # (N,)
            next_idx = int(np.argmax(min_d2))
            if float(min_d2[next_idx]) < min_dist_sq:
                break
            selected.append(coords[next_idx])

        return [(int(p[1]), int(p[0])) for p in selected]

    def _load_candidate_model(self) -> Optional[Dict[str, Any]]:
        path = str(self.candidate_model_path or "").strip()
        if not path:
            return None
        if not os.path.exists(path):
            print(f"[Hotrack] Candidate model not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            weights_raw = payload.get("weights", {})
            if not isinstance(weights_raw, dict):
                return None
            weights: Dict[str, float] = {}
            for k, v in weights_raw.items():
                try:
                    weights[str(k)] = float(v)
                except Exception:
                    continue
            bias = float(payload.get("bias", 0.0) or 0.0)
            print(f"[Hotrack] Candidate model loaded: {path} ({len(weights)} features)")
            return {
                "bias": float(bias),
                "weights": weights,
            }
        except Exception as exc:
            print(f"[Hotrack] Candidate model load failed: {exc}")
            return None

    def _is_new_accept_cap_reached(self, accepted_in_frame: Dict[int, np.ndarray]) -> bool:
        cap = int(getattr(self, "max_new_per_frame", 0) or 0)
        return bool(cap > 0 and len(accepted_in_frame) >= cap)

    @staticmethod
    def _sigmoid_score(x: float) -> float:
        x = float(np.clip(x, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _score_candidate_features(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        model = self._candidate_model
        if isinstance(model, dict):
            weights = model.get("weights", {}) or {}
            z = float(model.get("bias", 0.0) or 0.0)
            for k, w in weights.items():
                z += float(w) * float(features.get(str(k), 0.0) or 0.0)
            return self._sigmoid_score(z), float(z), "model"

        q = float(np.clip(features.get("quality", 0.0), 0.0, 1.0))
        compactness = float(np.clip(features.get("compactness", 0.0), 0.0, 1.0))
        near_hand = float(np.clip(features.get("near_hand", 0.0), 0.0, 1.0))
        dup_flag = float(np.clip(features.get("duplicate_flag", 0.0), 0.0, 1.0))
        obj_overlap = float(np.clip(features.get("obj_overlap_max", 0.0), 0.0, 1.0))
        hand_overlap = float(np.clip(features.get("hand_overlap_max", 0.0), 0.0, 1.0))
        largest_ratio = float(np.clip(features.get("largest_component_ratio", 1.0), 0.0, 1.0))
        area_ratio = float(max(features.get("mask_area_ratio", 0.0), 0.0))
        bbox_ratio = float(max(features.get("bbox_area_ratio", 0.0), 0.0))
        area_pen = float(max(0.0, area_ratio - float(self.max_obj_area_ratio)) / max(float(self.max_obj_area_ratio), 1e-6))
        bbox_pen = float(max(0.0, bbox_ratio - float(self.max_obj_area_ratio)) / max(float(self.max_obj_area_ratio), 1e-6))
        lost_overlap = float(np.clip(features.get("lost_overlap_max", 0.0), 0.0, 1.0))

        z = (
            + 2.8 * q
            + 1.1 * compactness
            + 0.8 * near_hand
            + 0.4 * lost_overlap
            + 0.5 * largest_ratio
            - 1.8 * obj_overlap
            - 1.6 * hand_overlap
            - 1.4 * dup_flag
            - 1.2 * area_pen
            - 0.8 * bbox_pen
        )
        return self._sigmoid_score(z), float(z), "heuristic"

    def _build_candidate_features(
        self,
        *,
        mask: np.ndarray,
        bbox: List[int],
        det_box: Optional[List[int]],
        image_h: int,
        image_w: int,
        hand_masks: List[np.ndarray],
        object_masks: Dict[int, np.ndarray],
        hand_area_ref: float,
        quality: float,
        near_hand: bool,
        duplicate_flag: bool,
    ) -> Dict[str, float]:
        mask_area = float(np.sum(mask))
        img_area = float(max(1, image_h * image_w))
        x1, y1, x2, y2 = [int(v) for v in (bbox or [0, 0, 0, 0])]
        bbox_area = float(max(0, x2 - x1) * max(0, y2 - y1))
        det_bbox_area = 0.0
        if det_box is not None:
            dx1, dy1, dx2, dy2 = [int(v) for v in det_box]
            det_bbox_area = float(max(0, dx2 - dx1) * max(0, dy2 - dy1))

        mask_u8 = mask.astype(np.uint8)
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        comp_count = max(0, int(num_labels - 1))
        largest_comp_area = 0.0
        if num_labels > 1 and stats is not None:
            largest_comp_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
        largest_comp_ratio = float(largest_comp_area / max(mask_area, 1.0))

        max_h2m = 0.0
        max_m2h = 0.0
        for hm in hand_masks:
            if hm is None or np.count_nonzero(hm) == 0:
                continue
            _, ioa_h2m, ioa_m2h = calculate_ioa_bidirectional(hm, mask)
            max_h2m = max(max_h2m, float(ioa_h2m))
            max_m2h = max(max_m2h, float(ioa_m2h))

        max_e2n = 0.0
        max_n2e = 0.0
        for om in object_masks.values():
            if om is None or np.count_nonzero(om) == 0:
                continue
            _, ioa_e2n, ioa_n2e = calculate_ioa_bidirectional(om, mask)
            max_e2n = max(max_e2n, float(ioa_e2n))
            max_n2e = max(max_n2e, float(ioa_n2e))

        lost_overlap = 0.0
        for info in self.confusion_objects_history.values():
            lm = info.get("mask")
            if lm is None or np.count_nonzero(lm) == 0:
                continue
            if lm.shape != mask.shape:
                lm = cv2.resize(
                    lm.astype(np.uint8),
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            _, ioa_l2n, ioa_n2l = calculate_ioa_bidirectional(lm, mask)
            lost_overlap = max(lost_overlap, float(ioa_l2n), float(ioa_n2l))

        max_iou_prev_hand = 0.0
        for pb in self._prev_hand_boxes:
            max_iou_prev_hand = max(max_iou_prev_hand, float(bbox_iou(bbox, pb)))
        max_iou_prev_obj = 0.0
        for pb in self._prev_object_boxes:
            max_iou_prev_obj = max(max_iou_prev_obj, float(bbox_iou(bbox, pb)))

        return {
            "mask_area": float(mask_area),
            "mask_area_ratio": float(mask_area / img_area),
            "bbox_area_ratio": float(bbox_area / img_area) if img_area > 0 else 0.0,
            "det_bbox_area_ratio": float(det_bbox_area / img_area) if img_area > 0 else 0.0,
            "compactness": float(mask_area / max(bbox_area, 1.0)),
            "component_count": float(comp_count),
            "largest_component_ratio": float(np.clip(largest_comp_ratio, 0.0, 1.0)),
            "max_hand_ioa_h2m": float(max_h2m),
            "max_hand_ioa_m2h": float(max_m2h),
            "hand_overlap_max": float(max(max_h2m, max_m2h)),
            "max_obj_ioa_e2n": float(max_e2n),
            "max_obj_ioa_n2e": float(max_n2e),
            "obj_overlap_max": float(max(max_e2n, max_n2e)),
            "near_hand": 1.0 if near_hand else 0.0,
            "duplicate_flag": 1.0 if duplicate_flag else 0.0,
            "mask_vs_hand_ratio": float(mask_area / max(hand_area_ref, 1.0)) if hand_area_ref > 0.0 else 0.0,
            "bbox_vs_hand_ratio": float(bbox_area / max(hand_area_ref, 1.0)) if hand_area_ref > 0.0 else 0.0,
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "lost_overlap_max": float(np.clip(lost_overlap, 0.0, 1.0)),
            "prev_hand_bbox_iou_max": float(max_iou_prev_hand),
            "prev_object_bbox_iou_max": float(max_iou_prev_obj),
        }

    def _candidate_backfill_metrics(
        self,
        *,
        candidate_mask: np.ndarray,
        hand_masks_current: List[np.ndarray],
    ) -> Dict[str, float]:
        if self.backfill_window <= 1 or len(self._frame_history) < 2:
            return {}
        frames_list = list(self._frame_history)
        max_steps = min(int(self.candidate_backfill_max_steps), len(frames_list) - 1)
        if max_steps <= 0:
            return {}
        seed_id = -999
        tracked = self.backfill_track_backward(
            frames_list,
            {int(seed_id): candidate_mask},
            max_backfill=max_steps,
        )
        seq = tracked.get(int(seed_id))
        if not seq or len(seq) < 2:
            return {}

        ious: List[float] = []
        for prev_m, cur_m in zip(seq[:-1], seq[1:]):
            iou, _, _ = calculate_ioa_bidirectional(prev_m, cur_m)
            ious.append(float(iou))
        mean_iou = float(np.mean(ious)) if ious else 0.0
        areas = [float(np.sum(m)) for m in seq]
        area_mean = float(np.mean(areas)) if areas else 0.0
        area_std = float(np.std(areas)) if areas else 0.0
        area_cv = float(area_std / max(area_mean, 1.0))

        hand_hist = list(self._tracked_masks_history)
        need_prev = max(0, len(seq) - 1)
        hand_hist_tail = hand_hist[-need_prev:] if need_prev > 0 else []
        hand_overlap_vals: List[float] = []
        for idx, m in enumerate(seq):
            if m is None or np.count_nonzero(m) == 0:
                hand_overlap_vals.append(0.0)
                continue
            hand_union = None
            if idx < len(seq) - 1:
                hist_idx = idx - max(0, need_prev - len(hand_hist_tail))
                if 0 <= hist_idx < len(hand_hist_tail):
                    frame_masks = hand_hist_tail[hist_idx] or {}
                    for oid, hm in frame_masks.items():
                        if int(oid) >= 100 or hm is None or np.count_nonzero(hm) == 0:
                            continue
                        hand_union = hm.astype(bool) if hand_union is None else np.logical_or(hand_union, hm.astype(bool))
            else:
                for hm in hand_masks_current:
                    if hm is None or np.count_nonzero(hm) == 0:
                        continue
                    hand_union = hm.astype(bool) if hand_union is None else np.logical_or(hand_union, hm.astype(bool))
            if hand_union is None or not hand_union.any():
                hand_overlap_vals.append(0.0)
                continue
            _, ioa_h2m, ioa_m2h = calculate_ioa_bidirectional(hand_union, m.astype(bool))
            hand_overlap_vals.append(float(max(ioa_h2m, ioa_m2h)))
        hand_overlap_mean = float(np.mean(hand_overlap_vals)) if hand_overlap_vals else 0.0
        stability = (
            0.50 * mean_iou
            + 0.30 * (1.0 - float(np.clip(area_cv, 0.0, 1.0)))
            + 0.20 * (1.0 - float(np.clip(hand_overlap_mean, 0.0, 1.0)))
        )
        return {
            "backfill_mean_iou": float(mean_iou),
            "backfill_area_cv": float(area_cv),
            "backfill_hand_overlap_mean": float(hand_overlap_mean),
            "backfill_stability_score": float(np.clip(stability, 0.0, 1.0)),
        }

    def _append_candidate_logs(self, rows: List[Dict[str, Any]]) -> None:
        if not self.candidate_log_enabled or not rows:
            return
        try:
            os.makedirs(os.path.dirname(self.candidate_log_path), exist_ok=True)
            with open(self.candidate_log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
        except Exception:
            return

    def _save_attach_replay_debug_visuals(
        self,
        *,
        pair: Tuple[int, int],
        start_frame_idx: int,
        seed_box: List[int],
        replay: Dict[str, Any],
        reverse_eval: Optional[Dict[str, Any]],
        forward_replay: Optional[Dict[str, Any]],
        session_status: Optional[str],
        pair_qualified: bool = False,
        frames_by_index: Dict[int, np.ndarray],
        target_boxes_by_frame: Dict[int, List[List[int]]],
        tracked_masks_by_frame: Dict[int, Dict[int, np.ndarray]],
    ) -> None:
        if not bool(self.attach_replay_debug):
            return
        if bool(self.attach_replay_debug_only_passed):
            status_txt = str(session_status or "")
            # Save debug for qualified candidates even when final session status is fail.
            # This keeps candidate-pass traces visible for investigation.
            if not status_txt.startswith("passed") and not bool(pair_qualified):
                return
        try:
            os.makedirs(self.attach_replay_debug_dir, exist_ok=True)
            a_id, b_id = int(min(pair[0], pair[1])), int(max(pair[0], pair[1]))
            sx1, sy1, sx2, sy2 = [int(v) for v in list(seed_box or [0, 0, 0, 0])[:4]]
            session_name = (
                f"{int(start_frame_idx):06d}_{int(a_id)}-{int(b_id)}"
                f"_seed_{int(sx1)}_{int(sy1)}_{int(sx2)}_{int(sy2)}"
            )
            session_dir = os.path.join(self.attach_replay_debug_dir, session_name)
            reverse_dir = os.path.join(session_dir, "reverse")
            forward_dir = os.path.join(session_dir, "forward")
            os.makedirs(reverse_dir, exist_ok=True)
            os.makedirs(forward_dir, exist_ok=True)

            def _draw_mask_outline(canvas: np.ndarray, mask: Optional[np.ndarray], color: Tuple[int, int, int], label: str) -> None:
                mm = _to_bool_mask(mask)
                if mm is None or np.count_nonzero(mm) == 0:
                    return
                mm_u8 = (mm.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(canvas, contours, -1, color, 2)
                ys, xs = np.where(mm)
                if len(xs) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    cv2.putText(canvas, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            reverse_rows = [r for r in list((replay or {}).get("rows", []) or []) if isinstance(r, dict)]
            reverse_row_by_frame = {
                int(r.get("frame_idx")): r
                for r in reverse_rows
                if r.get("frame_idx") is not None
            }
            try:
                seed_frame_idx = (
                    int((replay or {}).get("seed_frame_idx"))
                    if (replay or {}).get("seed_frame_idx") is not None
                    else None
                )
            except Exception:
                seed_frame_idx = None
            reverse_proxy_masks = dict((replay or {}).get("proxy_masks_by_frame") or {})
            reverse_new_masks = dict((replay or {}).get("new_masks_by_frame") or {})
            reverse_all_new_masks = dict((replay or {}).get("all_new_masks_by_frame") or {})
            reverse_frame_indices = [int(x) for x in list((replay or {}).get("frame_indices", []) or [])]
            reverse_y_frames = [int(x) for x in list((replay or {}).get("new_object_frames", []) or [])]
            reverse_y_detected = bool(len(reverse_y_frames) > 0)

            reverse_eval = dict(reverse_eval or {})
            try:
                eval_y_first = (
                    int(reverse_eval.get("y_first_frame"))
                    if reverse_eval.get("y_first_frame") is not None
                    else None
                )
            except Exception:
                eval_y_first = None
            eval_y_within = bool(reverse_eval.get("y_within_5", False))
            eval_xy_pass = bool(reverse_eval.get("xy_ab_match_pass", False))
            eval_xy_reason = str(reverse_eval.get("xy_ab_match_reason") or "")
            eval_xy_score = float(reverse_eval.get("xy_ab_match_score", 0.0) or 0.0)
            eval_xy_assignment = str(reverse_eval.get("xy_ab_match_assignment") or "")

            for fi in reverse_frame_indices:
                frame = frames_by_index.get(int(fi))
                if frame is None:
                    continue
                vis = frame.copy()
                for box in list(target_boxes_by_frame.get(int(fi), []) or []):
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
                if (
                    seed_frame_idx is not None
                    and int(fi) == int(seed_frame_idx)
                    and len(seed_box) == 4
                ):
                    x1, y1, x2, y2 = [int(v) for v in seed_box]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        f"SEED-X@{int(seed_frame_idx)}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                frame_masks = dict(tracked_masks_by_frame.get(int(fi), {}) or {})
                _draw_mask_outline(vis, frame_masks.get(int(a_id)), (0, 255, 0), f"A{int(a_id)}")
                _draw_mask_outline(vis, frame_masks.get(int(b_id)), (255, 255, 0), f"B{int(b_id)}")
                _draw_mask_outline(vis, reverse_proxy_masks.get(int(fi)), (255, 255, 255), "X")

                rrow = reverse_row_by_frame.get(int(fi), {})
                new_map = reverse_new_masks.get(int(fi), {}) or {}
                all_new_map = reverse_all_new_masks.get(int(fi), {}) or {}
                # Draw all newly created Y (pair-matched and unmatched) on the birth frame.
                y_birth_all_ids: Set[int] = set()
                for yid_raw in list(rrow.get("accepted_new_ids", []) or []):
                    try:
                        y_birth_all_ids.add(int(yid_raw))
                    except Exception:
                        continue
                for yid_raw in list(rrow.get("purged_unmatched_ids", []) or []):
                    try:
                        y_birth_all_ids.add(int(yid_raw))
                    except Exception:
                        continue
                y_birth_rel_ids: Set[int] = set()
                for yid_raw in list(rrow.get("accepted_relevant_new_ids", []) or []):
                    try:
                        y_birth_rel_ids.add(int(yid_raw))
                    except Exception:
                        continue
                if isinstance(all_new_map, dict):
                    for nid in sorted(y_birth_all_ids):
                        nm = _to_bool_mask(all_new_map.get(int(nid)))
                        if nm is None or np.count_nonzero(nm) == 0:
                            continue
                        # Orange: new Y exists but not yet A/B pair-matched.
                        _draw_mask_outline(vis, nm, (0, 165, 255), f"Y?{int(nid)}")
                if isinstance(new_map, dict):
                    for nid in sorted(y_birth_rel_ids):
                        nm = _to_bool_mask(new_map.get(int(nid)))
                        if nm is None or np.count_nonzero(nm) == 0:
                            nm = _to_bool_mask(all_new_map.get(int(nid)))
                        if nm is None or np.count_nonzero(nm) == 0:
                            continue
                        # Red: Y that also passed A/B pair relevance.
                        _draw_mask_outline(vis, nm, (0, 0, 255), f"Y{int(nid)}")

                y = 18
                y_text = (
                    f"Y_detected={str(reverse_y_detected)} "
                    f"y_first={str(eval_y_first) if eval_y_first is not None else 'None'} "
                    f"within_win={str(eval_y_within)}"
                )
                y_col = (0, 220, 0) if reverse_y_detected else (0, 0, 220)
                cv2.putText(vis, y_text[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, y_col, 1, cv2.LINE_AA)
                y += 16
                if reverse_y_detected:
                    if eval_xy_pass:
                        match_txt = (
                            f"AB_match=PASS score={eval_xy_score:.3f} "
                            f"assign={eval_xy_assignment}"
                        )
                        match_col = (0, 220, 0)
                    else:
                        reason_txt = eval_xy_reason if eval_xy_reason else "unmatched_or_below_th"
                        match_txt = (
                            f"AB_match=FAIL reason={reason_txt} "
                            f"score={eval_xy_score:.3f}"
                        )
                        match_col = (0, 0, 220)
                    cv2.putText(vis, match_txt[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, match_col, 1, cv2.LINE_AA)
                    y += 16
                if eval_y_first is not None and int(fi) == int(eval_y_first):
                    cv2.putText(
                        vis,
                        "Y-FIRST",
                        (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 16
                for crow in list(rrow.get("candidates", []) or [])[:8]:
                    box = crow.get("box")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        accepted = bool(crow.get("accepted", False))
                        col = (0, 220, 0) if accepted else (0, 0, 220)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
                    txt = f"{int(crow.get('cand_id', -1))} {str(crow.get('reason', ''))}"
                    cv2.putText(vis, txt[:64], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                    y += 15
                hdr = f"REV fi={int(fi)} pair={int(a_id)}-{int(b_id)} start={int(start_frame_idx)}"
                cv2.putText(vis, hdr, (8, max(0, vis.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(reverse_dir, f"{int(fi):06d}.png"), vis)

            if isinstance(forward_replay, dict) and bool(forward_replay.get("ok", False)):
                forward_rows = [r for r in list(forward_replay.get("rows", []) or []) if isinstance(r, dict)]
                forward_row_by_frame = {
                    int(r.get("frame_idx")): r
                    for r in forward_rows
                    if r.get("frame_idx") is not None
                }
                forward_mode = str(forward_replay.get("mode") or "")
                forward_proxy_masks = dict(forward_replay.get("proxy_masks_by_frame") or {})
                forward_x_masks = dict(forward_replay.get("x_masks_by_frame") or {})
                forward_frame_indices = [int(x) for x in list(forward_replay.get("frame_indices", []) or [])]
                for fi in forward_frame_indices:
                    frame = frames_by_index.get(int(fi))
                    if frame is None:
                        continue
                    vis = frame.copy()
                    frame_masks = dict(tracked_masks_by_frame.get(int(fi), {}) or {})
                    if forward_mode == "y_in_x":
                        _draw_mask_outline(vis, forward_x_masks.get(int(fi)), (255, 255, 255), "X")
                        _draw_mask_outline(vis, forward_proxy_masks.get(int(fi)), (0, 0, 255), "Yfwd")
                    else:
                        _draw_mask_outline(vis, frame_masks.get(int(a_id)), (0, 255, 0), f"A{int(a_id)}")
                        _draw_mask_outline(vis, frame_masks.get(int(b_id)), (255, 255, 0), f"B{int(b_id)}")
                        _draw_mask_outline(vis, forward_proxy_masks.get(int(fi)), (255, 255, 255), "Xfwd")
                    frow = forward_row_by_frame.get(int(fi), {})
                    if forward_mode == "y_in_x":
                        txt = (
                            f"FWD(YinX) fi={int(fi)} pass={bool(frow.get('pass', False))} "
                            f"YinX={float(frow.get('y_in_x', 0.0)):.3f}"
                        )
                    else:
                        txt = (
                            f"FWD fi={int(fi)} pass={bool(frow.get('pass', False))} "
                            f"AinX={float(frow.get('a_in_x', 0.0)):.3f} BinX={float(frow.get('b_in_x', 0.0)):.3f}"
                        )
                    cv2.putText(vis, txt[:96], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(forward_dir, f"{int(fi):06d}.png"), vis)

            meta = {
                "pair": [int(a_id), int(b_id)],
                "pair_qualified": bool(pair_qualified),
                "start_frame_idx": int(start_frame_idx),
                "seed_frame_idx": int(seed_frame_idx) if seed_frame_idx is not None else None,
                "seed_box": [int(v) for v in list(seed_box or [0, 0, 0, 0])[:4]],
                "proxy_id": int((replay or {}).get("proxy_id", -1) or -1),
                "session_status": str(session_status or ""),
                "reverse_frames": [int(x) for x in reverse_frame_indices],
                "forward_frames": [
                    int(x) for x in list((forward_replay or {}).get("frame_indices", []) or [])
                ] if isinstance(forward_replay, dict) else [],
                "reverse_y_frames": [int(x) for x in list((replay or {}).get("new_object_frames", []) or [])],
                "merge_confirmed": False,
                "merge_confirmed_frame_idx": None,
                "merge_confirmed_frame_tag": None,
                "reverse_eval": {
                    "y_detected": bool(reverse_y_detected),
                    "y_first_frame": None if eval_y_first is None else int(eval_y_first),
                    "y_within_5": bool(eval_y_within),
                    "xy_ab_match_pass": bool(eval_xy_pass),
                    "xy_ab_match_reason": str(eval_xy_reason),
                    "xy_ab_match_score": float(eval_xy_score),
                    "xy_ab_match_assignment": str(eval_xy_assignment),
                },
            }
            with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _save_split_replay_debug_visuals(
        self,
        *,
        parent_id: int,
        child_id: Optional[int],
        frame_indices: List[int],
        frames_by_index: Dict[int, np.ndarray],
        parent_masks_by_frame: Dict[int, np.ndarray],
        keep_fwd_masks_by_frame: Dict[int, np.ndarray],
        split_fwd_masks_by_frame: Dict[int, np.ndarray],
        keep_back_masks_by_frame: Dict[int, np.ndarray],
        split_back_masks_by_frame: Dict[int, np.ndarray],
        result: Dict[str, Any],
    ) -> Optional[str]:
        if not bool(self.split_replay_debug):
            return None
        try:
            os.makedirs(self.split_replay_debug_dir, exist_ok=True)
            self._split_replay_debug_counter = int(getattr(self, "_split_replay_debug_counter", 0)) + 1
            debug_idx = int(self._split_replay_debug_counter)
            parent_i = int(parent_id)
            child_i: Optional[int]
            try:
                child_i = int(child_id) if child_id is not None else None
            except Exception:
                child_i = None
            pass_flag = bool((result or {}).get("pass", False))
            reason_raw = str((result or {}).get("reason", "unknown") or "unknown")
            reason_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", reason_raw).strip("_")
            reason_slug = (reason_slug[:48] if reason_slug else "unknown")
            child_tag = f"c{int(child_i)}" if child_i is not None else "cNA"
            session_name = (
                f"{int(self.frame_idx):06d}_s{int(debug_idx):04d}_"
                f"p{int(parent_i)}_{child_tag}_{'pass' if pass_flag else 'fail'}_{reason_slug}"
            )
            session_dir = os.path.join(self.split_replay_debug_dir, session_name)
            os.makedirs(session_dir, exist_ok=True)

            def _draw_mask_outline(
                canvas: np.ndarray,
                mask: Optional[np.ndarray],
                color: Tuple[int, int, int],
                label: str,
            ) -> None:
                mm = _to_bool_mask(mask)
                if mm is None or np.count_nonzero(mm) <= 0:
                    return
                mm_u8 = (mm.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(canvas, contours, -1, color, 2)
                ys, xs = np.where(mm)
                if len(xs) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    cv2.putText(canvas, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            rows = [dict(r) for r in list((result or {}).get("rows", []) or []) if isinstance(r, dict)]
            row_by_frame = {
                int(r.get("frame_idx")): r
                for r in rows
                if r.get("frame_idx") is not None
            }

            saved_frames: List[str] = []
            for fi in [int(x) for x in list(frame_indices or [])]:
                frame = frames_by_index.get(int(fi))
                if frame is None:
                    continue
                vis = frame.copy()
                _draw_mask_outline(vis, parent_masks_by_frame.get(int(fi)), (255, 255, 255), f"P{int(parent_i)}")
                _draw_mask_outline(vis, keep_fwd_masks_by_frame.get(int(fi)), (0, 220, 0), "Kf")
                _draw_mask_outline(vis, split_fwd_masks_by_frame.get(int(fi)), (0, 0, 255), "Sf")
                _draw_mask_outline(vis, keep_back_masks_by_frame.get(int(fi)), (0, 255, 255), "Kb")
                _draw_mask_outline(vis, split_back_masks_by_frame.get(int(fi)), (255, 0, 255), "Sb")

                row = row_by_frame.get(int(fi), {})
                row_pass = bool(row.get("pass", False))
                row_color = (0, 220, 0) if row_pass else (0, 0, 220)
                y = 18
                head_txt = (
                    f"SPLIT replay fi={int(fi)} p={int(parent_i)} "
                    f"c={int(child_i) if child_i is not None else 'NA'}"
                )
                cv2.putText(vis, head_txt[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                y += 16
                pass_txt = (
                    f"frame_pass={str(row_pass)} "
                    f"best={str(row.get('best_container_id'))} "
                    f"best_in={float(row.get('best_container_in', 0.0) or 0.0):.3f}"
                )
                cv2.putText(vis, pass_txt[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, row_color, 1, cv2.LINE_AA)
                y += 16
                split_in_parent = float(
                    row.get("candidate_parent_in", row.get("split_in_parent", 0.0)) or 0.0
                )
                split_in_th = float(row.get("same_parent_in_th", 0.0) or 0.0)
                cv2.putText(
                    vis,
                    f"split_in_parent={split_in_parent:.3f} th={split_in_th:.3f}"[:110],
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 16
                included = list(row.get("included_ids", []) or [])
                if included:
                    uv_txt = f"included_ids={','.join([str(int(x)) for x in included[:6]])}"
                else:
                    union_in_parent = float(row.get("union_in_parent", 0.0) or 0.0)
                    union_in_parent_th = float(row.get("union_in_parent_th", 0.0) or 0.0)
                    parent_in_union = float(row.get("parent_in_union", 0.0) or 0.0)
                    parent_in_union_th = float(row.get("parent_in_union_th", 0.0) or 0.0)
                    uv_txt = (
                        f"union_in_parent={union_in_parent:.3f}/{union_in_parent_th:.3f} "
                        f"parent_in_union={parent_in_union:.3f}/{parent_in_union_th:.3f}"
                    )
                cv2.putText(vis, uv_txt[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                y += 16

                legend_txt = "white=P, green=Kf, red=Sf, cyan=Kb, magenta=Sb"
                cv2.putText(vis, legend_txt[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                y += 16

                status_line = (
                    f"session={'pass' if pass_flag else 'fail'} reason={reason_raw}"
                )
                cv2.putText(
                    vis,
                    status_line[:110],
                    (8, max(0, vis.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                fname = f"{int(fi):06d}.png"
                cv2.imwrite(os.path.join(session_dir, fname), vis)
                saved_frames.append(str(fname))

            meta = {
                "session_name": str(session_name),
                "frame_idx": int(self.frame_idx),
                "parent_id": int(parent_i),
                "child_id": int(child_i) if child_i is not None else None,
                "pass": bool(pass_flag),
                "reason": str(reason_raw),
                "frame_indices": [int(x) for x in list(frame_indices or [])],
                "required_frames": int((result or {}).get("required_frames", 0) or 0),
                "same_parent_required_frames": int((result or {}).get("same_parent_required_frames", 0) or 0),
                "pass_count": int((result or {}).get("pass_count", 0) or 0),
                "max_tail_pass": int((result or {}).get("max_tail_pass", 0) or 0),
                "same_parent_pass_count": int((result or {}).get("same_parent_pass_count", 0) or 0),
                "same_parent_max_tail": int((result or {}).get("same_parent_max_tail", 0) or 0),
                "rows": rows,
                "saved_frames": saved_frames,
            }
            with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            return session_dir
        except Exception:
            return None
    
    def _check_lost_object_recovery(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        frame_idx: Optional[int] = None,
    ) -> Optional[int]:
        """Check if mask matches an object in confusion memory (ID recovery)."""
        fi = int(self.frame_idx if frame_idx is None else frame_idx)
        if not self.enable_id_recovery:
            return None
        if int(fi) <= int(getattr(self, "_struct_recovery_block_until_frame", -1)):
            return None
        if mask is None or np.count_nonzero(mask) == 0:
            return None

        # DINO-only recovery gate: replace ID when similarity >= threshold.
        if not self.use_dino_id:
            return None
        sim_threshold = float(self.dino_recovery_sim_threshold)
        new_dino_embed: Optional[np.ndarray] = self._dino_encode(image_bgr, mask)
        if new_dino_embed is None:
            return None

        best_id: Optional[int] = None
        best_key: Optional[Tuple[float, float]] = None
        best_info: Optional[Dict[str, float]] = None

        for lost_id, info in self.confusion_objects_history.items():
            if int(lost_id) in self._retired_struct_ids:
                continue
            lost_frame = int(info.get("frame", 0))
            age = float(fi - lost_frame)
            reason = str(info.get("reason", "") or "")
            is_overlap_confusion = bool(
                reason.startswith("active_overlap") or reason.startswith("post_duplicate")
            )
            if is_overlap_confusion and age > float(max(1, int(self.confusion_overlap_max_age_frames))):
                continue

            sim = -1.0
            lost_embed = info.get("embed")
            if lost_embed is None:
                # Fallback: use historical embedding even if confusion snapshot
                # did not store an embed at registration time.
                lost_embed = self._dino_embedding_history.get(int(lost_id))
            if new_dino_embed is not None and lost_embed is not None:
                sim = float(self._cosine_sim(new_dino_embed, lost_embed))
            dino_pass = bool(sim >= sim_threshold)
            if not dino_pass:
                continue

            # Prefer higher DINO similarity, then more recent confusion snapshot.
            key = (float(sim), float(-age))
            if best_key is None or key > best_key:
                best_key = key
                best_id = int(lost_id)
                best_info = {
                    "sim": float(sim),
                }

        # Fallback: if confusion memory is empty/missing for an ID, allow recovery
        # from recent absent-ID DINO history.
        if best_id is None:
            present_ids: Set[int] = {
                int(x) for x in (self.all_ids or []) if int(x) >= 100
            }
            max_age = int(max(1, int(self.confusion_overlap_max_age_frames)))
            for hist_id, hist_embed in self._dino_embedding_history.items():
                oid = int(hist_id)
                if oid < 100:
                    continue
                if oid in self._retired_struct_ids:
                    continue
                if oid in present_ids:
                    continue
                if hist_embed is None:
                    continue
                last_seen = self._id_last_seen_frame.get(oid)
                if last_seen is None:
                    continue
                age = int(fi - int(last_seen))
                if age > max_age:
                    continue
                sim = float(self._cosine_sim(new_dino_embed, hist_embed))
                if sim < sim_threshold:
                    continue
                key = (float(sim), float(-age))
                if best_key is None or key > best_key:
                    best_key = key
                    best_id = int(oid)
                    best_info = {
                        "sim": float(sim),
                    }

        if best_id is not None and best_info is not None:
            print(
                f"[Hotrack] F{fi} CONFUSION_MATCH: "
                f"new mask -> obj-{best_id-100} "
                f"(DINO={best_info['sim']:.2f}, th_dino={sim_threshold:.2f})"
            )
        return best_id

    def _log_dino_similarity(self, image_bgr: np.ndarray, new_id: int, new_mask: np.ndarray,
                              tracked_masks: Dict[int, np.ndarray], topk: int = 5) -> None:
        if not self.compute_dino_similarity:
            return
        try:
            new_embed = self._dino_encode(image_bgr, new_mask)
        except Exception:
            return
        if new_embed is None:
            return
        sims = []
        for oid, mask in tracked_masks.items():
            if int(oid) == int(new_id):
                continue
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            try:
                emb = self._dino_encode(image_bgr, mask)
            except Exception:
                continue
            if emb is None:
                continue
            sims.append((int(oid), float(self._cosine_sim(new_embed, emb))))
        sims.sort(key=lambda x: x[1], reverse=True)
        if sims:
            msg = ", ".join([f"obj-{oid-100}:{sim:.2f}" for oid, sim in sims[:topk]])
            print(f"[Hotrack] F{self.frame_idx} DINO_SIM: new obj-{new_id-100} vs {msg}")
        return None

    def _dino_similarity_for_new_objects(
        self,
        image_bgr: np.ndarray,
        new_object_ids: List[int],
        tracked_masks: Dict[int, np.ndarray],
    ) -> List[Dict[str, Any]]:
        if not self.compute_dino_similarity or not new_object_ids:
            return []
        embed_cache: Dict[int, Optional[np.ndarray]] = {}

        def _get_emb(oid: int, mask: np.ndarray) -> Optional[np.ndarray]:
            key = int(oid)
            if key in embed_cache:
                return embed_cache[key]
            try:
                emb = self._dino_encode(image_bgr, mask)
            except Exception:
                emb = None
            embed_cache[key] = emb
            return emb

        # Ensure history contains embeddings for tracked objects (lazy, one-time per ID).
        for oid, mask in tracked_masks.items():
            oid_int = int(oid)
            if oid_int < 100:
                continue
            if oid_int in self._dino_embedding_history:
                continue
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            emb = _get_emb(oid_int, mask)
            if emb is not None:
                self._update_dino_embedding_history(oid_int, emb)

        out: List[Dict[str, Any]] = []
        for new_id in new_object_ids:
            mask_new = tracked_masks.get(int(new_id))
            if mask_new is None or np.count_nonzero(mask_new) == 0:
                continue
            new_emb = _get_emb(int(new_id), mask_new)
            if new_emb is None:
                continue
            self._update_dino_embedding_history(int(new_id), new_emb)
            sims: List[Dict[str, Any]] = []
            for oid_int, emb in self._dino_embedding_history.items():
                if oid_int == int(new_id):
                    continue
                sims.append({
                    "obj_id": int(oid_int),
                    "sim": float(self._cosine_sim(new_emb, emb)),
                    "present": bool(oid_int in tracked_masks),
                })
            sims.sort(key=lambda x: x["sim"], reverse=True)
            out.append({
                "new_id": int(new_id),
                "sims": sims,
            })
        return out

    def _prune_recent_post_duplicates(self) -> None:
        if not self._recent_post_duplicates:
            return
        stale_ids: List[int] = []
        for rid, info in self._recent_post_duplicates.items():
            frame_removed = int(info.get("frame", -1))
            if self.frame_idx - frame_removed > int(self.dup_readd_cooldown_frames):
                stale_ids.append(int(rid))
        for rid in stale_ids:
            self._recent_post_duplicates.pop(int(rid), None)

    def _register_recent_post_duplicate(
        self,
        removed_id: int,
        kept_id: Optional[int],
        removed_mask: Optional[np.ndarray],
        *,
        frame_idx: Optional[int] = None,
    ) -> None:
        if removed_mask is None or np.count_nonzero(removed_mask) == 0:
            return
        fi = int(self.frame_idx if frame_idx is None else frame_idx)
        self._recent_post_duplicates[int(removed_id)] = {
            "frame": int(fi),
            "kept_id": None if kept_id is None else int(kept_id),
            "mask": removed_mask.copy(),
        }

    def _duplicate_cooldown_hit(
        self,
        candidate_mask: Optional[np.ndarray],
        *,
        frame_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if candidate_mask is None or np.count_nonzero(candidate_mask) == 0:
            return None
        if not self._recent_post_duplicates:
            return None
        fi = int(self.frame_idx if frame_idx is None else frame_idx)
        best: Optional[Dict[str, Any]] = None
        for removed_id, info in list(self._recent_post_duplicates.items()):
            removed_mask = info.get("mask")
            if removed_mask is None or np.count_nonzero(removed_mask) == 0:
                continue
            if removed_mask.shape != candidate_mask.shape:
                removed_mask = cv2.resize(
                    removed_mask.astype(np.uint8),
                    (candidate_mask.shape[1], candidate_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            _, ioa_r_to_c, ioa_c_to_r = calculate_ioa_bidirectional(removed_mask, candidate_mask)
            max_ioa = float(max(ioa_r_to_c, ioa_c_to_r))
            if max_ioa < float(self.dup_readd_cooldown_ioa):
                continue
            hit = {
                "removed_id": int(removed_id),
                "kept_id": info.get("kept_id"),
                "age": int(fi - int(info.get("frame", fi))),
                "ioa_removed_to_candidate": float(ioa_r_to_c),
                "ioa_candidate_to_removed": float(ioa_c_to_r),
                "max_ioa": float(max_ioa),
            }
            if best is None or float(hit["max_ioa"]) > float(best["max_ioa"]):
                best = hit
        return best

    def _decide_post_duplicate_keep_remove(
        self,
        *,
        image_bgr: np.ndarray,
        obj_id_a: int,
        obj_id_b: int,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        dino_cache: Optional[Dict[int, Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """Decide which ID to keep/remove for a duplicate-overlap pair.

        Decision order:
        1) DINO self-similarity (current mask embed vs ID history embed)
        2) Temporal mask stability
        3) Area tie-break
        """
        a_id = int(obj_id_a)
        b_id = int(obj_id_b)
        dino_sim_a: Optional[float] = None
        dino_sim_b: Optional[float] = None
        keep_id: Optional[int] = None
        remove_id: Optional[int] = None
        decision_basis = "temporal_stability"

        if self.use_dino_id:
            emb_hist_a = self._dino_embedding_history.get(a_id)
            emb_hist_b = self._dino_embedding_history.get(b_id)
            emb_cur_a: Optional[np.ndarray] = None
            emb_cur_b: Optional[np.ndarray] = None

            if dino_cache is not None:
                if a_id not in dino_cache:
                    dino_cache[a_id] = self._dino_encode(image_bgr, mask_a)
                if b_id not in dino_cache:
                    dino_cache[b_id] = self._dino_encode(image_bgr, mask_b)
                emb_cur_a = dino_cache.get(a_id)
                emb_cur_b = dino_cache.get(b_id)
            else:
                emb_cur_a = self._dino_encode(image_bgr, mask_a)
                emb_cur_b = self._dino_encode(image_bgr, mask_b)

            if emb_cur_a is not None and emb_hist_a is not None:
                dino_sim_a = float(self._cosine_sim(emb_cur_a, emb_hist_a))
            if emb_cur_b is not None and emb_hist_b is not None:
                dino_sim_b = float(self._cosine_sim(emb_cur_b, emb_hist_b))

            if dino_sim_a is not None and dino_sim_b is not None:
                if dino_sim_a > dino_sim_b:
                    keep_id, remove_id = a_id, b_id
                    decision_basis = "dino_self_similarity"
                elif dino_sim_b > dino_sim_a:
                    keep_id, remove_id = b_id, a_id
                    decision_basis = "dino_self_similarity"
            elif dino_sim_a is not None:
                keep_id, remove_id = a_id, b_id
                decision_basis = "dino_self_similarity_single"
            elif dino_sim_b is not None:
                keep_id, remove_id = b_id, a_id
                decision_basis = "dino_self_similarity_single"

        # Fallback: temporal stability + area.
        stab_a = float(self._temporal_mask_stability_score(a_id, mask_a))
        stab_b = float(self._temporal_mask_stability_score(b_id, mask_b))
        area_a = float(np.sum(mask_a))
        area_b = float(np.sum(mask_b))
        if keep_id is None or remove_id is None:
            if stab_a > stab_b:
                keep_id, remove_id = a_id, b_id
            elif stab_b > stab_a:
                keep_id, remove_id = b_id, a_id
            else:
                if area_a >= area_b:
                    keep_id, remove_id = a_id, b_id
                else:
                    keep_id, remove_id = b_id, a_id

        return {
            "keep_id": int(keep_id),
            "remove_id": int(remove_id),
            "decision_basis": str(decision_basis),
            "dino_self_sim_a": None if dino_sim_a is None else float(dino_sim_a),
            "dino_self_sim_b": None if dino_sim_b is None else float(dino_sim_b),
            "stability_a": float(stab_a),
            "stability_b": float(stab_b),
            "area_a": float(area_a),
            "area_b": float(area_b),
        }

    def _temporal_mask_stability_score(self, obj_id: int, mask_now: Optional[np.ndarray]) -> float:
        """How consistently current mask matches this ID's recent history."""
        if mask_now is None or np.count_nonzero(mask_now) == 0:
            return 0.0
        oid = int(obj_id)
        scores: List[float] = []

        hist = list(self.mask_history.get(oid, []) or [])
        for fidx, hm in hist:
            if hm is None or np.count_nonzero(hm) == 0:
                continue
            if int(fidx) == int(self.frame_idx):
                continue
            hmask = hm
            if hmask.shape != mask_now.shape:
                hmask = cv2.resize(
                    hmask.astype(np.uint8),
                    (mask_now.shape[1], mask_now.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            scores.append(float(calculate_iou(hmask, mask_now)))

        if not scores:
            prev = self._prev_tracked_masks.get(oid)
            if prev is not None and np.count_nonzero(prev) > 0:
                pm = prev
                if pm.shape != mask_now.shape:
                    pm = cv2.resize(
                        pm.astype(np.uint8),
                        (mask_now.shape[1], mask_now.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                scores.append(float(calculate_iou(pm, mask_now)))

        if not scores:
            return 0.0
        return float(np.mean(np.asarray(scores, dtype=np.float32)))

    def _register_confusion_object(
        self,
        obj_id: int,
        mask: Optional[np.ndarray],
        *,
        reason: str,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """Register/update an object in unified confusion memory."""
        if mask is None or np.count_nonzero(mask) == 0:
            return False
        oid = int(obj_id)
        if int(oid) in self._retired_struct_ids:
            return False
        mem_mask = mask.astype(bool).copy()
        embed = None
        if self.use_dino_id and frame_bgr is not None:
            embed = self._dino_encode(frame_bgr, mem_mask)
        # Keep confusion memory usable even when frame-level encode fails
        # (e.g., tiny/invalid crop at registration time).
        if embed is None and self.use_dino_id:
            embed = self._dino_embedding_history.get(oid)
        self.confusion_objects_history[oid] = {
            "frame": int(self.frame_idx),
            "embed": embed,
            "mask": mem_mask,
            "reason": str(reason),
        }
        if embed is not None:
            self._update_dino_embedding_history(oid, embed)
        return True

    def _update_confusion_objects_history(self, prev_object_ids: List[int], current_object_ids: List[int],
                                          prev_masks: Dict[int, np.ndarray],
                                          prev_frame_bgr: Optional[np.ndarray],
                                          lost_reason_by_id: Optional[Dict[int, str]] = None):
        """Update unified confusion memory for disappeared objects."""
        # Find objects that disappeared
        lost_ids = set(prev_object_ids) - set(current_object_ids)
        reason_map = dict(lost_reason_by_id or {})
        
        for lost_id in lost_ids:
            if int(lost_id) in self._retired_struct_ids:
                continue
            reason = str(reason_map.get(int(lost_id), "disappeared"))
            # post_duplicate removal is not a physical confusion state; skip registration.
            if reason == "post_duplicate":
                continue
            if lost_id in prev_masks:
                lost_mask = prev_masks[lost_id]
                self._register_confusion_object(
                    int(lost_id),
                    lost_mask,
                    reason=reason,
                    frame_bgr=prev_frame_bgr,
                )
                area = np.sum(lost_mask)
                print(
                    f"[Hotrack] F{self.frame_idx} EVENT_CONFUSION_ADD: obj-{lost_id-100} "
                    f"reason={reason} (last_area={area} px)"
                )
                self.events.append({
                    "type": "confusion_add",
                    "frame": self.frame_idx,
                    "obj_id": lost_id,
                    "area": int(area),
                    "reason": reason,
                })
        
        # No age-based cleanup: keep lost IDs until they are explicitly recovered.

    def _logits_to_bool_mask(self, logit: torch.Tensor, height: int, width: int) -> np.ndarray:
        mask = (logit > 0.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).squeeze()
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    def _estimate_bg_motion(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_fg: Optional[np.ndarray],
        cur_fg: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], bool, float, int]:
        orb = cv2.ORB_create(nfeatures=1000)
        prev_mask = None
        cur_mask = None
        if prev_fg is not None:
            prev_mask = (~prev_fg).astype(np.uint8) * 255
        if cur_fg is not None:
            cur_mask = (~cur_fg).astype(np.uint8) * 255

        k1, d1 = orb.detectAndCompute(prev_gray, prev_mask)
        k2, d2 = orb.detectAndCompute(gray, cur_mask)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None, False, 0.0, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = list(bf.match(d1, d2))
        if len(matches) < 8:
            return None, False, 0.0, len(matches)
        matches.sort(key=lambda m: m.distance)
        matches = matches[: max(12, len(matches) // 2)]
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        H_aff, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H_aff is not None and inliers is not None:
            inlier_ratio = float(np.mean(inliers))
            return H_aff, True, inlier_ratio, len(matches)

        return None, False, 0.0, len(matches)

    @torch.inference_mode()
    def backfill_track_backward(
        self,
        frames_bgr: List[np.ndarray],
        seed_masks_t: Dict[int, np.ndarray],
        *,
        max_backfill: int = 9,
    ) -> Dict[int, List[np.ndarray]]:
        """
        현재(t)에서 seed 마스크들을 기준으로,
        과거 max_backfill 프레임까지 "별도 state"로 역방향(backfill) 트래킹을 수행.

        frames_bgr: [t-K+1, ..., t] (시간 순)
        return:     frames_bgr와 동일 길이/순서의 mask list (id -> list)
        """
        if frames_bgr is None or len(frames_bgr) == 0:
            return {}

        H, W = frames_bgr[-1].shape[:2]
        seeds: Dict[int, np.ndarray] = {}
        for obj_id, mask in seed_masks_t.items():
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            m = mask.astype(bool)
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            seeds[int(obj_id)] = m
        if not seeds:
            return {}

        K = min(len(frames_bgr), int(max_backfill) + 1)
        frames = frames_bgr[-K:]               # [t-K+1 ... t]
        frames_rev = list(reversed(frames))    # [t ... t-K+1]

        state = self.sam2_tracker.init_state(frames_rev[0])
        for obj_id, m in seeds.items():
            self.sam2_tracker.add_new_mask(
                state,
                frame_idx=0,
                obj_id=int(obj_id),
                mask=m,
            )

        rev_masks: Dict[int, List[np.ndarray]] = {int(obj_id): [m] for obj_id, m in seeds.items()}
        for local_idx in range(1, K):
            state = self.sam2_tracker.add_frame(state, frames_rev[local_idx])
            _, ids, logits, state = self.sam2_tracker.get_mask(state, local_idx)

            if ids is None or logits is None or len(ids) != len(logits):
                ids = []
                logits = []
            id_to_logits = {int(oid): logits[j] for j, oid in enumerate(ids)}
            for obj_id, hist in rev_masks.items():
                if obj_id in id_to_logits:
                    chosen = self._logits_to_bool_mask(id_to_logits[obj_id], H, W)
                else:
                    chosen = hist[-1]
                hist.append(chosen)

        masks_out: Dict[int, List[np.ndarray]] = {}
        for obj_id, hist in rev_masks.items():
            seq = list(reversed(hist))
            if len(seq) < len(frames_bgr):
                pad = [seq[0]] * (len(frames_bgr) - len(seq))
                seq = pad + seq
            masks_out[int(obj_id)] = seq

        return masks_out

    @torch.inference_mode()
    def backfill_track_backward_from_boxes(
        self,
        frames_bgr: List[np.ndarray],
        seed_boxes_t: List[List[int]],
        *,
        max_backfill: int = 9,
    ) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
        """
        현재(t) detector box seed를 기준으로 별도 state에서 역방향(backfill) 트래킹.

        Returns:
            masks_out: proxy_obj_id -> masks list (frames_bgr와 동일 길이/순서)
            seed_box_by_id: proxy_obj_id -> [x1,y1,x2,y2]
        """
        if frames_bgr is None or len(frames_bgr) == 0:
            return {}, {}
        if not seed_boxes_t:
            return {}, {}

        H, W = frames_bgr[-1].shape[:2]
        boxes: List[List[int]] = []
        for box in list(seed_boxes_t):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        if not boxes:
            return {}, {}

        K = min(len(frames_bgr), int(max_backfill) + 1)
        frames = frames_bgr[-K:]
        frames_rev = list(reversed(frames))

        state = self.sam2_tracker.init_state(frames_rev[0])
        rev_masks: Dict[int, List[np.ndarray]] = {}
        seed_box_by_id: Dict[int, List[int]] = {}

        # Use large temporary IDs to avoid collision with normal tracked IDs.
        base_id = int(900000)
        for idx, box in enumerate(boxes):
            proxy_obj_id = int(base_id + idx)
            box_np = np.array(box, dtype=np.float32)
            try:
                _, _, logits, state = self.sam2_tracker.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=proxy_obj_id,
                    box=box_np,
                )
            except Exception:
                continue
            if logits is None or len(logits) == 0:
                continue
            try:
                seed_mask = self._logits_to_bool_mask(logits[-1], H, W)
            except Exception:
                continue
            if seed_mask is None or np.count_nonzero(seed_mask) == 0:
                continue
            rev_masks[int(proxy_obj_id)] = [seed_mask]
            seed_box_by_id[int(proxy_obj_id)] = [int(v) for v in box]

        if not rev_masks:
            return {}, {}

        for local_idx in range(1, K):
            state = self.sam2_tracker.add_frame(state, frames_rev[local_idx])
            _, ids, logits, state = self.sam2_tracker.get_mask(state, local_idx)

            if ids is None or logits is None or len(ids) != len(logits):
                ids = []
                logits = []
            id_to_logits = {int(oid): logits[j] for j, oid in enumerate(ids)}
            for proxy_obj_id, hist in rev_masks.items():
                if int(proxy_obj_id) in id_to_logits:
                    chosen = self._logits_to_bool_mask(id_to_logits[int(proxy_obj_id)], H, W)
                else:
                    chosen = hist[-1]
                hist.append(chosen)

        masks_out: Dict[int, List[np.ndarray]] = {}
        for proxy_obj_id, hist in rev_masks.items():
            seq = list(reversed(hist))
            if len(seq) < len(frames_bgr):
                pad = [seq[0]] * (len(frames_bgr) - len(seq))
                seq = pad + seq
            masks_out[int(proxy_obj_id)] = seq

        return masks_out, seed_box_by_id

    @torch.inference_mode()
    def backfill_replay_backward_from_seed_box(
        self,
        frames_bgr: List[np.ndarray],
        frame_indices: List[int],
        target_boxes_by_frame: List[List[List[int]]],
        seed_box_t: List[int],
        *,
        seed_mask_t: Optional[np.ndarray] = None,
        proxy_obj_id: int,
        a_id: Optional[int] = None,
        b_id: Optional[int] = None,
        tracked_masks_by_frame: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
        max_backfill: int = 9,
        max_steps: Optional[int] = None,
        stop_on_first_new_object: bool = False,
        stop_on_first_matched_new_object: bool = False,
        post_new_object_extra_steps: int = 0,
    ) -> Dict[str, Any]:
        """
        Attach 검증용 역방향 재실행:
        - 별도 state에서 start-frame의 X(seed_mask 우선, 없으면 seed_box)로 시작
        - 프레임 번호 감소 방향으로 추적
        - 각 프레임에서 메인 object 후처리와 동일한 경로(후보 -> det-mask add -> pending split)를 실행
          하여 새 객체(Y) 발생 여부를 기록
        """
        if frames_bgr is None or len(frames_bgr) == 0:
            return {"ok": False, "reason": "no_frames"}
        if frame_indices is None or len(frame_indices) != len(frames_bgr):
            return {"ok": False, "reason": "frame_indices_mismatch"}
        if target_boxes_by_frame is None or len(target_boxes_by_frame) != len(frames_bgr):
            return {"ok": False, "reason": "target_boxes_mismatch"}
        H, W = frames_bgr[-1].shape[:2]
        seed_mask_input = _to_bool_mask(seed_mask_t)
        if seed_mask_input is not None:
            if seed_mask_input.shape[:2] != (H, W):
                seed_mask_input = cv2.resize(
                    seed_mask_input.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if np.count_nonzero(seed_mask_input) == 0:
                seed_mask_input = None

        seed_box_valid = False
        seed_box = [0, 0, 0, 0]
        if isinstance(seed_box_t, (list, tuple)) and len(seed_box_t) == 4:
            try:
                sx1, sy1, sx2, sy2 = [int(v) for v in seed_box_t]
                sx1 = max(0, min(W - 1, sx1))
                sx2 = max(0, min(W - 1, sx2))
                sy1 = max(0, min(H - 1, sy1))
                sy2 = max(0, min(H - 1, sy2))
                if sx2 > sx1 and sy2 > sy1:
                    seed_box = [int(sx1), int(sy1), int(sx2), int(sy2)]
                    seed_box_valid = True
            except Exception:
                seed_box_valid = False

        if seed_mask_input is None and not bool(seed_box_valid):
            return {"ok": False, "reason": "invalid_seed_input"}

        K = min(len(frames_bgr), int(max_backfill) + 1)
        frames = frames_bgr[-K:]
        indices = [int(x) for x in frame_indices[-K:]]
        boxes_seq: List[List[List[int]]] = []
        for row in target_boxes_by_frame[-K:]:
            rr: List[List[int]] = []
            for box in list(row or []):
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                except Exception:
                    continue
                bx1 = max(0, min(W - 1, bx1))
                bx2 = max(0, min(W - 1, bx2))
                by1 = max(0, min(H - 1, by1))
                by2 = max(0, min(H - 1, by2))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                # Keep per-frame detector boxes as-is (clamped only).
                # Box expansion is applied only to seed-X before replay starts.
                rr.append([int(bx1), int(by1), int(bx2), int(by2)])
            boxes_seq.append(rr)

        frames_rev = list(reversed(frames))
        indices_rev = list(reversed(indices))
        boxes_rev = list(reversed(boxes_seq))

        state = self.sam2_tracker.init_state(frames_rev[0])
        seed_mask: Optional[np.ndarray] = None
        seed_source = "union_mask" if seed_mask_input is not None else "seed_box"
        if seed_mask_input is not None:
            try:
                self.sam2_tracker.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=int(proxy_obj_id),
                    mask=seed_mask_input,
                )
                seed_mask = seed_mask_input.copy()
            except Exception:
                seed_mask = None
                seed_source = "seed_box_fallback"

        if seed_mask is None:
            if not bool(seed_box_valid):
                return {"ok": False, "reason": "seed_add_failed"}
            try:
                _, _ids, logits_seed, state = self.sam2_tracker.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=int(proxy_obj_id),
                    box=np.asarray(seed_box, dtype=np.float32),
                )
            except Exception:
                return {"ok": False, "reason": "seed_add_failed"}
            if logits_seed is None or len(logits_seed) == 0:
                return {"ok": False, "reason": "seed_logits_empty"}
            try:
                seed_mask = self._logits_to_bool_mask(logits_seed[-1], H, W)
            except Exception:
                return {"ok": False, "reason": "seed_mask_decode_failed"}
            if seed_mask is None or np.count_nonzero(seed_mask) == 0:
                return {"ok": False, "reason": "seed_mask_empty"}

            # Refine proxy seed-X with in-mask points (same strategy as main object candidates).
            if bool(self.enable_prompt_refinement):
                try:
                    seed_centers = self._mask_uniform_points(
                        seed_mask,
                        max_points=self.dt_point_count,
                        min_dist=self.dt_point_min_dist,
                    )
                    if not seed_centers:
                        seed_centers = self._mask_center_points_dt(
                            seed_mask,
                            max_points=self.dt_point_count,
                            min_dist=self.dt_point_min_dist,
                        )
                    if seed_centers:
                        pts = np.array(seed_centers, dtype=np.float32)
                        labels = np.ones(len(seed_centers), dtype=np.int32)
                        _, _ids_seed2, logits_seed2, state = self.sam2_tracker.add_new_points_or_box(
                            state,
                            frame_idx=0,
                            obj_id=int(proxy_obj_id),
                            points=pts,
                            labels=labels,
                            clear_old_points=True,
                        )
                        if logits_seed2 is not None and len(logits_seed2) > 0:
                            seed_mask2 = self._logits_to_bool_mask(logits_seed2[-1], H, W)
                            if seed_mask2 is not None and np.count_nonzero(seed_mask2) > 0:
                                seed_mask = seed_mask2
                except Exception:
                    pass
            if seed_mask is None or np.count_nonzero(seed_mask) == 0:
                return {"ok": False, "reason": "seed_mask_refine_empty"}
            if seed_source != "seed_box_fallback":
                seed_source = "seed_box"

        if seed_mask is None or np.count_nonzero(seed_mask) == 0:
            return {"ok": False, "reason": "seed_mask_empty"}

        zero_mask = np.zeros((H, W), dtype=bool)
        hist_rev: Dict[int, List[np.ndarray]] = {int(proxy_obj_id): [seed_mask]}
        # Relevant Y set used by attach reverse-stop logic (A/B included in Y).
        new_object_ids: List[int] = []
        new_object_frames: List[int] = []
        # All spawned objects are kept for debug.
        all_new_object_ids: List[int] = []
        all_new_object_frames: List[int] = []
        frame_rows: List[Dict[str, Any]] = []
        # Debug-only birth masks for Y (including unmatched/purged ones).
        # This keeps reverse visualization faithful even when unmatched Y is purged
        # from replay state after pair-check.
        debug_birth_new_masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        candidate_base = int(max(910000, int(proxy_obj_id) + 1000))
        temp_counter = 0
        new_counter = 0
        duplicate_ioa_threshold = 0.5
        pending_splits_rev: Dict[int, Dict[str, Any]] = {}
        prev_tracked_masks_rev: Dict[int, np.ndarray] = {int(proxy_obj_id): seed_mask.copy()}

        def _alloc_replay_new_id() -> int:
            nonlocal new_counter
            while True:
                new_counter += 1
                nid = int(candidate_base + 100000 + new_counter)
                if int(nid) not in hist_rev:
                    return int(nid)

        max_local_idx = int(K - 1)
        if max_steps is not None:
            max_local_idx = min(max_local_idx, int(max(1, int(max_steps))))
        processed_last = 0
        first_new_local_idx: Optional[int] = None
        post_new_object_extra_steps = int(max(0, int(post_new_object_extra_steps)))

        for local_idx in range(1, max_local_idx + 1):
            fi = int(indices_rev[local_idx])
            state = self.sam2_tracker.add_frame(state, frames_rev[local_idx])
            _, ids_now, logits_now, state = self.sam2_tracker.get_mask(state, local_idx)

            id_to_mask: Dict[int, np.ndarray] = {}
            if ids_now is not None and logits_now is not None and len(ids_now) == len(logits_now):
                for j, oid in enumerate(ids_now):
                    try:
                        oid_i = int(oid)
                    except Exception:
                        continue
                    try:
                        mm = self._logits_to_bool_mask(logits_now[j], H, W)
                    except Exception:
                        continue
                    if mm is None:
                        continue
                    id_to_mask[int(oid_i)] = mm

            # Update masks for already tracked IDs in this replay state.
            # No fallback to previous mask: if proxy/object mask is not returned
            # on this frame, mark as missing (zero mask).
            for oid in list(hist_rev.keys()):
                cur_m = id_to_mask.get(int(oid))
                if cur_m is None:
                    cur_m = zero_mask.copy()
                hist_rev[int(oid)].append(cur_m)

            current_masks = {int(oid): hist_rev[int(oid)][-1] for oid in list(hist_rev.keys())}
            current_boxes = {
                int(oid): self._get_mask_bbox(mm)
                for oid, mm in current_masks.items()
                if mm is not None and np.count_nonzero(mm) > 0
            }
            accepted_new_ids: List[int] = []
            candidate_rows: List[Dict[str, Any]] = []
            frame_candidates: List[Dict[str, Any]] = []
            accepted_in_frame: Dict[int, np.ndarray] = {}
            replay_target_masks: List[np.ndarray] = []
            replay_split_target_masks: List[np.ndarray] = []
            frame_tracked_masks = dict((tracked_masks_by_frame or {}).get(int(fi), {}) or {})
            hand_masks_frame: List[np.ndarray] = [
                _to_bool_mask(m)
                for oid, m in frame_tracked_masks.items()
                if int(oid) < 100 and _to_bool_mask(m) is not None and np.count_nonzero(_to_bool_mask(m)) > 0
            ]
            hand_area_ref = 0.0
            if hand_masks_frame:
                hand_areas = [float(np.sum(m)) for m in hand_masks_frame if m is not None]
                if hand_areas:
                    hand_area_ref = float(max(hand_areas))

            # Re-run detector-box seeding on this reverse frame (separate state).
            for box in list(boxes_rev[local_idx] or []):
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                temp_counter += 1
                cand_id = int(candidate_base + temp_counter)
                try:
                    _, _ids_tmp, cand_logits, state = self.sam2_tracker.add_new_points_or_box(
                        state,
                        frame_idx=local_idx,
                        obj_id=int(cand_id),
                        box=np.asarray([int(v) for v in box], dtype=np.float32),
                    )
                except Exception:
                    continue
                if cand_logits is None or len(cand_logits) == 0:
                    continue
                try:
                    cand_mask = self._logits_to_bool_mask(cand_logits[-1], H, W)
                except Exception:
                    continue
                if cand_mask is None or np.count_nonzero(cand_mask) == 0:
                    continue
                # Refine mask using the same center-point strategy as main object tracking.
                if bool(self.enable_prompt_refinement):
                    try:
                        centers = self._mask_uniform_points(
                            cand_mask,
                            max_points=self.dt_point_count,
                            min_dist=self.dt_point_min_dist,
                        )
                        if not centers:
                            centers = self._mask_center_points_dt(
                                cand_mask,
                                max_points=self.dt_point_count,
                                min_dist=self.dt_point_min_dist,
                            )
                        if centers:
                            pts = np.array(centers, dtype=np.float32)
                            labels = np.ones(len(centers), dtype=np.int32)
                            _, _ids_tmp2, cand_logits_pts, state = self.sam2_tracker.add_new_points_or_box(
                                state,
                                frame_idx=local_idx,
                                obj_id=int(cand_id),
                                points=pts,
                                labels=labels,
                                clear_old_points=True,
                            )
                            if cand_logits_pts is not None and len(cand_logits_pts) > 0:
                                cand_mask_pts = self._logits_to_bool_mask(cand_logits_pts[-1], H, W)
                                if cand_mask_pts is not None and np.count_nonzero(cand_mask_pts) > 0:
                                    cand_mask = cand_mask_pts
                    except Exception:
                        pass
                # Keep all components at candidate stage; split/refine handles pruning later.
                target_mask_for_split = cand_mask.copy()
                split_mask_appended = False
                cand_bbox = self._get_mask_bbox(cand_mask)
                cand_area = int(np.count_nonzero(cand_mask))

                # Same major gates as main object-candidate path.
                decision_reason = "candidate"
                max_new_to_existing = 0.0
                max_existing_to_new = 0.0
                proxy_new_to_x = 0.0
                det_box_force_new = False
                if cand_area < int(self.min_mask_area):
                    decision_reason = "small_mask"
                else:
                    box_area = float(max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1])))
                    img_area = float(max(1, H * W))
                    if hand_area_ref > 0.0 and (box_area / max(hand_area_ref, 1.0)) > float(self.max_obj_vs_hand_ratio):
                        decision_reason = "bbox_vs_hand_ratio"
                    elif (box_area / img_area) > float(self.max_obj_area_ratio):
                        decision_reason = "bbox_ratio"
                    else:
                        mask_area_f = float(cand_area)
                        if hand_area_ref > 0.0 and (mask_area_f / max(hand_area_ref, 1.0)) > float(self.max_obj_vs_hand_ratio):
                            decision_reason = "mask_vs_hand_ratio"
                        elif (mask_area_f / img_area) > float(self.max_obj_area_ratio):
                            decision_reason = "mask_ratio"
                        else:
                            # Same det-box isolated force-new policy as forward path.
                            if (
                                bool(getattr(self, "force_new_when_det_box_isolated", True))
                                and isinstance(box, (list, tuple))
                                and len(box) == 4
                                and current_masks
                            ):
                                try:
                                    bx1, by1, bx2, by2 = [int(v) for v in list(box)[:4]]
                                    bx1 = max(0, min(W - 1, bx1))
                                    bx2 = max(0, min(W - 1, bx2))
                                    by1 = max(0, min(H - 1, by1))
                                    by2 = max(0, min(H - 1, by2))
                                    if bx2 > bx1 and by2 > by1:
                                        det_box_i = [int(bx1), int(by1), int(bx2), int(by2)]
                                        det_box_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
                                        cand_in_box = float(self._mask_in_box_ratio(cand_mask, det_box_i))
                                        max_exist_in_box = 0.0
                                        max_exist_box_iou = 0.0
                                        for exist_id, exist_mask in current_masks.items():
                                            if exist_mask is None or np.count_nonzero(exist_mask) <= 0:
                                                continue
                                            max_exist_in_box = max(
                                                float(max_exist_in_box),
                                                float(self._mask_in_box_ratio(exist_mask, det_box_i)),
                                            )
                                            exist_box = current_boxes.get(int(exist_id))
                                            if isinstance(exist_box, (list, tuple)) and len(exist_box) == 4:
                                                try:
                                                    exb = [int(v) for v in list(exist_box)[:4]]
                                                    max_exist_box_iou = max(
                                                        float(max_exist_box_iou),
                                                        float(bbox_iou(det_box_i, exb)),
                                                    )
                                                except Exception:
                                                    pass
                                        iso_iou_th = float(
                                            np.clip(float(getattr(self, "force_new_det_box_isolation_iou_th", 0.05)), 0.0, 1.0)
                                        )
                                        iso_mask_th = float(
                                            np.clip(float(getattr(self, "force_new_det_box_isolation_mask_in_th", 0.05)), 0.0, 1.0)
                                        )
                                        min_cand_in_box = float(
                                            np.clip(float(getattr(self, "force_new_det_box_min_candidate_in_box_ratio", 0.01)), 0.0, 1.0)
                                        )
                                        if (
                                            float(max_exist_in_box) <= float(iso_mask_th)
                                            and float(max_exist_box_iou) <= float(iso_iou_th)
                                            and float(cand_in_box) >= float(min_cand_in_box)
                                        ):
                                            clipped_mask = np.zeros_like(cand_mask, dtype=bool)
                                            clipped_mask[by1:by2, bx1:bx2] = cand_mask[by1:by2, bx1:bx2]
                                            clipped_area = int(np.count_nonzero(clipped_mask))
                                            min_clipped_area = int(max(16, round(det_box_area * 0.02)))
                                            if clipped_area >= int(min_clipped_area):
                                                det_box_force_new = True
                                                cand_mask = clipped_mask.astype(bool)
                                                target_mask_for_split = cand_mask.copy()
                                                cand_bbox = self._get_mask_bbox(cand_mask)
                                                cand_area = int(np.count_nonzero(cand_mask))
                                except Exception:
                                    pass

                            max_ioa = 0.0
                            max_ioa_n2e = 0.0
                            for oid, om in current_masks.items():
                                if om is None or np.count_nonzero(om) == 0:
                                    continue
                                _, ioa_e2n, ioa_n2e = calculate_ioa_bidirectional(om, cand_mask)
                                max_ioa = max(max_ioa, float(ioa_e2n), float(ioa_n2e))
                                max_ioa_n2e = max(max_ioa_n2e, float(ioa_n2e))
                                max_existing_to_new = max(max_existing_to_new, float(ioa_e2n))
                                max_new_to_existing = max(max_new_to_existing, float(ioa_n2e))
                                if int(oid) == int(proxy_obj_id):
                                    proxy_new_to_x = max(proxy_new_to_x, float(ioa_n2e))
                            if max_ioa >= float(self.new_mask_reject_ioa) or max_ioa_n2e >= float(self.new_mask_inclusion_reject_ioa):
                                # Same det-box rescue as forward path:
                                # if clipped mask is isolated from existing objects,
                                # keep it as a new candidate.
                                if isinstance(box, (list, tuple)) and len(box) == 4:
                                    try:
                                        bx1, by1, bx2, by2 = [int(v) for v in list(box)[:4]]
                                        bx1 = max(0, min(W - 1, bx1))
                                        bx2 = max(0, min(W - 1, bx2))
                                        by1 = max(0, min(H - 1, by1))
                                        by2 = max(0, min(H - 1, by2))
                                        if bx2 > bx1 and by2 > by1:
                                            clipped_mask = np.zeros_like(cand_mask, dtype=bool)
                                            clipped_mask[by1:by2, bx1:bx2] = cand_mask[by1:by2, bx1:bx2]
                                            clipped_area = int(np.count_nonzero(clipped_mask))
                                            if clipped_area > 0:
                                                max_clip_ioa = 0.0
                                                max_clip_n2e = 0.0
                                                for om in current_masks.values():
                                                    if om is None or np.count_nonzero(om) == 0:
                                                        continue
                                                    _, ioa_e2n_clip, ioa_n2e_clip = calculate_ioa_bidirectional(om, clipped_mask)
                                                    max_clip_ioa = max(max_clip_ioa, float(ioa_e2n_clip), float(ioa_n2e_clip))
                                                    max_clip_n2e = max(max_clip_n2e, float(ioa_n2e_clip))
                                                if (
                                                    max_clip_ioa < float(self.new_mask_reject_ioa)
                                                    and max_clip_n2e < float(self.new_mask_inclusion_reject_ioa)
                                                ):
                                                    det_box_force_new = True
                                                    cand_mask = clipped_mask.astype(bool)
                                                    target_mask_for_split = cand_mask.copy()
                                                    cand_bbox = self._get_mask_bbox(cand_mask)
                                                    cand_area = int(np.count_nonzero(cand_mask))
                                    except Exception:
                                        pass

                                if not bool(det_box_force_new):
                                    decision_reason = "overlap_gate"
                                    if max_ioa_n2e >= float(self.new_mask_inclusion_reject_ioa):
                                        decision_reason = "inclusion_overlap_gate"
                                        best_parent = None
                                        best_comp_mask = None
                                        best_comp_score = 0.0
                                        for exist_id, exist_mask in current_masks.items():
                                            if int(exist_id) < 100:
                                                continue
                                            if exist_mask is None or np.count_nonzero(exist_mask) == 0:
                                                continue
                                            mask_uint8 = exist_mask.astype(np.uint8)
                                            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                                            if num_labels <= 2:
                                                continue
                                            for label in range(1, num_labels):
                                                comp_mask = (labels == label)
                                                if np.count_nonzero(comp_mask) == 0:
                                                    continue
                                                _, ioa_c_to_n, ioa_n_to_c = calculate_ioa_bidirectional(comp_mask, target_mask_for_split)
                                                if ioa_c_to_n < duplicate_ioa_threshold or ioa_n_to_c < duplicate_ioa_threshold:
                                                    continue
                                                score = float(min(ioa_c_to_n, ioa_n_to_c))
                                                if score > best_comp_score:
                                                    best_comp_score = score
                                                    best_parent = int(exist_id)
                                                    best_comp_mask = comp_mask
                                        if best_parent is not None and best_comp_mask is not None:
                                            best_other_overlap = 0.0
                                            for other_id, other_mask in current_masks.items():
                                                if int(other_id) < 100 or int(other_id) == int(best_parent):
                                                    continue
                                                if other_mask is None or np.count_nonzero(other_mask) == 0:
                                                    continue
                                                _, ioa_o_to_c, ioa_c_to_o = calculate_ioa_bidirectional(other_mask, best_comp_mask)
                                                best_other_overlap = max(best_other_overlap, float(ioa_o_to_c), float(ioa_c_to_o))
                                            if best_other_overlap < float(self.new_mask_inclusion_reject_ioa):
                                                replay_target_masks.append(target_mask_for_split.copy())
                                                replay_split_target_masks.append(target_mask_for_split.copy())
                                                split_mask_appended = True
                                    else:
                                        replay_target_masks.append(target_mask_for_split.copy())
                                        split_mask_appended = True

                if decision_reason != "candidate":
                    try:
                        self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                    except Exception:
                        pass
                    candidate_rows.append({
                        "cand_id": int(cand_id),
                        "box": [int(v) for v in box],
                        "accepted": False,
                        "reason": str(decision_reason),
                        "area": int(cand_area),
                        "score": 0.0,
                        "proxy_new_to_x": float(proxy_new_to_x),
                        "max_new_to_existing": float(max_new_to_existing),
                        "max_existing_to_new": float(max_existing_to_new),
                    })
                    continue
                if not split_mask_appended:
                    replay_target_masks.append(target_mask_for_split.copy())
                    split_mask_appended = True

                # Build candidate score exactly like main object path.
                near_hand = self._is_near_hand_dilated(cand_mask, hand_masks_frame) if hand_masks_frame else False
                is_dup, dup_reason = self._check_duplicate(
                    cand_mask,
                    cand_bbox,
                    current_masks,
                    current_boxes,
                    ioa_threshold=float(duplicate_ioa_threshold),
                )
                quality = self._compute_quality_score(cand_mask, cand_bbox, hand_masks_frame)
                features = self._build_candidate_features(
                    mask=cand_mask,
                    bbox=cand_bbox,
                    det_box=[int(v) for v in box],
                    image_h=H,
                    image_w=W,
                    hand_masks=hand_masks_frame,
                    object_masks=current_masks,
                    hand_area_ref=hand_area_ref,
                    quality=quality,
                    near_hand=near_hand,
                    duplicate_flag=is_dup,
                )
                score, score_raw, score_source = self._score_candidate_features(features)
                frame_candidates.append({
                    "cand_id": int(cand_id),
                    "box": [int(v) for v in box],
                    "mask": cand_mask,
                    "bbox": cand_bbox,
                    "score": float(score),
                    "score_raw": float(score_raw),
                    "score_source": str(score_source),
                    "quality": float(quality),
                    "near_hand": bool(near_hand),
                    "is_dup": bool(is_dup),
                    "dup_reason": str(dup_reason),
                    "det_box_force_new": bool(det_box_force_new),
                    "proxy_new_to_x": float(proxy_new_to_x),
                    "max_new_to_existing": float(max_new_to_existing),
                    "max_existing_to_new": float(max_existing_to_new),
                })

            def _ensure_hist_mask_at_current_step(oid: int, mask_in: np.ndarray) -> None:
                oid_i = int(oid)
                mm = _to_bool_mask(mask_in)
                if mm is None or np.count_nonzero(mm) == 0:
                    return
                if int(oid_i) not in hist_rev:
                    pre_pad_local = [zero_mask.copy() for _ in range(local_idx)]
                    pre_pad_local.append(mm.copy())
                    hist_rev[int(oid_i)] = pre_pad_local
                    return
                seq = hist_rev[int(oid_i)]
                need_len = int(local_idx) + 1
                if len(seq) < need_len:
                    seq.extend([zero_mask.copy() for _ in range(need_len - len(seq))])
                seq[-1] = mm.copy()

            def _find_temporal_reassign_target_replay(
                new_mask_local: np.ndarray,
                current_object_masks_local: Dict[int, np.ndarray],
            ) -> Optional[Dict[str, Any]]:
                if not self.enable_temporal_id_reassign:
                    return None
                nm = _to_bool_mask(new_mask_local)
                if nm is None or np.count_nonzero(nm) == 0:
                    return None
                new_area = float(np.sum(nm))
                min_prev_area = int(max(self.temporal_reassign_min_prev_area, self.min_mask_area))
                best_match_local: Optional[Dict[str, Any]] = None
                for obj_id_local, cur_mask_local in current_object_masks_local.items():
                    oid_local = int(obj_id_local)
                    if int(oid_local) < 100:
                        continue
                    prev_mask_local = _to_bool_mask(prev_tracked_masks_rev.get(int(oid_local)))
                    cur_mask_bool = _to_bool_mask(cur_mask_local)
                    if (
                        prev_mask_local is None
                        or cur_mask_bool is None
                        or np.count_nonzero(prev_mask_local) <= 0
                        or np.count_nonzero(cur_mask_bool) <= 0
                    ):
                        continue
                    if prev_mask_local.shape != nm.shape:
                        prev_mask_local = cv2.resize(
                            prev_mask_local.astype(np.uint8),
                            (nm.shape[1], nm.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    if cur_mask_bool.shape != nm.shape:
                        cur_mask_bool = cv2.resize(
                            cur_mask_bool.astype(np.uint8),
                            (nm.shape[1], nm.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    prev_area_local = float(np.sum(prev_mask_local))
                    cur_area_local = float(np.sum(cur_mask_bool))
                    if prev_area_local < float(min_prev_area):
                        continue
                    if cur_area_local <= 0.0:
                        continue
                    if cur_area_local > prev_area_local * float(self.temporal_reassign_shrink_ratio):
                        continue
                    if new_area < cur_area_local * float(self.temporal_reassign_new_vs_cur_ratio):
                        continue
                    _, ioa_prev_to_new_local, ioa_new_to_prev_local = calculate_ioa_bidirectional(
                        prev_mask_local,
                        nm,
                    )
                    if ioa_prev_to_new_local < float(self.temporal_reassign_prev_to_new_ioa):
                        continue
                    if ioa_new_to_prev_local < float(self.temporal_reassign_new_to_prev_ioa):
                        continue
                    score_local = float(min(ioa_prev_to_new_local, ioa_new_to_prev_local))
                    if (
                        best_match_local is None
                        or score_local > float(best_match_local.get("score", -1.0))
                    ):
                        best_match_local = {
                            "obj_id": int(oid_local),
                            "score": float(score_local),
                            "prev_area": float(prev_area_local),
                            "cur_area": float(cur_area_local),
                            "new_area": float(new_area),
                            "ioa_prev_to_new": float(ioa_prev_to_new_local),
                            "ioa_new_to_prev": float(ioa_new_to_prev_local),
                        }
                return best_match_local

            frame_candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            resolved_candidate_ids: Set[int] = set()
            image_now_rev = frames_rev[local_idx]
            prev_hand_boxes_rev: List[List[int]] = []
            if local_idx > 0:
                prev_fi = int(indices_rev[local_idx - 1])
                prev_frame_masks = dict((tracked_masks_by_frame or {}).get(int(prev_fi), {}) or {})
                for phid, phm in prev_frame_masks.items():
                    try:
                        if int(phid) >= 100:
                            continue
                    except Exception:
                        continue
                    phm_b = _to_bool_mask(phm)
                    if phm_b is None or np.count_nonzero(phm_b) <= 0:
                        continue
                    prev_hand_boxes_rev.append(self._get_mask_bbox(phm_b))

            for cand in frame_candidates:
                cand_id = int(cand.get("cand_id", -1))
                cand_mask = _to_bool_mask(cand.get("mask"))
                if cand_mask is None or np.count_nonzero(cand_mask) == 0:
                    resolved_candidate_ids.add(int(cand_id))
                    try:
                        self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                    except Exception:
                        pass
                    continue

                cand_box = [int(v) for v in list(cand.get("box", []) or [])[:4]]
                cand_bbox = [int(v) for v in list(cand.get("bbox", []) or [])[:4]]
                score_val = float(cand.get("score", 0.0))
                reason = "accepted"
                row_extra: Dict[str, Any] = {}
                special_handled = False

                if score_val < float(self.candidate_score_threshold):
                    reason = "low_score"
                else:
                    for acc_id, acc_mask in accepted_in_frame.items():
                        _, ioa_acc_to_new, ioa_new_to_acc = calculate_ioa_bidirectional(acc_mask, cand_mask)
                        if (
                            float(ioa_acc_to_new) >= float(duplicate_ioa_threshold)
                            and float(ioa_new_to_acc) >= float(duplicate_ioa_threshold)
                        ):
                            reason = f"dup_in_frame_with_{int(acc_id)}"
                            break

                if reason == "accepted":
                    ambiguous = False
                    if len(cand_bbox) == 4:
                        for prev_box in prev_hand_boxes_rev:
                            if len(prev_box) != 4:
                                continue
                            if bbox_iou(cand_bbox, prev_box) >= float(self.ambiguous_bbox_iou_threshold):
                                ambiguous = True
                                break
                    if not ambiguous and hand_masks_frame:
                        for hand_mask in hand_masks_frame:
                            _, ioa_obj_to_hand, ioa_hand_to_obj = calculate_ioa_bidirectional(cand_mask, hand_mask)
                            if max(float(ioa_obj_to_hand), float(ioa_hand_to_obj)) >= float(self.ambiguous_mask_ioa_threshold):
                                ambiguous = True
                                break
                    if ambiguous:
                        reason = "ambiguous_region"

                if reason == "accepted":
                    cooldown_hit = self._duplicate_cooldown_hit(cand_mask, frame_idx=int(fi))
                    if cooldown_hit is not None:
                        reason = "recent_post_duplicate"
                        row_extra.update({
                            "cooldown_removed_id": int(cooldown_hit.get("removed_id", -1)),
                            "cooldown_kept_id": (
                                int(cooldown_hit.get("kept_id"))
                                if cooldown_hit.get("kept_id") is not None
                                else None
                            ),
                            "cooldown_age": int(cooldown_hit.get("age", -1)),
                            "cooldown_max_ioa": float(cooldown_hit.get("max_ioa", 0.0)),
                        })

                recovered_id: Optional[int] = None
                if reason == "accepted":
                    recovered_id = self._check_lost_object_recovery(
                        image_now_rev,
                        cand_mask,
                        frame_idx=int(fi),
                    )
                    if recovered_id is not None and int(recovered_id) in self._retired_struct_ids:
                        reason = "id_recovery_retired_struct_id"
                        recovered_id = None
                    if (
                        recovered_id is not None
                        and self._is_id_recovery_blocked(int(recovered_id), int(fi))
                    ):
                        reason = "id_recovery_blocked_after_struct_op"
                        recovered_id = None

                if reason == "accepted" and recovered_id is not None:
                    conflict_id = None
                    conflict_ioa = 0.0
                    for oid_conflict, exist_mask in current_masks.items():
                        oid_conflict_i = int(oid_conflict)
                        if oid_conflict_i < 100:
                            continue
                        if oid_conflict_i in {int(cand_id), int(recovered_id)}:
                            continue
                        if exist_mask is None or np.count_nonzero(exist_mask) <= 0:
                            continue
                        _, ioa_e2n_conflict, ioa_n2e_conflict = calculate_ioa_bidirectional(exist_mask, cand_mask)
                        max_ioa_conflict = float(max(ioa_e2n_conflict, ioa_n2e_conflict))
                        if max_ioa_conflict > float(conflict_ioa):
                            conflict_ioa = float(max_ioa_conflict)
                            conflict_id = int(oid_conflict_i)

                    if (
                        conflict_id is not None
                        and float(conflict_ioa) >= float(self.post_track_dup_ioa_threshold)
                    ):
                        conflict_mask = _to_bool_mask(current_masks.get(int(conflict_id)))
                        if conflict_mask is not None and np.count_nonzero(conflict_mask) > 0:
                            self._register_recent_post_duplicate(
                                int(conflict_id),
                                int(recovered_id),
                                conflict_mask,
                                frame_idx=int(fi),
                            )
                        try:
                            self.sam2_tracker.remove_object(
                                state,
                                int(conflict_id),
                                strict=False,
                                need_output=False,
                            )
                        except Exception:
                            pass
                        current_masks.pop(int(conflict_id), None)
                        current_boxes.pop(int(conflict_id), None)
                        accepted_in_frame.pop(int(conflict_id), None)
                        hist_rev.pop(int(conflict_id), None)

                    try:
                        self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                    except Exception:
                        pass
                    try:
                        self.sam2_tracker.add_new_mask(
                            state,
                            frame_idx=local_idx,
                            obj_id=int(recovered_id),
                            mask=cand_mask,
                        )
                    except Exception:
                        reason = "id_recovery_add_failed"
                    else:
                        rec_bbox = self._get_mask_bbox(cand_mask)
                        _ensure_hist_mask_at_current_step(int(recovered_id), cand_mask)
                        current_masks[int(recovered_id)] = cand_mask
                        current_boxes[int(recovered_id)] = rec_bbox
                        accepted_in_frame[int(recovered_id)] = cand_mask
                        accepted_new_ids.append(int(recovered_id))
                        all_new_object_ids.append(int(recovered_id))
                        all_new_object_frames.append(int(fi))
                        resolved_candidate_ids.add(int(cand_id))
                        reason = "id_recovery"
                        row_extra.update({
                            "recovered_id": int(recovered_id),
                            "conflict_removed_id": int(conflict_id) if conflict_id is not None else None,
                            "conflict_max_ioa": float(conflict_ioa),
                        })
                        special_handled = True

                if reason == "accepted" and not special_handled:
                    temporal_match = _find_temporal_reassign_target_replay(
                        cand_mask,
                        {
                            int(oid): mm
                            for oid, mm in current_masks.items()
                            if int(oid) >= 100 and mm is not None and np.count_nonzero(mm) > 0
                        },
                    )
                    if temporal_match is not None:
                        target_id = int(temporal_match.get("obj_id"))
                        try:
                            self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                        except Exception:
                            pass
                        try:
                            self.sam2_tracker.add_new_mask(
                                state,
                                frame_idx=local_idx,
                                obj_id=int(target_id),
                                mask=cand_mask,
                            )
                        except Exception:
                            reason = "temporal_reassign_add_failed"
                        else:
                            tbox = self._get_mask_bbox(cand_mask)
                            _ensure_hist_mask_at_current_step(int(target_id), cand_mask)
                            current_masks[int(target_id)] = cand_mask
                            current_boxes[int(target_id)] = tbox
                            accepted_in_frame[int(target_id)] = cand_mask
                            resolved_candidate_ids.add(int(cand_id))
                            reason = "temporal_reassign"
                            row_extra.update({
                                "target_id": int(target_id),
                                "temporal_score": float(temporal_match.get("score", 0.0)),
                            })
                            special_handled = True

                accepted = bool(reason in {"accepted", "id_recovery", "temporal_reassign"})
                if not accepted:
                    resolved_candidate_ids.add(int(cand_id))
                    try:
                        self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                    except Exception:
                        pass
                elif not special_handled:
                    # New object appears in reverse replay state.
                    pre_pad = [zero_mask.copy() for _ in range(local_idx)]
                    pre_pad.append(cand_mask)
                    hist_rev[int(cand_id)] = pre_pad
                    current_masks[int(cand_id)] = cand_mask
                    current_boxes[int(cand_id)] = self._get_mask_bbox(cand_mask)
                    accepted_in_frame[int(cand_id)] = cand_mask
                    accepted_new_ids.append(int(cand_id))
                    all_new_object_ids.append(int(cand_id))
                    all_new_object_frames.append(int(fi))
                    resolved_candidate_ids.add(int(cand_id))

                row = {
                    "cand_id": int(cand_id),
                    "box": [int(v) for v in cand_box],
                    "accepted": bool(accepted),
                    "reason": str(reason),
                    "area": int(np.count_nonzero(cand_mask)),
                    "score": float(cand.get("score", 0.0)),
                    "score_raw": float(cand.get("score_raw", 0.0)),
                    "score_source": str(cand.get("score_source", "")),
                    "near_hand": bool(cand.get("near_hand", False)),
                    "is_dup": bool(cand.get("is_dup", False)),
                    "dup_reason": str(cand.get("dup_reason", "")),
                    "proxy_new_to_x": float(cand.get("proxy_new_to_x", 0.0)),
                    "max_new_to_existing": float(cand.get("max_new_to_existing", 0.0)),
                    "max_existing_to_new": float(cand.get("max_existing_to_new", 0.0)),
                }
                if row_extra:
                    row.update(row_extra)
                candidate_rows.append(row)
                if self._is_new_accept_cap_reached(accepted_in_frame):
                    break

            # Remove candidate IDs not finalized this frame (same cleanup behavior as main path).
            for cand in frame_candidates:
                cand_id = int(cand.get("cand_id", -1))
                if int(cand_id) in resolved_candidate_ids:
                    continue
                try:
                    self.sam2_tracker.remove_object(state, int(cand_id), strict=False, need_output=False)
                except Exception:
                    pass

            # Main-equivalent stage: add from detection masks (moderate overlap, non-duplicate).
            if replay_target_masks:
                object_ids_local = [int(i) for i in current_masks.keys() if int(i) >= 100]
                for det_mask in list(replay_target_masks):
                    if det_mask is None or np.count_nonzero(det_mask) == 0:
                        continue
                    component_ioa = 0.0
                    component_parent = None
                    component_mask = None
                    if bool(self.enable_component_promote_from_det):
                        for oid in object_ids_local:
                            obj_mask = current_masks.get(int(oid))
                            if obj_mask is None or np.count_nonzero(obj_mask) == 0:
                                continue
                            mask_uint8 = obj_mask.astype(np.uint8)
                            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                            if num_labels <= 2:
                                continue
                            for label in range(1, num_labels):
                                comp_mask = (labels == label)
                                if np.count_nonzero(comp_mask) == 0:
                                    continue
                                _, ioa_c_to_d, ioa_d_to_c = calculate_ioa_bidirectional(comp_mask, det_mask)
                                # Component promotion should require near-duplicate
                                # agreement in both directions.
                                ioa = float(min(ioa_c_to_d, ioa_d_to_c))
                                if ioa > component_ioa:
                                    component_ioa = float(ioa)
                                    component_parent = int(oid)
                                    component_mask = comp_mask
                    component_promote_th = float(
                        max(
                            float(self.detect_component_add_ioa_threshold),
                            float(self.split_component_duplicate_ioa_th),
                        )
                    )
                    promote_from_component = bool(
                        bool(self.enable_component_promote_from_det)
                        and component_parent is not None
                        and component_mask is not None
                        and float(component_ioa) >= float(component_promote_th)
                    )

                    best_ioa = 0.0
                    best_ioa_other = 0.0
                    for oid in object_ids_local:
                        obj_mask = current_masks.get(int(oid))
                        if obj_mask is None or np.count_nonzero(obj_mask) == 0:
                            continue
                        _, ioa_o_to_d, ioa_d_to_o = calculate_ioa_bidirectional(obj_mask, det_mask)
                        pair_ioa = float(max(ioa_o_to_d, ioa_d_to_o))
                        best_ioa = max(float(best_ioa), pair_ioa)
                        if (
                            promote_from_component
                            and component_parent is not None
                            and int(oid) != int(component_parent)
                        ):
                            best_ioa_other = max(float(best_ioa_other), pair_ioa)

                    if promote_from_component and float(best_ioa_other) >= float(self.new_mask_inclusion_reject_ioa):
                        continue
                    force_promote_from_component = bool(promote_from_component)
                    if (not force_promote_from_component) and float(best_ioa) >= float(self.new_mask_reject_ioa):
                        continue
                    if (not force_promote_from_component) and float(best_ioa) < float(self.detect_add_ioa_threshold):
                        continue
                    mask_bool = _to_bool_mask(det_mask)
                    if mask_bool is None or np.count_nonzero(mask_bool) == 0:
                        continue

                    if not bool(force_promote_from_component):
                        temporal_match_det = _find_temporal_reassign_target_replay(
                            mask_bool,
                            {
                                int(oid): mm
                                for oid, mm in current_masks.items()
                                if int(oid) >= 100 and mm is not None and np.count_nonzero(mm) > 0
                            },
                        )
                        if temporal_match_det is not None:
                            target_id_det = int(temporal_match_det.get("obj_id"))
                            try:
                                self.sam2_tracker.add_new_mask(
                                    state,
                                    frame_idx=local_idx,
                                    obj_id=int(target_id_det),
                                    mask=mask_bool,
                                )
                            except Exception:
                                continue
                            tbox_det = self._get_mask_bbox(mask_bool)
                            _ensure_hist_mask_at_current_step(int(target_id_det), mask_bool)
                            current_masks[int(target_id_det)] = mask_bool
                            current_boxes[int(target_id_det)] = tbox_det
                            candidate_rows.append({
                                "cand_id": int(target_id_det),
                                "box": None,
                                "accepted": True,
                                "reason": "temporal_reassign_from_det",
                                "area": int(np.count_nonzero(mask_bool)),
                                "score": float(best_ioa),
                                "proxy_new_to_x": 0.0,
                                "max_new_to_existing": 0.0,
                                "max_existing_to_new": float(best_ioa),
                            })
                            continue

                    cooldown_hit_det = self._duplicate_cooldown_hit(
                        mask_bool,
                        frame_idx=int(fi),
                    )
                    if cooldown_hit_det is not None:
                        candidate_rows.append({
                            "cand_id": -1,
                            "box": None,
                            "accepted": False,
                            "reason": "recent_post_duplicate_from_det",
                            "area": int(np.count_nonzero(mask_bool)),
                            "score": float(best_ioa),
                            "proxy_new_to_x": 0.0,
                            "max_new_to_existing": 0.0,
                            "max_existing_to_new": float(best_ioa),
                            "cooldown_removed_id": int(cooldown_hit_det.get("removed_id", -1)),
                            "cooldown_kept_id": (
                                int(cooldown_hit_det.get("kept_id"))
                                if cooldown_hit_det.get("kept_id") is not None
                                else None
                            ),
                            "cooldown_age": int(cooldown_hit_det.get("age", -1)),
                            "cooldown_max_ioa": float(cooldown_hit_det.get("max_ioa", 0.0)),
                        })
                        continue

                    new_id = _alloc_replay_new_id()
                    try:
                        self.sam2_tracker.add_new_mask(
                            state,
                            frame_idx=local_idx,
                            obj_id=int(new_id),
                            mask=mask_bool,
                        )
                    except Exception:
                        continue

                    pre_pad = [zero_mask.copy() for _ in range(local_idx)]
                    pre_pad.append(mask_bool)
                    hist_rev[int(new_id)] = pre_pad
                    current_masks[int(new_id)] = mask_bool
                    current_boxes[int(new_id)] = self._get_mask_bbox(mask_bool)
                    accepted_new_ids.append(int(new_id))
                    all_new_object_ids.append(int(new_id))
                    all_new_object_frames.append(int(fi))
                    object_ids_local.append(int(new_id))
                    candidate_rows.append({
                        "cand_id": int(new_id),
                        "box": None,
                        "accepted": True,
                        "reason": "new_object_from_det",
                        "area": int(np.count_nonzero(mask_bool)),
                        "score": float(best_ioa),
                        "proxy_new_to_x": 0.0,
                        "max_new_to_existing": 0.0,
                        "max_existing_to_new": float(best_ioa),
                    })

            # Optional replay structural split stage.
            # Default is OFF so attach replay runs only new-object generation logic.
            object_ids_for_split: List[int] = []
            if bool(getattr(self, "replay_enable_structural_split_logic", False)):
                object_ids_for_split = [int(i) for i in current_masks.keys() if int(i) >= 100]
            for oid in object_ids_for_split:
                mask = current_masks.get(int(oid))
                if mask is None or np.count_nonzero(mask) == 0:
                    pending_splits_rev.pop(int(oid), None)
                    continue
                mask_uint8 = mask.astype(np.uint8)
                num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                if num_labels <= 2:
                    pending_splits_rev.pop(int(oid), None)
                    continue

                comp_info: List[Dict[str, Any]] = []
                for label in range(1, num_labels):
                    comp_mask = (labels == label)
                    area = int(np.sum(comp_mask))
                    if area <= 0:
                        continue
                    ys, xs = np.where(comp_mask)
                    comp_info.append({
                        "label": int(label),
                        "mask": comp_mask,
                        "area": int(area),
                        "centroid": (float(xs.mean()), float(ys.mean())),
                    })
                if not comp_info:
                    continue

                prev_mask = prev_tracked_masks_rev.get(int(oid))
                if prev_mask is not None and prev_mask.shape != mask.shape:
                    prev_mask = cv2.resize(
                        prev_mask.astype(np.uint8),
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                for comp in comp_info:
                    if prev_mask is None:
                        comp["overlap_prev"] = 0
                        comp["ioa"] = 0.0
                    else:
                        overlap = int(np.sum(np.logical_and(comp["mask"], prev_mask)))
                        comp["overlap_prev"] = int(overlap)
                        comp["ioa"] = float(overlap / max(comp["area"], 1))

                comp_info.sort(
                    key=lambda x: (
                        float(x.get("ioa", 0.0)),
                        int(x.get("overlap_prev", 0)),
                        int(x.get("area", 0)),
                    ),
                    reverse=True,
                )
                main = comp_info[0]
                main_area = float(main["area"])

                valid = [
                    c for c in comp_info
                    if int(c["area"]) >= int(self.min_mask_area)
                    and float(c["area"]) >= float(main_area) * float(self.split_min_area_ratio)
                ]
                if not valid:
                    current_masks[int(oid)] = main["mask"]
                    current_boxes[int(oid)] = self._get_mask_bbox(main["mask"])
                    if int(oid) in hist_rev and hist_rev[int(oid)]:
                        hist_rev[int(oid)][-1] = main["mask"]
                    continue

                duplicate_labels: Set[int] = set()
                for comp in valid:
                    if int(comp["label"]) == int(main["label"]):
                        continue
                    other_masks = {
                        int(other_id): other_mask
                        for other_id, other_mask in current_masks.items()
                        if int(other_id) >= 100
                        and int(other_id) != int(oid)
                        and other_mask is not None
                        and np.count_nonzero(other_mask) > 0
                    }
                    overlap_obj_id = self._find_duplicate_object(
                        comp["mask"],
                        other_masks,
                        ioa_threshold=float(duplicate_ioa_threshold),
                    )
                    if overlap_obj_id is not None:
                        duplicate_labels.add(int(comp["label"]))

                union_mask = np.zeros_like(mask_uint8, dtype=bool)
                for comp in valid:
                    if int(comp["label"]) in duplicate_labels:
                        continue
                    union_mask |= comp["mask"]
                current_masks[int(oid)] = union_mask
                current_boxes[int(oid)] = self._get_mask_bbox(union_mask)
                if int(oid) in hist_rev and hist_rev[int(oid)]:
                    hist_rev[int(oid)][-1] = union_mask
                try:
                    self.sam2_tracker.add_new_mask(
                        state,
                        frame_idx=local_idx,
                        obj_id=int(oid),
                        mask=union_mask,
                    )
                except Exception:
                    pass

                split_signal_masks: List[np.ndarray] = []
                for src in [list(replay_target_masks or []), list(replay_split_target_masks or [])]:
                    for m in src:
                        mb = _to_bool_mask(m)
                        if mb is None or np.count_nonzero(mb) <= 0:
                            continue
                        mb = mb.astype(bool)
                        dup = False
                        for prev in split_signal_masks:
                            if prev.shape == mb.shape and np.array_equal(prev, mb):
                                dup = True
                                break
                        if not dup:
                            split_signal_masks.append(mb.copy())
                det_duplicate_by_label: Dict[int, Dict[str, float]] = {}
                if split_signal_masks:
                    comp_dup_th = float(np.clip(self.split_component_duplicate_ioa_th, 0.0, 1.0))
                    for candidate in valid:
                        cand_label = int(candidate["label"])
                        if cand_label in duplicate_labels:
                            continue
                        best_det_idx = None
                        best_det_score = -1.0
                        for det_idx, det_mask in enumerate(split_signal_masks):
                            if det_mask is None or np.count_nonzero(det_mask) == 0:
                                continue
                            _, ioa_c_to_d, ioa_d_to_c = calculate_ioa_bidirectional(candidate["mask"], det_mask)
                            if ioa_c_to_d < comp_dup_th or ioa_d_to_c < comp_dup_th:
                                continue
                            score = float(min(ioa_c_to_d, ioa_d_to_c))
                            if score > best_det_score:
                                best_det_score = score
                                best_det_idx = int(det_idx)
                        if best_det_idx is not None:
                            det_duplicate_by_label[cand_label] = {
                                "det_idx": int(best_det_idx),
                                "score": float(best_det_score),
                            }
                direct_split_labels = {
                    int(lbl)
                    for lbl in det_duplicate_by_label.keys()
                    if int(lbl) != int(main["label"]) and int(lbl) not in duplicate_labels
                }
                has_direct_split_evidence = bool(len(direct_split_labels) > 0)
                any_component_det_backed = bool(len(det_duplicate_by_label) > 0)
                fallback_det_info = det_duplicate_by_label.get(int(main["label"]))
                if fallback_det_info is None and det_duplicate_by_label:
                    fallback_det_info = max(
                        det_duplicate_by_label.values(),
                        key=lambda r: float((r or {}).get("score", 0.0)),
                    )

                pending = pending_splits_rev.get(int(oid), {})
                updated_pending: Dict[str, Any] = {}
                split_applied = False
                for comp in valid:
                    if int(comp["label"]) in duplicate_labels:
                        continue
                    if int(comp["label"]) == int(main["label"]):
                        continue
                    det_info_comp = det_duplicate_by_label.get(int(comp["label"]))
                    fallback_split_evidence = bool(
                        det_info_comp is None
                        and not bool(has_direct_split_evidence)
                        and bool(any_component_det_backed)
                    )
                    split_evidence = bool(det_info_comp is not None or fallback_split_evidence)

                    best_key = None
                    best_ioa = 0.0
                    for key, entry in pending.items():
                        pmask = entry.get("mask")
                        if pmask is None or pmask.shape != comp["mask"].shape:
                            continue
                        overlap = int(np.sum(np.logical_and(pmask, comp["mask"])))
                        ioa = float(overlap / max(int(np.sum(comp["mask"])), 1))
                        if ioa > best_ioa:
                            best_ioa = float(ioa)
                            best_key = key

                    if best_key is None or best_ioa < 0.3:
                        key = f"{int(oid)}:{int(comp['label'])}:{int(fi)}"
                        entry = {
                            "frames": 0,
                            "mask": comp["mask"].copy(),
                            "centroid": comp["centroid"],
                            "split_score": 0,
                        }
                    else:
                        key = best_key
                        entry = pending[best_key]

                    if split_evidence:
                        entry["frames"] = int(entry.get("frames", 0)) + 1
                        entry["split_score"] = int(entry.get("split_score", 0)) + 1
                    else:
                        entry["frames"] = 0
                        entry["split_score"] = 0
                    entry["mask"] = comp["mask"].copy()
                    entry["centroid"] = comp["centroid"]

                    confirm_frames = max(1, int(self.split_confirm_frames))
                    if bool(self.split_multicomponent_instant_new):
                        split_ready = bool(split_evidence)
                    else:
                        split_ready = bool(split_evidence and int(entry.get("frames", 0)) >= confirm_frames)
                    if split_ready:
                        det_match = comp
                        det_info = det_info_comp if det_info_comp is not None else fallback_det_info
                        if det_info is None:
                            updated_pending[key] = entry
                            continue
                        keep_comp = main

                        # Keep component detector match is optional.
                        # Split component only needs detector-backed split evidence.
                        det_match_det_idx = int(det_info.get("det_idx", -1))
                        keep_comp_det_idx = -1
                        keep_comp_best_in = 0.0
                        for bidx, det_box in enumerate(list(boxes_rev[local_idx] or [])):
                            keep_in = float(self._mask_in_box_ratio(keep_comp["mask"], det_box))
                            if keep_in >= float(self.split_same_box_in_ratio_th) and keep_in > keep_comp_best_in:
                                keep_comp_best_in = float(keep_in)
                                keep_comp_det_idx = int(bidx)

                        if not bool(fallback_split_evidence):
                            if det_match_det_idx < 0:
                                updated_pending[key] = entry
                                continue
                            if keep_comp_det_idx >= 0 and int(det_match_det_idx) == int(keep_comp_det_idx):
                                updated_pending[key] = entry
                                continue

                        split_mask = det_match["mask"].copy()
                        keep_mask = np.zeros_like(split_mask, dtype=bool)
                        for keep_comp_cand in list(valid):
                            lbl_keep = int(keep_comp_cand.get("label", -1))
                            if int(lbl_keep) == int(det_match.get("label", -1)):
                                continue
                            if int(lbl_keep) in duplicate_labels:
                                continue
                            km = _to_bool_mask(keep_comp_cand.get("mask"))
                            if km is None or np.count_nonzero(km) <= 0:
                                continue
                            keep_mask |= km
                        keep_mask = np.logical_and(keep_mask, np.logical_not(split_mask))
                        if np.count_nonzero(keep_mask) <= 0:
                            updated_pending[key] = entry
                            continue

                        new_id = _alloc_replay_new_id()
                        try:
                            self.sam2_tracker.add_new_mask(
                                state,
                                frame_idx=local_idx,
                                obj_id=int(new_id),
                                mask=split_mask,
                            )
                        except Exception:
                            updated_pending[key] = entry
                            continue

                        pre_pad_new = [zero_mask.copy() for _ in range(local_idx)]
                        pre_pad_new.append(split_mask)
                        hist_rev[int(new_id)] = pre_pad_new
                        current_masks[int(new_id)] = split_mask
                        current_boxes[int(new_id)] = self._get_mask_bbox(split_mask)
                        accepted_new_ids.append(int(new_id))
                        all_new_object_ids.append(int(new_id))
                        all_new_object_frames.append(int(fi))

                        current_masks[int(oid)] = keep_mask
                        current_boxes[int(oid)] = self._get_mask_bbox(keep_mask)
                        if int(oid) in hist_rev and hist_rev[int(oid)]:
                            hist_rev[int(oid)][-1] = keep_mask
                        try:
                            self.sam2_tracker.add_new_mask(
                                state,
                                frame_idx=local_idx,
                                obj_id=int(oid),
                                mask=keep_mask,
                            )
                        except Exception:
                            pass

                        candidate_rows.append({
                            "cand_id": int(new_id),
                            "box": None,
                            "accepted": True,
                            "reason": "split_confirm_fallback" if bool(fallback_split_evidence) else "split_confirm",
                            "area": int(np.count_nonzero(split_mask)),
                            "score": float(det_info.get("score", 0.0)),
                            "proxy_new_to_x": 0.0,
                            "max_new_to_existing": 0.0,
                            "max_existing_to_new": 0.0,
                        })
                        split_applied = True
                        break

                    updated_pending[key] = entry

                pending_splits_rev[int(oid)] = updated_pending
                if split_applied:
                    continue

            # Relevant Y = Y overlapping either A or B on this reverse frame.
            # (X is kept as replay state anchor, but not used as hard pair gate.)
            accepted_relevant_ids: List[int] = []
            accepted_relevant_rows: List[Dict[str, Any]] = []
            if accepted_new_ids:
                if a_id is None and b_id is None:
                    accepted_relevant_ids = [int(x) for x in accepted_new_ids]
                    for nid in list(accepted_new_ids):
                        accepted_relevant_rows.append({
                            "cand_id": int(nid),
                            "pass": True,
                            "reason": "ab_id_missing_pair_check_skipped",
                        })
                else:
                    m_a_rel = (
                        _to_bool_mask(frame_tracked_masks.get(int(a_id)))
                        if a_id is not None
                        else None
                    )
                    m_b_rel = (
                        _to_bool_mask(frame_tracked_masks.get(int(b_id)))
                        if b_id is not None
                        else None
                    )
                    has_a = bool(m_a_rel is not None and np.count_nonzero(m_a_rel) > 0)
                    has_b = bool(m_b_rel is not None and np.count_nonzero(m_b_rel) > 0)
                    m_x_rel = _to_bool_mask(current_masks.get(int(proxy_obj_id)))
                    pair_th = float(self.attach_reverse_xy_match_th)
                    for nid in list(accepted_new_ids):
                        m_y_rel = _to_bool_mask(current_masks.get(int(nid)))
                        if m_y_rel is None or np.count_nonzero(m_y_rel) == 0:
                            accepted_relevant_rows.append({
                                "cand_id": int(nid),
                                "pair_score": 0.0,
                                "pass": False,
                                "reason": "y_missing",
                            })
                            continue
                        if not bool(has_a or has_b):
                            accepted_relevant_rows.append({
                                "cand_id": int(nid),
                                "pair_score": 0.0,
                                "pass": False,
                                "reason": "ab_missing",
                            })
                            continue

                        a_in_x = 0.0
                        b_in_x = 0.0
                        if m_x_rel is not None and np.count_nonzero(m_x_rel) > 0:
                            if has_a:
                                a_in_x = float(self._mask_in_mask_ratio(m_a_rel, m_x_rel))
                            if has_b:
                                b_in_x = float(self._mask_in_mask_ratio(m_b_rel, m_x_rel))

                        a_in_y = 0.0
                        b_in_y = 0.0
                        a_overlap_y = 0.0
                        b_overlap_y = 0.0
                        if has_a:
                            a_in_y = float(self._mask_in_mask_ratio(m_a_rel, m_y_rel))
                            _, ioa_a_to_y, ioa_y_to_a = calculate_ioa_bidirectional(m_a_rel, m_y_rel)
                            a_overlap_y = float(max(ioa_a_to_y, ioa_y_to_a))
                        if has_b:
                            b_in_y = float(self._mask_in_mask_ratio(m_b_rel, m_y_rel))
                            _, ioa_b_to_y, ioa_y_to_b = calculate_ioa_bidirectional(m_b_rel, m_y_rel)
                            b_overlap_y = float(max(ioa_b_to_y, ioa_y_to_b))

                        pass_a = bool(has_a and float(a_overlap_y) >= float(pair_th))
                        pass_b = bool(has_b and float(b_overlap_y) >= float(pair_th))
                        pair_score = float(max(a_overlap_y, b_overlap_y))
                        assignment = "Y~(A|B)"
                        if bool(pass_a) and bool(pass_b):
                            match_mode = "y_overlaps_ab"
                        elif bool(pass_a) or bool(pass_b):
                            match_mode = "y_overlaps_one_of_ab"
                        else:
                            match_mode = "no_ab_overlap"

                        rel_pass = bool(pass_a or pass_b)
                        accepted_relevant_rows.append({
                            "cand_id": int(nid),
                            "pair_score": float(pair_score),
                            "pair_th": float(pair_th),
                            "assignment": str(assignment),
                            "match_mode": str(match_mode),
                            "pass_xy_xA_yB": False,
                            "pass_yx_xB_yA": False,
                            "pass_y_overlaps_a": bool(pass_a),
                            "pass_y_overlaps_b": bool(pass_b),
                            "a_in_x": float(a_in_x),
                            "b_in_x": float(b_in_x),
                            "a_in_y": float(a_in_y),
                            "b_in_y": float(b_in_y),
                            "a_overlap_y": float(a_overlap_y),
                            "b_overlap_y": float(b_overlap_y),
                            "pass": bool(rel_pass),
                        })
                        if rel_pass:
                            accepted_relevant_ids.append(int(nid))
            if accepted_new_ids:
                birth_map = debug_birth_new_masks_by_frame.setdefault(int(fi), {})
                for nid in list(accepted_new_ids):
                    nm_birth = _to_bool_mask(current_masks.get(int(nid)))
                    if nm_birth is None or np.count_nonzero(nm_birth) == 0:
                        continue
                    birth_map[int(nid)] = nm_birth.copy()
            # In merge reverse replay, unmatched Y candidates should not poison later
            # frames. Keep X (and matched Y only), purge unmatched accepted new IDs.
            purged_unmatched_ids: List[int] = []
            if accepted_new_ids and a_id is not None and b_id is not None:
                relevant_set = set(int(x) for x in accepted_relevant_ids)
                purge_ids = [int(nid) for nid in list(accepted_new_ids) if int(nid) not in relevant_set]
                if purge_ids:
                    purge_set = set(int(x) for x in purge_ids)
                    for nid in list(purge_ids):
                        try:
                            self.sam2_tracker.remove_object(
                                state,
                                int(nid),
                                strict=False,
                                need_output=False,
                            )
                        except Exception:
                            pass
                        hist_rev.pop(int(nid), None)
                        current_masks.pop(int(nid), None)
                        current_boxes.pop(int(nid), None)
                        accepted_in_frame.pop(int(nid), None)
                        pending_splits_rev.pop(int(nid), None)
                    accepted_new_ids = [int(nid) for nid in list(accepted_new_ids) if int(nid) not in purge_set]
                    all_pairs = list(zip(list(all_new_object_ids), list(all_new_object_frames)))
                    all_pairs = [
                        (int(nid), int(ff))
                        for nid, ff in all_pairs
                        if int(nid) not in purge_set
                    ]
                    all_new_object_ids = [int(nid) for nid, _ in all_pairs]
                    all_new_object_frames = [int(ff) for _, ff in all_pairs]
                    purged_unmatched_ids = [int(x) for x in list(purge_ids)]
                    for crow in candidate_rows:
                        try:
                            cid_row = int(crow.get("cand_id", -1))
                        except Exception:
                            cid_row = -1
                        if cid_row in purge_set and bool(crow.get("accepted", False)):
                            crow["post_pair_action"] = "purged_unmatched_y"
            # Replay new-object log should follow forward semantics:
            # once a new object is accepted by candidate logic, record it.
            # Pair relevance (A/B match) is evaluated later.
            for rid in list(accepted_new_ids):
                new_object_ids.append(int(rid))
                new_object_frames.append(int(fi))

            prev_tracked_masks_rev = {
                int(oid): m.copy()
                for oid, m in current_masks.items()
                if m is not None and np.count_nonzero(m) > 0
            }

            frame_rows.append({
                "frame_idx": int(fi),
                "accepted_new_ids": [int(x) for x in accepted_new_ids],
                "accepted_count": int(len(accepted_new_ids)),
                "accepted_relevant_new_ids": [int(x) for x in accepted_relevant_ids],
                "accepted_relevant_count": int(len(accepted_relevant_ids)),
                "accepted_relevant_rows": accepted_relevant_rows[:16],
                "purged_unmatched_ids": [int(x) for x in purged_unmatched_ids],
                "candidate_count": int(len(candidate_rows)),
                "det_target_count": int(len(replay_target_masks)),
                "split_target_count": int(len(replay_split_target_masks)),
                "candidates": candidate_rows[:16],
            })
            processed_last = int(local_idx)
            # Forward-equivalent stop condition:
            # - base: stop when a new object is created
            # - optional: keep N extra reverse steps after first new object
            #   so XY<->AB can be checked on immediate following frames.
            if bool(stop_on_first_new_object):
                if len(accepted_new_ids) > 0 and first_new_local_idx is None:
                    first_new_local_idx = int(local_idx)
                if first_new_local_idx is not None:
                    if int(local_idx - int(first_new_local_idx)) >= int(post_new_object_extra_steps):
                        break
            # Stop reverse replay as soon as a pair-matched Y appears.
            if bool(stop_on_first_matched_new_object) and len(accepted_relevant_ids) > 0:
                break

        # Convert reverse histories to forward (increasing frame_idx) mapping.
        indices_proc_rev = indices_rev[: processed_last + 1]
        indices_used = list(reversed(indices_proc_rev))
        proxy_masks_by_frame: Dict[int, np.ndarray] = {}
        new_masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        all_new_masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        relevant_new_set = set(int(x) for x in new_object_ids)
        all_new_set = set(int(x) for x in all_new_object_ids)
        for oid, rev_hist in hist_rev.items():
            seq = list(reversed(rev_hist))
            if len(seq) < len(indices_used):
                pad = [zero_mask.copy() for _ in range(len(indices_used) - len(seq))]
                seq = pad + seq
            if len(seq) > len(indices_used):
                seq = seq[-len(indices_used):]
            for fi, m in zip(indices_used, seq):
                if m is None or np.count_nonzero(m) == 0:
                    continue
                if int(oid) == int(proxy_obj_id):
                    proxy_masks_by_frame[int(fi)] = m
                elif int(oid) in all_new_set:
                    all_new_masks_by_frame.setdefault(int(fi), {})
                    all_new_masks_by_frame[int(fi)][int(oid)] = m
                    if int(oid) in relevant_new_set:
                        new_masks_by_frame.setdefault(int(fi), {})
                        new_masks_by_frame[int(fi)][int(oid)] = m
        # Merge debug birth masks so unmatched/purged Y remains visible on its
        # birth frame in reverse debug visuals.
        for fi_dbg, id_map_dbg in debug_birth_new_masks_by_frame.items():
            dst = all_new_masks_by_frame.setdefault(int(fi_dbg), {})
            for nid_dbg, mm_dbg in (id_map_dbg or {}).items():
                mm_b = _to_bool_mask(mm_dbg)
                if mm_b is None or np.count_nonzero(mm_b) == 0:
                    continue
                dst.setdefault(int(nid_dbg), mm_b.copy())

        seed_mask_bbox = self._get_mask_bbox(seed_mask)
        return {
            "ok": True,
            "reason": "ok",
            "proxy_id": int(proxy_obj_id),
            "a_id": None if a_id is None else int(a_id),
            "b_id": None if b_id is None else int(b_id),
            "seed_box": [int(v) for v in seed_box],
            "seed_source": str(seed_source),
            "seed_mask_area": int(np.count_nonzero(seed_mask)),
            "seed_mask_bbox": [int(v) for v in list(seed_mask_bbox or [0, 0, 0, 0])[:4]],
            "frame_indices": [int(fi) for fi in indices_used],
            "proxy_masks_by_frame": proxy_masks_by_frame,
            "new_masks_by_frame": new_masks_by_frame,
            "all_new_masks_by_frame": all_new_masks_by_frame,
            "new_object_ids": [int(x) for x in sorted(set(new_object_ids))],
            "new_object_frames": [int(x) for x in sorted(set(new_object_frames), reverse=True)],
            "all_new_object_ids": [int(x) for x in sorted(set(all_new_object_ids))],
            "all_new_object_frames": [int(x) for x in sorted(set(all_new_object_frames), reverse=True)],
            # Keep full reverse rows for downstream Y-candidate extraction.
            # Truncating here can hide the frame where relevant Y first appears.
            "rows": frame_rows,
        }

    @staticmethod
    def _attach_session_cache_key(
        pair: Tuple[int, int],
        start_frame_idx: int,
    ) -> Tuple[int, int, int]:
        a, b = int(min(pair[0], pair[1])), int(max(pair[0], pair[1]))
        return (int(a), int(b), int(start_frame_idx))

    def _prune_attach_sessions(self, current_frame_idx: int) -> None:
        ttl = int(max(1, int(self.attach_session_ttl_frames)))
        keep_sessions: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for key, row in self._attach_sessions.items():
            if not isinstance(row, dict):
                continue
            try:
                start_fi = int(row.get("start_frame_idx", -1))
            except Exception:
                start_fi = -1
            if start_fi >= 0 and int(current_frame_idx) - int(start_fi) <= ttl:
                keep_sessions[key] = row
        self._attach_sessions = keep_sessions

        keep_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for key, row in self._attach_replay_cache.items():
            if not isinstance(row, dict):
                continue
            try:
                start_fi = int(row.get("start_frame_idx", -1))
            except Exception:
                start_fi = -1
            if start_fi >= 0 and int(current_frame_idx) - int(start_fi) <= ttl:
                keep_cache[key] = row
        self._attach_replay_cache = keep_cache

    @torch.inference_mode()
    def replay_forward_from_seed_mask(
        self,
        frames_bgr: List[np.ndarray],
        frame_indices: List[int],
        seed_mask_start: np.ndarray,
        *,
        proxy_obj_id: int,
        a_id: int,
        b_id: int,
        tracked_masks_by_frame: Dict[int, Dict[int, np.ndarray]],
        max_forward: int = 15,
        count_on_th: int = 2,
    ) -> Dict[str, Any]:
        if frames_bgr is None or len(frames_bgr) == 0:
            return {"ok": False, "reason": "no_frames"}
        if frame_indices is None or len(frame_indices) != len(frames_bgr):
            return {"ok": False, "reason": "frame_indices_mismatch"}

        H, W = frames_bgr[0].shape[:2]
        seed_mask = _to_bool_mask(seed_mask_start)
        if seed_mask is None or np.count_nonzero(seed_mask) == 0:
            return {"ok": False, "reason": "invalid_seed_mask"}
        if seed_mask.shape[:2] != (H, W):
            seed_mask = cv2.resize(seed_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        K = min(len(frames_bgr), int(max(1, int(max_forward))))
        frames = frames_bgr[:K]
        indices = [int(x) for x in frame_indices[:K]]

        state = self.sam2_tracker.init_state(frames[0])
        self.sam2_tracker.add_new_mask(
            state,
            frame_idx=0,
            obj_id=int(proxy_obj_id),
            mask=seed_mask,
        )

        proxy_masks_by_frame: Dict[int, np.ndarray] = {int(indices[0]): seed_mask}
        rows: List[Dict[str, Any]] = []
        hit_frames: List[int] = []
        count_on = 0

        for local_idx in range(0, K):
            if local_idx > 0:
                state = self.sam2_tracker.add_frame(state, frames[local_idx])
                _, ids_now, logits_now, state = self.sam2_tracker.get_mask(state, local_idx)
                cur_mask = None
                if ids_now is not None and logits_now is not None and len(ids_now) == len(logits_now):
                    for j, oid in enumerate(ids_now):
                        try:
                            if int(oid) != int(proxy_obj_id):
                                continue
                            mm = self._logits_to_bool_mask(logits_now[j], H, W)
                            if mm is not None and np.count_nonzero(mm) > 0:
                                cur_mask = mm
                            break
                        except Exception:
                            continue
                if cur_mask is not None:
                    proxy_masks_by_frame[int(indices[local_idx])] = cur_mask

            fi = int(indices[local_idx])
            m_x = _to_bool_mask(proxy_masks_by_frame.get(int(fi)))
            frame_masks = tracked_masks_by_frame.get(int(fi), {}) or {}
            m_a = _to_bool_mask(frame_masks.get(int(a_id)))
            m_b = _to_bool_mask(frame_masks.get(int(b_id)))
            a_in_x = 0.0
            b_in_x = 0.0
            pass_now = False
            if (
                m_x is not None and np.count_nonzero(m_x) > 0
                and m_a is not None and np.count_nonzero(m_a) > 0
                and m_b is not None and np.count_nonzero(m_b) > 0
            ):
                a_in_x = float(self._mask_in_mask_ratio(m_a, m_x))
                b_in_x = float(self._mask_in_mask_ratio(m_b, m_x))
                pass_now = bool(
                    a_in_x >= float(self.attach_mask_in_th)
                    and b_in_x >= float(self.attach_mask_in_th)
                )
            if pass_now:
                count_on += 1
                hit_frames.append(int(fi))
            rows.append({
                "frame_idx": int(fi),
                "a_in_x": float(a_in_x),
                "b_in_x": float(b_in_x),
                "pass": bool(pass_now),
            })
            if int(count_on) >= int(max(1, int(count_on_th))):
                return {
                    "ok": True,
                    "reason": "ok",
                    "proxy_id": int(proxy_obj_id),
                    "frame_indices": [int(x) for x in indices[: local_idx + 1]],
                    "proxy_masks_by_frame": {
                        int(x): m for x, m in proxy_masks_by_frame.items() if int(x) <= int(fi)
                    },
                    "window": int(max_forward),
                    "count_on": int(count_on),
                    "hit_frames": [int(x) for x in hit_frames],
                    "rows": rows,
                    "pass": True,
                    "early_stop": True,
                }

        return {
            "ok": True,
            "reason": "ok",
            "proxy_id": int(proxy_obj_id),
            "frame_indices": [int(x) for x in indices],
            "proxy_masks_by_frame": proxy_masks_by_frame,
            "window": int(max_forward),
            "count_on": int(count_on),
            "hit_frames": [int(x) for x in hit_frames],
            "rows": rows,
            "pass": bool(int(count_on) >= int(max(1, int(count_on_th)))),
            "early_stop": False,
        }

    @torch.inference_mode()
    def _replay_track_masks_from_seed(
        self,
        *,
        frames_bgr: List[np.ndarray],
        frame_indices: List[int],
        seed_mask_start: np.ndarray,
        proxy_obj_id: int,
    ) -> Dict[int, np.ndarray]:
        """Track a single seeded mask forward on a local replay state."""
        out: Dict[int, np.ndarray] = {}
        if frames_bgr is None or len(frames_bgr) == 0:
            return out
        if frame_indices is None or len(frame_indices) != len(frames_bgr):
            return out
        H, W = frames_bgr[0].shape[:2]
        seed_mask = _to_bool_mask(seed_mask_start)
        if seed_mask is None or np.count_nonzero(seed_mask) <= 0:
            return out
        if seed_mask.shape[:2] != (H, W):
            seed_mask = cv2.resize(
                seed_mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if np.count_nonzero(seed_mask) <= 0:
            return out
        try:
            state = self.sam2_tracker.init_state(frames_bgr[0])
            self.sam2_tracker.add_new_mask(
                state,
                frame_idx=0,
                obj_id=int(proxy_obj_id),
                mask=seed_mask,
            )
        except Exception:
            return out

        out[int(frame_indices[0])] = seed_mask.copy()
        for local_idx in range(1, len(frames_bgr)):
            fi = int(frame_indices[local_idx])
            try:
                state = self.sam2_tracker.add_frame(state, frames_bgr[local_idx])
                _, ids_now, logits_now, state = self.sam2_tracker.get_mask(state, local_idx)
            except Exception:
                continue
            cur_mask = None
            if (
                ids_now is not None
                and logits_now is not None
                and len(ids_now) == len(logits_now)
            ):
                for j, oid in enumerate(ids_now):
                    try:
                        if int(oid) != int(proxy_obj_id):
                            continue
                        mm = self._logits_to_bool_mask(logits_now[j], H, W)
                        if mm is not None and np.count_nonzero(mm) > 0:
                            cur_mask = mm
                        break
                    except Exception:
                        continue
            if cur_mask is not None and np.count_nonzero(cur_mask) > 0:
                out[int(fi)] = cur_mask.copy()
        return out

    @torch.inference_mode()
    def _validate_split_union_vs_parent_forward_track(
        self,
        *,
        parent_id: int,
        child_id: Optional[int] = None,
        split_start_frame: Optional[int] = None,
        split_mask_now: np.ndarray,
        tracked_masks_now: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Validate split by replaying only new-object mask and checking mask inclusion against forward tracked objects."""
        req_frames = int(max(1, int(self.attach_proxy_signal_persist_frames)))
        same_parent_req_frames = int(max(1, int(self.split_new_det_pair_persist_frames)))
        same_parent_in_th = float(np.clip(self.split_new_det_pair_in_th, 0.0, 1.0))
        replay_window_frames = int(max(1, int(getattr(self, "split_new_replay_window_frames", same_parent_req_frames))))
        frame_hist = list(self._frame_history)
        idx_hist = [int(x) for x in list(self._frame_idx_history)]
        if len(frame_hist) == 0 or len(idx_hist) != len(frame_hist):
            return {"pass": False, "reason": "history_unavailable", "required_frames": int(req_frames)}
        need_frames = int(max(req_frames, same_parent_req_frames))
        if len(frame_hist) < int(need_frames):
            return {
                "pass": False,
                "reason": "insufficient_history_frames",
                "required_frames": int(need_frames),
                "available_frames": int(len(frame_hist)),
            }
        window_use = int(max(need_frames, replay_window_frames))
        window_use = int(min(window_use, len(frame_hist)))
        frames_win = frame_hist[-window_use:]
        idx_win = idx_hist[-window_use:]
        split_start_fi: Optional[int] = None
        if split_start_frame is not None:
            try:
                split_start_fi = int(split_start_frame)
            except Exception:
                split_start_fi = None
        if not frames_win or not idx_win or len(frames_win) != len(idx_win):
            return {"pass": False, "reason": "window_build_failed", "required_frames": int(need_frames)}
        if split_start_fi is not None:
            available_before_seed = int(sum(1 for fi in idx_win if int(fi) < int(split_start_fi)))
            if available_before_seed < int(need_frames):
                return {
                    "pass": False,
                    "reason": "insufficient_history_frames_before_seed",
                    "required_frames": int(need_frames),
                    "available_frames": int(available_before_seed),
                    "split_start_frame": int(split_start_fi),
                }
        elif len(idx_win) < int(need_frames):
            return {
                "pass": False,
                "reason": "insufficient_history_frames",
                "required_frames": int(need_frames),
                "available_frames": int(len(idx_win)),
            }
        frames_by_index = {
            int(fi): frame
            for fi, frame in zip(idx_win, frames_win)
            if frame is not None
        }
        debug_frame_indices: List[int] = [int(x) for x in idx_win]

        def _finalize(
            payload: Dict[str, Any],
            *,
            parent_masks_dbg: Optional[Dict[int, np.ndarray]] = None,
            split_fwd_dbg: Optional[Dict[int, np.ndarray]] = None,
            split_back_dbg: Optional[Dict[int, np.ndarray]] = None,
        ) -> Dict[str, Any]:
            out = dict(payload or {})
            if bool(self.split_replay_debug):
                debug_parent_id = int(parent_id)
                try:
                    if out.get("matched_parent_id") is not None:
                        debug_parent_id = int(out.get("matched_parent_id"))
                except Exception:
                    debug_parent_id = int(parent_id)
                debug_dir = self._save_split_replay_debug_visuals(
                    parent_id=int(debug_parent_id),
                    child_id=int(child_id) if child_id is not None else None,
                    frame_indices=[int(x) for x in list(debug_frame_indices or [])],
                    frames_by_index=frames_by_index,
                    parent_masks_by_frame=dict(parent_masks_dbg or {}),
                    keep_fwd_masks_by_frame={},
                    split_fwd_masks_by_frame=dict(split_fwd_dbg or {}),
                    keep_back_masks_by_frame={},
                    split_back_masks_by_frame=dict(split_back_dbg or {}),
                    result=out,
                )
                if isinstance(debug_dir, str) and debug_dir:
                    out["debug_dir"] = str(debug_dir)
            return out

        tracked_hist = list(self._tracked_masks_history)
        tracked_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        idx_prev = [int(x) for x in idx_hist[:-1]]
        hist_use = int(min(len(tracked_hist), len(idx_prev)))
        if hist_use > 0:
            tracked_hist_use = tracked_hist[-hist_use:]
            idx_prev_use = idx_prev[-hist_use:]
        else:
            tracked_hist_use = []
            idx_prev_use = []
        for fi, row_raw in zip(idx_prev_use, tracked_hist_use):
            row = dict(row_raw or {})
            tracked_by_frame[int(fi)] = {
                int(oid): mm.copy()
                for oid, mm in row.items()
                if mm is not None and np.count_nonzero(mm) > 0
            }

        cur_frame_idx = int(self.frame_idx)
        tracked_by_frame[int(cur_frame_idx)] = {
            int(oid): mm.copy()
            for oid, mm in dict(tracked_masks_now or {}).items()
            if mm is not None and np.count_nonzero(mm) > 0
        }
        if int(idx_win[-1]) != int(cur_frame_idx):
            tracked_by_frame[int(idx_win[-1])] = {
                int(oid): mm.copy()
                for oid, mm in dict(tracked_masks_now or {}).items()
                if mm is not None and np.count_nonzero(mm) > 0
            }

        parent_hint_masks: Dict[int, np.ndarray] = {}
        eval_object_ids: Set[int] = set()
        for fi in idx_win:
            frame_masks = dict(tracked_by_frame.get(int(fi), {}) or {})
            pm = _to_bool_mask(frame_masks.get(int(parent_id)))
            if pm is not None and np.count_nonzero(pm) > 0:
                parent_hint_masks[int(fi)] = pm.astype(bool).copy()
            for oid_raw, mm_raw in frame_masks.items():
                try:
                    oid_i = int(oid_raw)
                except Exception:
                    continue
                if int(oid_i) < 100:
                    continue
                if child_id is not None and int(oid_i) == int(child_id):
                    continue
                mm = _to_bool_mask(mm_raw)
                if mm is None or np.count_nonzero(mm) <= 0:
                    continue
                eval_object_ids.add(int(oid_i))
        if not eval_object_ids:
            return _finalize({
                "pass": False,
                "reason": "no_compare_objects",
                "required_frames": int(same_parent_req_frames),
                "frame_indices": [int(x) for x in idx_win],
            }, parent_masks_dbg=parent_hint_masks)

        split_now_b = _to_bool_mask(split_mask_now)
        if (
            split_now_b is None
            or np.count_nonzero(split_now_b) <= 0
        ):
            return _finalize(
                {"pass": False, "reason": "invalid_child_seed_mask", "required_frames": int(req_frames)},
                parent_masks_dbg=parent_hint_masks,
            )

        proxy_base = int(960000 + (int(parent_id) % 10000) * 10)
        split_back = self._replay_track_masks_from_seed(
            frames_bgr=list(reversed(frames_win)),
            frame_indices=list(reversed(idx_win)),
            seed_mask_start=split_now_b,
            proxy_obj_id=int(proxy_base + 1),
        )
        seed_frame_idx = int(idx_win[0])
        split_seed = _to_bool_mask(split_back.get(int(seed_frame_idx)))
        if (
            split_seed is None
            or np.count_nonzero(split_seed) <= 0
        ):
            return _finalize({
                "pass": False,
                "reason": "backward_seed_missing",
                "required_frames": int(req_frames),
                "seed_frame_idx": int(seed_frame_idx),
            }, parent_masks_dbg=parent_hint_masks, split_back_dbg=split_back)

        split_fwd = self._replay_track_masks_from_seed(
            frames_bgr=frames_win,
            frame_indices=idx_win,
            seed_mask_start=split_seed,
            proxy_obj_id=int(proxy_base + 2),
        )
        if split_start_fi is not None:
            eval_frame_indices = [int(fi) for fi in idx_win if int(fi) < int(split_start_fi)]
            eval_frame_indices = sorted(eval_frame_indices, reverse=True)
            split_eval_masks: Dict[int, np.ndarray] = split_back
            split_eval_direction = "backward_to_past"
        else:
            eval_frame_indices = [int(fi) for fi in idx_win]
            split_eval_masks = split_fwd
            split_eval_direction = "forward_chronological"
        if not eval_frame_indices:
            return _finalize({
                "pass": False,
                "reason": "no_eval_frames",
                "required_frames": int(same_parent_req_frames),
                "split_start_frame": None if split_start_fi is None else int(split_start_fi),
            }, parent_masks_dbg=parent_hint_masks, split_fwd_dbg=split_fwd, split_back_dbg=split_back)
        debug_frame_indices = [int(x) for x in eval_frame_indices]

        rows: List[Dict[str, Any]] = []
        obj_streak: Dict[int, int] = {int(oid): 0 for oid in eval_object_ids}
        obj_max_streak: Dict[int, int] = {int(oid): 0 for oid in eval_object_ids}
        obj_hit_count: Dict[int, int] = {int(oid): 0 for oid in eval_object_ids}
        obj_sum_ratio: Dict[int, float] = {int(oid): 0.0 for oid in eval_object_ids}
        parent_pass_count = 0
        parent_tail = 0
        parent_max_tail = 0
        for fi in eval_frame_indices:
            ms = _to_bool_mask(split_eval_masks.get(int(fi)))
            frame_masks = dict(tracked_by_frame.get(int(fi), {}) or {})
            if ms is None or np.count_nonzero(ms) <= 0:
                for oid in eval_object_ids:
                    obj_streak[int(oid)] = 0
                rows.append({
                    "frame_idx": int(fi),
                    "reason": "split_mask_missing",
                    "pass": False,
                    "best_container_id": None,
                    "best_container_in": 0.0,
                    "candidate_parent_in": 0.0,
                    "included_ids": [],
                })
                parent_tail = 0
                continue

            best_container_id: Optional[int] = None
            best_container_in = 0.0
            included_ids: List[int] = []
            ratio_by_obj: Dict[int, float] = {}
            for oid in sorted(eval_object_ids):
                mm = _to_bool_mask(frame_masks.get(int(oid)))
                if mm is None or np.count_nonzero(mm) <= 0:
                    ratio = 0.0
                else:
                    ratio = float(self._mask_in_mask_ratio(ms, mm))
                ratio_by_obj[int(oid)] = float(ratio)
                if float(ratio) >= float(same_parent_in_th):
                    included_ids.append(int(oid))
                    if best_container_id is None or float(ratio) > float(best_container_in):
                        best_container_id = int(oid)
                        best_container_in = float(ratio)

            for oid in eval_object_ids:
                oid_i = int(oid)
                ratio = float(ratio_by_obj.get(int(oid_i), 0.0))
                if ratio >= float(same_parent_in_th):
                    obj_streak[int(oid_i)] = int(obj_streak.get(int(oid_i), 0)) + 1
                    obj_max_streak[int(oid_i)] = max(
                        int(obj_max_streak.get(int(oid_i), 0)),
                        int(obj_streak[int(oid_i)]),
                    )
                    obj_hit_count[int(oid_i)] = int(obj_hit_count.get(int(oid_i), 0)) + 1
                    obj_sum_ratio[int(oid_i)] = float(obj_sum_ratio.get(int(oid_i), 0.0)) + float(ratio)
                else:
                    obj_streak[int(oid_i)] = 0

            candidate_parent_in = float(ratio_by_obj.get(int(parent_id), 0.0))
            pass_now = bool(candidate_parent_in >= float(same_parent_in_th))
            rows.append({
                "frame_idx": int(fi),
                "pass": bool(pass_now),
                "best_container_id": int(best_container_id) if best_container_id is not None else None,
                "best_container_in": float(best_container_in),
                "candidate_parent_in": float(candidate_parent_in),
                "included_ids": [int(x) for x in included_ids],
                "same_parent_in_th": float(same_parent_in_th),
                "split_eval_direction": str(split_eval_direction),
            })
            if pass_now:
                parent_pass_count += 1
                parent_tail += 1
                parent_max_tail = max(int(parent_max_tail), int(parent_tail))
            else:
                parent_tail = 0

        matched_parent_id: Optional[int] = None
        matched_parent_tail_now = 0
        matched_parent_max_tail = 0
        matched_parent_mean_ratio = 0.0
        best_rank: Optional[Tuple[int, int, float, int, int]] = None
        for oid in sorted(eval_object_ids):
            oid_i = int(oid)
            tail_now = int(obj_streak.get(int(oid_i), 0))
            max_tail = int(obj_max_streak.get(int(oid_i), 0))
            if max_tail <= 0:
                continue
            hit_count = int(obj_hit_count.get(int(oid_i), 0))
            mean_ratio = float(
                obj_sum_ratio.get(int(oid_i), 0.0) / max(1, int(hit_count))
            )
            pref_parent = 1 if int(oid_i) == int(parent_id) else 0
            birth = int(self._id_birth_frame.get(int(oid_i), self.frame_idx))
            if split_start_fi is not None:
                rank = (
                    int(max_tail),
                    int(hit_count),
                    float(mean_ratio),
                    int(pref_parent),
                    -int(birth),
                )
            else:
                rank = (
                    int(tail_now),
                    int(max_tail),
                    int(hit_count),
                    float(mean_ratio),
                    int(pref_parent),
                    -int(birth),
                )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                matched_parent_id = int(oid_i)
                matched_parent_tail_now = int(tail_now)
                matched_parent_max_tail = int(max_tail)
                matched_parent_mean_ratio = float(mean_ratio)

        if split_start_fi is not None:
            matched_ok = bool(
                matched_parent_id is not None
                and int(matched_parent_max_tail) >= int(same_parent_req_frames)
            )
        else:
            matched_ok = bool(
                matched_parent_id is not None
                and int(matched_parent_tail_now) >= int(same_parent_req_frames)
            )
        if not matched_ok:
            return _finalize({
                "pass": False,
                "reason": "same_container_persist_failed",
                "required_frames": int(same_parent_req_frames),
                "same_parent_required_frames": int(same_parent_req_frames),
                "candidate_parent_id": int(parent_id),
                "candidate_parent_tail_now": int(obj_streak.get(int(parent_id), 0)),
                "candidate_parent_max_tail": int(obj_max_streak.get(int(parent_id), 0)),
                "candidate_parent_pass_count": int(parent_pass_count),
                "candidate_parent_max_tail_pass": int(parent_max_tail),
                "matched_parent_id": None if matched_parent_id is None else int(matched_parent_id),
                "matched_parent_tail_now": int(matched_parent_tail_now),
                "matched_parent_max_tail": int(matched_parent_max_tail),
                "matched_parent_mean_ratio": float(matched_parent_mean_ratio),
                "split_start_frame": None if split_start_frame is None else int(split_start_frame),
                "split_eval_direction": str(split_eval_direction),
                "frame_indices": [int(x) for x in eval_frame_indices],
                "rows": rows,
            }, parent_masks_dbg=parent_hint_masks, split_fwd_dbg=split_fwd, split_back_dbg=split_back)

        matched_masks: Dict[int, np.ndarray] = {}
        for fi in eval_frame_indices:
            fm = dict(tracked_by_frame.get(int(fi), {}) or {})
            pm = _to_bool_mask(fm.get(int(matched_parent_id)))
            if pm is not None and np.count_nonzero(pm) > 0:
                matched_masks[int(fi)] = pm.astype(bool).copy()

        return _finalize({
            "pass": True,
            "reason": "ok",
            "required_frames": int(same_parent_req_frames),
            "same_parent_required_frames": int(same_parent_req_frames),
            "candidate_parent_id": int(parent_id),
            "candidate_parent_tail_now": int(obj_streak.get(int(parent_id), 0)),
            "candidate_parent_max_tail": int(obj_max_streak.get(int(parent_id), 0)),
            "candidate_parent_pass_count": int(parent_pass_count),
            "candidate_parent_max_tail_pass": int(parent_max_tail),
            "matched_parent_id": int(matched_parent_id),
            "matched_parent_tail_now": int(matched_parent_tail_now),
            "matched_parent_max_tail": int(matched_parent_max_tail),
            "matched_parent_mean_ratio": float(matched_parent_mean_ratio),
            "split_start_frame": None if split_start_frame is None else int(split_start_frame),
            "split_eval_direction": str(split_eval_direction),
            "frame_indices": [int(x) for x in eval_frame_indices],
            "rows": rows,
        }, parent_masks_dbg=matched_masks if matched_masks else parent_hint_masks, split_fwd_dbg=split_fwd, split_back_dbg=split_back)

    def replay_forward_y_in_x_from_seed_mask(
        self,
        frames_bgr: List[np.ndarray],
        frame_indices: List[int],
        seed_y_mask_start: np.ndarray,
        *,
        proxy_obj_id: int,
        x_masks_by_frame: Dict[int, np.ndarray],
        max_forward: int = 15,
        count_on_th: int = 2,
    ) -> Dict[str, Any]:
        """Forward replay for attach confirmation (Y seeded, check Y in X over time)."""
        if frames_bgr is None or len(frames_bgr) == 0:
            return {"ok": False, "reason": "no_frames"}
        if frame_indices is None or len(frame_indices) != len(frames_bgr):
            return {"ok": False, "reason": "frame_indices_mismatch"}

        H, W = frames_bgr[0].shape[:2]
        seed_y_mask = _to_bool_mask(seed_y_mask_start)
        if seed_y_mask is None or np.count_nonzero(seed_y_mask) == 0:
            return {"ok": False, "reason": "invalid_seed_y_mask"}
        if seed_y_mask.shape[:2] != (H, W):
            seed_y_mask = cv2.resize(seed_y_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        K = min(len(frames_bgr), int(max(1, int(max_forward))))
        frames = frames_bgr[:K]
        indices = [int(x) for x in frame_indices[:K]]

        x_masks_clean: Dict[int, np.ndarray] = {}
        for fi, mx in dict(x_masks_by_frame or {}).items():
            try:
                fi_i = int(fi)
            except Exception:
                continue
            mm = _to_bool_mask(mx)
            if mm is None or np.count_nonzero(mm) == 0:
                continue
            x_masks_clean[int(fi_i)] = mm

        state = self.sam2_tracker.init_state(frames[0])
        self.sam2_tracker.add_new_mask(
            state,
            frame_idx=0,
            obj_id=int(proxy_obj_id),
            mask=seed_y_mask,
        )

        y_masks_by_frame: Dict[int, np.ndarray] = {int(indices[0]): seed_y_mask}
        rows: List[Dict[str, Any]] = []
        hit_frames: List[int] = []
        count_on = 0
        tail_on = 0
        max_tail_on = 0
        tail_hit_frames: List[int] = []
        confirm_frame: Optional[int] = None
        y_in_x_th = float(self.attach_mask_in_th)

        for local_idx in range(0, K):
            if local_idx > 0:
                state = self.sam2_tracker.add_frame(state, frames[local_idx])
                _, ids_now, logits_now, state = self.sam2_tracker.get_mask(state, local_idx)
                cur_mask = None
                if ids_now is not None and logits_now is not None and len(ids_now) == len(logits_now):
                    for j, oid in enumerate(ids_now):
                        try:
                            if int(oid) != int(proxy_obj_id):
                                continue
                            mm = self._logits_to_bool_mask(logits_now[j], H, W)
                            if mm is not None and np.count_nonzero(mm) > 0:
                                cur_mask = mm
                            break
                        except Exception:
                            continue
                if cur_mask is not None:
                    y_masks_by_frame[int(indices[local_idx])] = cur_mask

            fi = int(indices[local_idx])
            m_y = _to_bool_mask(y_masks_by_frame.get(int(fi)))
            m_x = _to_bool_mask(x_masks_clean.get(int(fi)))

            y_in_x = 0.0
            pass_now = False
            if (
                m_y is not None and np.count_nonzero(m_y) > 0
                and m_x is not None and np.count_nonzero(m_x) > 0
            ):
                y_in_x = float(self._mask_in_mask_ratio(m_y, m_x))
                pass_now = bool(y_in_x >= y_in_x_th)

            if pass_now:
                count_on += 1
                hit_frames.append(int(fi))
                tail_on += 1
                tail_hit_frames.append(int(fi))
                max_tail_on = max(int(max_tail_on), int(tail_on))
                if int(tail_on) >= int(max(1, int(count_on_th))) and confirm_frame is None:
                    confirm_frame = int(fi)
            else:
                tail_on = 0
                tail_hit_frames = []

            rows.append({
                "frame_idx": int(fi),
                "y_in_x": float(y_in_x),
                "pass": bool(pass_now),
                "tail_on": int(tail_on),
            })

            if confirm_frame is not None:
                return {
                    "ok": True,
                    "reason": "ok",
                    "mode": "y_in_x",
                    "proxy_id": int(proxy_obj_id),
                    "frame_indices": [int(x) for x in indices[: local_idx + 1]],
                    "proxy_masks_by_frame": {
                        int(x): m for x, m in y_masks_by_frame.items() if int(x) <= int(fi)
                    },
                    "x_masks_by_frame": {
                        int(x): m for x, m in x_masks_clean.items() if int(x) <= int(fi)
                    },
                    "window": int(max_forward),
                    "count_on": int(count_on),
                    "tail_on": int(tail_on),
                    "max_tail_on": int(max_tail_on),
                    "hit_frames": [int(x) for x in hit_frames],
                    "tail_hit_frames": [int(x) for x in tail_hit_frames],
                    "confirm_frame": int(confirm_frame),
                    "rows": rows,
                    "pass": True,
                    "early_stop": True,
                }

        return {
            "ok": True,
            "reason": "ok",
            "mode": "y_in_x",
            "proxy_id": int(proxy_obj_id),
            "frame_indices": [int(x) for x in indices],
            "proxy_masks_by_frame": y_masks_by_frame,
            "x_masks_by_frame": x_masks_clean,
            "window": int(max_forward),
            "count_on": int(count_on),
            "tail_on": int(tail_on),
            "max_tail_on": int(max_tail_on),
            "hit_frames": [int(x) for x in hit_frames],
            "tail_hit_frames": [int(x) for x in tail_hit_frames],
            "confirm_frame": None,
            "rows": rows,
            "pass": False,
            "early_stop": False,
        }
    
    def print_events_summary(self):
        """Print summary of all tracking events"""
        print(f"\n{'='*70}")
        print("TRACKING EVENTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total frames processed: {self.frame_idx}")
        print(f"Total events: {len(self.events)}")
        
        # Group events by type
        event_types = {}
        for event in self.events:
            event_type = event["type"]
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        print("\nEvent counts by type:")
        for event_type, events in sorted(event_types.items()):
            print(f"  - {event_type}: {len(events)}")
        
        # Detailed breakdown
        if "new_hand" in event_types:
            print(f"\n--- New Hands ({len(event_types['new_hand'])}) ---")
            for event in event_types["new_hand"]:
                print(f"  F{event['frame']}: hand-{event['obj_id']} (area={event['area']} px)")
        
        if "new_object" in event_types:
            print(f"\n--- New Objects ({len(event_types['new_object'])}) ---")
            for event in event_types["new_object"]:
                print(f"  F{event['frame']}: obj-{event['obj_id']-100} (area={event['area']} px, quality={event.get('quality', 0):.2f})")
        
        if "lost" in event_types:
            print(f"\n--- Lost Objects ({len(event_types['lost'])}) ---")
            for event in event_types["lost"]:
                print(f"  F{event['frame']}: obj-{event['obj_id']-100} disappeared (last_area={event['area']} px)")
        
        if "id_recovery" in event_types:
            print(f"\n--- ID Recoveries ({len(event_types['id_recovery'])}) ---")
            for event in event_types["id_recovery"]:
                print(f"  F{event['frame']}: obj-{event['temp_id']-100} -> obj-{event['recovered_id']-100} recovered")
        
        if "post_duplicate" in event_types:
            print(f"\n--- Post-Tracking Duplicates Removed ({len(event_types['post_duplicate'])}) ---")
            for event in event_types["post_duplicate"]:
                print(f"  F{event['frame']}: obj-{event['removed_id']-100} removed (duplicate of obj-{event['kept_id']-100})")
        
        print(f"\n{'='*70}\n")

    @torch.inference_mode()
    def process_frame_with_tracking(
        self,
        image_bgr: np.ndarray,
        *,
        target_contact_code: str = "P",
        iou_threshold: float = 0.5,
        frame_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process frame with hand-object tracking (EXTREME precision focus)
        
        Pipeline:
        1. Init/add_frame to SAM2
        2. Run HO detector
        3. Update existing tracked masks
        4. Process hand candidates (strict duplicate removal)
        5. Process object candidates (extreme rejection gates + pending confirmation)
        
        Returns:
            Dict with keys: frame_idx, detections, new_hands, new_objects,
            tracked_masks, tracked_boxes, all_ids, hand_ids, object_ids, events, timing
        """
        timing = {}
        self._sam2_timing_frame = {}
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.events = []
        self._struct_events_frame = []
        self._id_transitions_frame = []
        # Per-frame reset: no frozen IDs initially.
        self._freeze_dino_ema_obj_ids = set()
        reject_events: List[Dict[str, Any]] = []
        split_confirm_pass_rows: List[Dict[str, Any]] = []
        split_confirm_emitted_rows: List[Dict[str, Any]] = []
        duplicate_ioa_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        self._prune_recent_post_duplicates()
        def _collect_split_signal_masks_current_frame() -> List[np.ndarray]:
            """Return deduplicated detector-guided masks for split logic.

            Split confirmation should use all detector-derived masks from the
            current frame, not only component-duplicate-tagged masks.
            """
            out: List[np.ndarray] = []
            for src in [list(self._current_target_masks or []), list(self._current_split_target_masks or [])]:
                for m in src:
                    mb = _to_bool_mask(m)
                    if mb is None or np.count_nonzero(mb) <= 0:
                        continue
                    mb = mb.astype(bool)
                    dup = False
                    for prev in out:
                        if prev.shape == mb.shape and np.array_equal(prev, mb):
                            dup = True
                            break
                    if not dup:
                        out.append(mb.copy())
            return out

        if self.backfill_window > 0:
            self._frame_history.append(image_bgr)
            self._frame_idx_history.append(int(self.frame_idx))
        
        # 1. Inference state 초기화 또는 프레임 
        if self.inference_state is None:
            self.inference_state = self.sam2_tracker.init_state(image_bgr)
        else:
            self.inference_state = self.sam2_tracker.add_frame(self.inference_state, image_bgr)
        
        # 2. HO Detector로 detection
        t_det_start = time.time()
        detections_raw = detect_image_bgr(
            self.ho_model,
            image_bgr,
            use_cuda=self.use_cuda,
            thresh_hand=self.ho_thresh_hand,
            thresh_obj=self.ho_thresh_obj,
        )
        detections = copy.deepcopy(detections_raw)
        if bool(getattr(self, "use_hand_matched_det_boxes_only", False)):
            detections = self._suppress_non_target_matches_for_same_object(
                detections,
                target_contact_code=str(target_contact_code),
            )
        timing["ho_detection"] = time.time() - t_det_start
        # 3. Detection 결과 분리
        detected_hands = detections.get("hands", [])
        detected_objects = detections.get("objects", [])
        target_boxes: List[List[int]] = []
        self._current_target_masks = []
        self._current_split_target_masks = []
        self._current_target_points = []
        target_match_by_obj_idx: Dict[int, Dict[str, Any]] = {}
        matches_raw = detections.get("matches", [])
        if not isinstance(matches_raw, list):
            matches_raw = []
        for match in matches_raw:
            if not isinstance(match, dict):
                continue
            hand_idx = match.get("hand_idx")
            obj_idx = match.get("object_idx", -1)
            if hand_idx is None:
                continue
            try:
                hand_idx = int(hand_idx)
                obj_idx = int(obj_idx)
            except (TypeError, ValueError):
                continue
            if obj_idx < 0:
                continue
            if bool(getattr(self, "use_hand_matched_det_boxes_only", False)):
                if hand_idx < 0 or hand_idx >= len(detected_hands):
                    continue
                if obj_idx >= len(detected_objects):
                    continue
            hand: Dict[str, Any] = {}
            if 0 <= hand_idx < len(detected_hands):
                hand_row = detected_hands[hand_idx]
                if isinstance(hand_row, dict):
                    hand = hand_row

            obj_bbox = match.get("object_bbox_xyxy")
            if (not isinstance(obj_bbox, (list, tuple)) or len(obj_bbox) != 4) and 0 <= obj_idx < len(detected_objects):
                obj_row = detected_objects[obj_idx]
                if isinstance(obj_row, dict):
                    obj_bbox = obj_row.get("bbox_xyxy")
            if not isinstance(obj_bbox, (list, tuple)) or len(obj_bbox) != 4:
                continue

            hand_score_val = match.get("hand_score", 0.0)
            try:
                hand_score = float(hand_score_val) if hand_score_val is not None else 0.0
            except (TypeError, ValueError):
                hand_score = 0.0
            score_val = match.get("object_score", 0.0)
            try:
                obj_score = float(score_val) if score_val is not None else 0.0
            except (TypeError, ValueError):
                obj_score = 0.0
            prev = target_match_by_obj_idx.get(obj_idx)
            if prev is None or hand_score > float(prev.get("hand_score", -1.0)):
                target_match_by_obj_idx[obj_idx] = {
                    "hand_idx": int(hand_idx),
                    "object_idx": int(obj_idx),
                    "contact_code": hand.get("contact_code"),
                    "contact_text": hand.get("contact_text"),
                    "hand_score": float(hand_score),
                    "hand_lr": hand.get("lr"),
                    "object_score": float(obj_score),
                    "object_bbox_xyxy": [int(v) for v in obj_bbox],
                }

        target_matches: List[Dict[str, Any]] = []
        selected_target_idxs: Set[int] = set()
        selected_target_hands: Set[int] = set()
        for obj_idx in sorted(target_match_by_obj_idx.keys()):
            entry = target_match_by_obj_idx[int(obj_idx)]
            target_matches.append(entry)
            selected_target_idxs.add(int(obj_idx))
            selected_target_hands.add(int(entry.get("hand_idx", -1)))
            obj_bbox = entry.get("object_bbox_xyxy")
            if obj_bbox is not None:
                target_boxes.append([int(v) for v in obj_bbox])
        if (
            bool(getattr(self, "use_all_object_boxes", False))
            and not bool(getattr(self, "use_hand_matched_det_boxes_only", False))
        ):
            for obj_idx, obj_row in enumerate(detected_objects):
                if int(obj_idx) in selected_target_idxs:
                    continue
                if not isinstance(obj_row, dict):
                    continue
                obj_bbox = obj_row.get("bbox_xyxy")
                if not isinstance(obj_bbox, (list, tuple)) or len(obj_bbox) != 4:
                    continue
                score_val = obj_row.get("score", 0.0)
                try:
                    obj_score = float(score_val) if score_val is not None else 0.0
                except (TypeError, ValueError):
                    obj_score = 0.0
                if obj_score < float(self.ho_thresh_obj):
                    continue
                obj_bbox_i = [int(v) for v in obj_bbox]
                target_boxes.append(obj_bbox_i)
                selected_target_idxs.add(int(obj_idx))
                target_matches.append({
                    "hand_idx": -1,
                    "object_idx": int(obj_idx),
                    "contact_code": None,
                    "contact_text": "unmatched_object",
                    "hand_score": 0.0,
                    "hand_lr": None,
                    "object_score": float(obj_score),
                    "object_bbox_xyxy": obj_bbox_i,
                    "source": "all_object_boxes",
                })
        # Global target-box size gate:
        # oversize detector boxes are removed before any downstream stage
        # (object birth/split/attach/merge) so they cannot affect decisions.
        target_matches_filtered: List[Dict[str, Any]] = []
        target_boxes_filtered: List[List[int]] = []
        suppressed_large_target_matches: List[Dict[str, Any]] = []
        frame_area = float(max(1, int(height) * int(width)))
        hand_box_areas: List[float] = []
        for hrow in list(detected_hands or []):
            if not isinstance(hrow, dict):
                continue
            hb = hrow.get("bbox_xyxy")
            if not isinstance(hb, (list, tuple)) or len(hb) != 4:
                continue
            try:
                hx1, hy1, hx2, hy2 = [int(v) for v in list(hb)[:4]]
            except Exception:
                continue
            harea = float(max(0, hx2 - hx1) * max(0, hy2 - hy1))
            if harea > 0.0:
                hand_box_areas.append(float(harea))
        hand_area_ref_det = float(max(hand_box_areas)) if hand_box_areas else 0.0

        for entry in list(target_matches or []):
            if not isinstance(entry, dict):
                continue
            box = entry.get("object_bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in list(box)[:4]]
            except Exception:
                continue
            box_i = [int(x1), int(y1), int(x2), int(y2)]
            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            ratio_img = float(box_area / frame_area)
            ratio_hand = (
                float(box_area / max(1e-6, hand_area_ref_det))
                if hand_area_ref_det > 0.0
                else 0.0
            )

            reject_reason = None
            if ratio_img > float(self.max_obj_area_ratio):
                reject_reason = "bbox_ratio_global"
            elif hand_area_ref_det > 0.0 and ratio_hand > float(self.max_obj_vs_hand_ratio):
                reject_reason = "bbox_vs_hand_ratio_global"

            if reject_reason is not None:
                suppressed_large_target_matches.append({
                    "reason": str(reject_reason),
                    "box": [int(v) for v in box_i],
                    "bbox_area_ratio": float(ratio_img),
                    "bbox_vs_hand_ratio": float(ratio_hand) if hand_area_ref_det > 0.0 else None,
                    "object_idx": int(entry.get("object_idx", -1)) if entry.get("object_idx") is not None else -1,
                    "hand_idx": int(entry.get("hand_idx", -1)) if entry.get("hand_idx") is not None else -1,
                })
                reject_events.append({
                    "kind": "object",
                    "stage": "target_box_global_gate",
                    "reason": str(reject_reason),
                    "box": [int(v) for v in box_i],
                    "bbox_area_ratio": float(ratio_img),
                    "bbox_vs_hand_ratio": float(ratio_hand) if hand_area_ref_det > 0.0 else None,
                })
                continue

            target_matches_filtered.append(entry)
            target_boxes_filtered.append([int(v) for v in box_i])

        target_matches = target_matches_filtered
        target_boxes = target_boxes_filtered
        selected_target_idxs = {
            int(row.get("object_idx"))
            for row in list(target_matches or [])
            if isinstance(row, dict) and row.get("object_idx") is not None
        }
        selected_target_hands = {
            int(row.get("hand_idx"))
            for row in list(target_matches or [])
            if isinstance(row, dict) and row.get("hand_idx") is not None and int(row.get("hand_idx")) >= 0
        }
        detections["target_contact_code"] = str(target_contact_code)
        detections["target_object_idxs"] = sorted([int(x) for x in selected_target_idxs])
        detections["target_hand_idxs"] = sorted([int(x) for x in selected_target_hands if int(x) >= 0])
        detections["target_matches"] = target_matches
        detections["suppressed_large_target_matches"] = int(len(suppressed_large_target_matches))
        detections["suppressed_large_target_rows"] = suppressed_large_target_matches

        current_hand_boxes = [hand["bbox_xyxy"] for hand in detected_hands]
        current_hand_proxy_masks: List[np.ndarray] = []
        if not bool(getattr(self, "track_hand_masks", True)):
            for hb in list(current_hand_boxes or []):
                if not isinstance(hb, (list, tuple)) or len(hb) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in list(hb)[:4]]
                except Exception:
                    continue
                x1 = max(0, min(int(width - 1), x1))
                x2 = max(0, min(int(width), x2))
                y1 = max(0, min(int(height - 1), y1))
                y2 = max(0, min(int(height), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                hm = np.zeros((height, width), dtype=bool)
                hm[y1:y2, x1:x2] = True
                current_hand_proxy_masks.append(hm)
        current_object_boxes = target_boxes
        self._current_target_boxes = target_boxes
        self._prune_ambiguous_regions()
        
        # 4. 기존 tracking 객체들의 마스크 업데이트
        t_track_start = time.time()
        tracked_masks: Dict[int, np.ndarray] = {}
        tracked_boxes: Dict[int, List[int]] = {}
        
        # Remember previous object state for lost object detection
        prev_object_ids = [i for i in self.all_ids if i >= 100]
        prev_object_masks = {}
        lost_reason_by_id: Dict[int, str] = {}
        
        if len(self.all_ids) > 0:
            _, self.all_ids, out_mask_logits, self.inference_state = self.sam2_tracker.get_mask(
                self.inference_state, self.frame_idx
            )
            ids_from_logits = [int(x) for x in list(self.all_ids)]
            if out_mask_logits is not None:
                print(f"[Hotrack] F{self.frame_idx} MASK_AREAS:")
                
                for idx, obj_id in enumerate(ids_from_logits):
                    if idx < len(out_mask_logits):
                        if int(obj_id) in self._retired_struct_ids:
                            try:
                                self.all_ids, _ = self.sam2_tracker.remove_object(
                                    self.inference_state, int(obj_id), strict=False, need_output=False
                                )
                            except Exception:
                                pass
                            if int(obj_id) in self.all_ids:
                                self.all_ids = [int(x) for x in self.all_ids if int(x) != int(obj_id)]
                            continue
                        mask = (out_mask_logits[idx] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze()
                        if mask.shape[0] != height or mask.shape[1] != width:
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        mask_bool_original = mask.astype(bool)
                        prev_hist_mask: Optional[np.ndarray] = None
                        if obj_id >= 100 and obj_id in self.mask_history and len(self.mask_history[obj_id]) > 0:
                            prev_hist_mask = self.mask_history[obj_id][-1][1]
                        # Check mask area before filtering
                        mask_area_before = np.sum(mask_bool_original)
                        
                        # Keep component that best matches historical masks (hands only)
                        if obj_id < 100:
                            mask_bool_filtered = self._keep_best_matching_component(mask_bool_original, obj_id)
                        else:
                            mask_bool_filtered = mask_bool_original
                        mask_area_after = np.sum(mask_bool_filtered)
                        # If filtering changed the mask, update SAM2 state
                        if not np.array_equal(mask_bool_original, mask_bool_filtered):
                            obj_type = "hand" if obj_id < 100 else "obj"
                            obj_num = obj_id if obj_id < 100 else obj_id - 100
                            delta = int(mask_area_after - mask_area_before)
                            delta_desc = f"+{delta}" if delta >= 0 else f"{delta}"
                            print(
                                f"[Hotrack] F{self.frame_idx} UPDATING SAM2 STATE: {obj_type}-{obj_num} "
                                f"(area {mask_area_before}->{mask_area_after}, delta {delta_desc} px)"
                            )
                            
                            # Update SAM2 state with filtered mask directly.
                            self.sam2_tracker.add_new_mask(
                                self.inference_state,
                                frame_idx=self.frame_idx,
                                obj_id=obj_id,
                                mask=mask_bool_filtered,
                            )
                        
                        # Update mask history
                        if obj_id not in self.mask_history:
                            self.mask_history[obj_id] = deque(maxlen=self.mask_history_length)
                        self.mask_history[obj_id].append((self.frame_idx, mask_bool_filtered.copy()))
                        
                        # Log mask area
                        obj_type = "hand" if obj_id < 100 else "obj"
                        obj_num = obj_id if obj_id < 100 else obj_id - 100
                        history_len = len(self.mask_history[obj_id])
                        hist_desc = f"hist={history_len}F" if history_len > 1 else "hist=prev"
                        print(f"  {obj_type}-{obj_num}: area={mask_area_after} px (before={mask_area_before}, {hist_desc})")

                        # Soft LOST registration for tiny/invisible objects:
                        # keep tracking alive (no remove), but allow ID recovery to reuse this ID
                        # when a better mask appears.
                        if obj_id >= 100:
                            if int(mask_area_after) < int(self.min_mask_area):
                                reason = "invisible_mask" if int(mask_area_after) <= 0 else "small_mask"
                                lost_reason_by_id[int(obj_id)] = str(reason)
                                seed_mask = None
                                if prev_hist_mask is not None and np.count_nonzero(prev_hist_mask) >= self.min_mask_area:
                                    seed_mask = prev_hist_mask.astype(bool).copy()
                                elif np.count_nonzero(mask_bool_filtered) > 0:
                                    seed_mask = mask_bool_filtered.astype(bool).copy()
                                if seed_mask is not None:
                                    self._register_confusion_object(
                                        int(obj_id),
                                        seed_mask,
                                        reason=f"soft_{reason}",
                                        frame_bgr=image_bgr,
                                    )
                                    print(
                                        f"[Hotrack] F{self.frame_idx} EVENT_CONFUSION_SOFT: obj-{obj_id-100} "
                                        f"reason={reason} (area={int(mask_area_after)} < {self.min_mask_area})"
                                    )
                            else:
                                # Object is healthy again; drop stale confusion entry for this active ID.
                                if int(obj_id) in self.confusion_objects_history:
                                    self.confusion_objects_history.pop(int(obj_id), None)

                        tracked_masks[obj_id] = mask_bool_filtered
                        tracked_boxes[obj_id] = self._get_mask_bbox(mask_bool_filtered)
                        
                        # Save object masks for lost detection
                        if obj_id >= 100:
                            prev_object_masks[obj_id] = mask_bool_filtered
            if self._retired_struct_ids:
                self.all_ids = [int(x) for x in self.all_ids if int(x) not in self._retired_struct_ids]
        
        hand_ids = [i for i in self.all_ids if i < 100]
        object_ids = [i for i in self.all_ids if i >= 100]
        if not bool(getattr(self, "track_hand_masks", True)) and hand_ids:
            for hid in list(hand_ids):
                try:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, int(hid), strict=False, need_output=False
                    )
                except Exception:
                    pass
                tracked_masks.pop(int(hid), None)
                tracked_boxes.pop(int(hid), None)
                self.mask_history.pop(int(hid), None)
            hand_ids = []
            self.all_ids = [int(x) for x in self.all_ids if int(x) >= 100]

        # Remove hands that are included in another hand (subset)
        if len(hand_ids) >= 2:
            hand_remove = set()
            checked_pairs = set()
            for i, hid_a in enumerate(hand_ids):
                for hid_b in hand_ids[i + 1:]:
                    pair_key = tuple(sorted([hid_a, hid_b]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    mask_a = tracked_masks.get(hid_a)
                    mask_b = tracked_masks.get(hid_b)
                    if mask_a is None or mask_b is None:
                        continue
                    _, ioa_a_to_b, ioa_b_to_a = calculate_ioa_bidirectional(mask_a, mask_b)
                    area_a = int(np.sum(mask_a))
                    area_b = int(np.sum(mask_b))
                    if ioa_a_to_b >= self.hand_inclusion_ioa_threshold and ioa_b_to_a >= self.hand_inclusion_ioa_threshold:
                        remove_id = hid_a if area_a <= area_b else hid_b
                        hand_remove.add(remove_id)
                        print(
                            f"[Hotrack] F{self.frame_idx} HAND_INCLUSION_BOTH: "
                            f"hand-{hid_a} area={area_a} ioa={ioa_a_to_b:.2f} vs "
                            f"hand-{hid_b} area={area_b} ioa={ioa_b_to_a:.2f} -> remove hand-{remove_id}"
                        )
                    elif ioa_a_to_b >= self.hand_inclusion_ioa_threshold:
                        remove_id = hid_a if area_a <= area_b else hid_b
                        hand_remove.add(remove_id)
                        print(
                            f"[Hotrack] F{self.frame_idx} HAND_INCLUSION: "
                            f"hand-{hid_a} area={area_a} in hand-{hid_b} area={area_b} "
                            f"ioa={ioa_a_to_b:.2f} -> remove hand-{remove_id}"
                        )
                    elif ioa_b_to_a >= self.hand_inclusion_ioa_threshold:
                        remove_id = hid_b if area_b <= area_a else hid_a
                        hand_remove.add(remove_id)
                        print(
                            f"[Hotrack] F{self.frame_idx} HAND_INCLUSION: "
                            f"hand-{hid_b} area={area_b} in hand-{hid_a} area={area_a} "
                            f"ioa={ioa_b_to_a:.2f} -> remove hand-{remove_id}"
                        )
            for hid in hand_remove:
                self.all_ids, _ = self.sam2_tracker.remove_object(
                    self.inference_state, hid, strict=False, need_output=False
                )
                tracked_masks.pop(hid, None)
                tracked_boxes.pop(hid, None)
                self.mask_history.pop(hid, None)
                print(f"[Hotrack] F{self.frame_idx} EVENT_HAND_INCLUSION_REMOVE: hand-{hid}")
                self.events.append({
                    "type": "hand_inclusion_remove",
                    "frame": self.frame_idx,
                    "obj_id": int(hid),
                })
            if hand_remove:
                hand_ids = [i for i in self.all_ids if i < 100]
        
        # Collect overlap information for all object pairs (for graph analysis)
        overlap_info = []
        
        # Post-tracking duplicate check with temporal confirmation.
        # Only IDs in duplicate-overlap pairs are frozen for EMA updates.
        if bool(self.enable_post_track_duplicate_removal) and len(object_ids) >= 2:
            checked_pairs = set()
            active_dup_pairs: Set[Tuple[int, int]] = set()
            removed_object_ids: Set[int] = set()
            # Cache current-frame DINO embeds per object for duplicate keep/remove decisions.
            dup_cur_dino_embed: Dict[int, Optional[np.ndarray]] = {}
            
            for i, obj_id_a in enumerate(object_ids):
                for obj_id_b in object_ids[i+1:]:
                    if int(obj_id_a) in removed_object_ids or int(obj_id_b) in removed_object_ids:
                        continue
                    # Skip if already checked this pair
                    pair_key = tuple(sorted([obj_id_a, obj_id_b]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    
                    # Get masks
                    mask_a = tracked_masks.get(obj_id_a)
                    mask_b = tracked_masks.get(obj_id_b)
                    
                    if mask_a is None or mask_b is None:
                        continue
                    
                    # Check IoA in both directions
                    iou, ioa_a_to_b, ioa_b_to_a = calculate_ioa_bidirectional(mask_a, mask_b)
                    
                    # Store overlap info for graph analysis
                    overlap_info.append({
                        "obj_a": obj_id_a,
                        "obj_b": obj_id_b,
                        "iou": float(iou),
                        "ioa_a_to_b": float(ioa_a_to_b),
                        "ioa_b_to_a": float(ioa_b_to_a),
                    })
                    
                    # Significant overlap detected; confirm over consecutive frames, then prune.
                    if ioa_a_to_b >= self.post_track_dup_ioa_threshold and ioa_b_to_a >= self.post_track_dup_ioa_threshold:
                        self._freeze_dino_ema_obj_ids.add(int(obj_id_a))
                        self._freeze_dino_ema_obj_ids.add(int(obj_id_b))
                        pair_key_i = (int(pair_key[0]), int(pair_key[1]))
                        active_dup_pairs.add(pair_key_i)
                        streak = int(self.post_dup_history.get(pair_key_i, 0)) + 1
                        self.post_dup_history[pair_key_i] = int(streak)
                        confirm_frames = 1 if bool(self.post_dup_immediate_compare) else int(max(1, self.post_dup_confirm_frames))
                        birth_a = self._id_birth_frame.get(int(obj_id_a))
                        birth_b = self._id_birth_frame.get(int(obj_id_b))
                        age_a = None if birth_a is None else int(self.frame_idx - int(birth_a))
                        age_b = None if birth_b is None else int(self.frame_idx - int(birth_b))
                        newborn_age_th = int(max(0, int(self.post_dup_newborn_max_age_frames)))
                        is_newborn_a = bool(age_a is not None and int(age_a) <= newborn_age_th)
                        is_newborn_b = bool(age_b is not None and int(age_b) <= newborn_age_th)
                        newborn_immediate = bool(
                            bool(self.post_dup_immediate_if_newborn)
                            and (bool(is_newborn_a) or bool(is_newborn_b))
                        )
                        if newborn_immediate:
                            confirm_frames = 1

                        # Lock keep/remove decision at overlap-start frame so that
                        # later removal uses start-time appearance evidence.
                        pair_lock = self.post_dup_locked_decisions.get(pair_key_i)
                        if pair_lock is None:
                            pair_lock = self._decide_post_duplicate_keep_remove(
                                image_bgr=image_bgr,
                                obj_id_a=int(obj_id_a),
                                obj_id_b=int(obj_id_b),
                                mask_a=mask_a,
                                mask_b=mask_b,
                                dino_cache=dup_cur_dino_embed,
                            )
                            pair_lock["start_frame"] = int(self.frame_idx)
                            pair_lock["start_streak"] = int(streak)
                            self.post_dup_locked_decisions[pair_key_i] = pair_lock
                            print(
                                f"[Hotrack] F{self.frame_idx} POST_DUP_LOCK: "
                                f"pair=({int(obj_id_a)-100},{int(obj_id_b)-100}) "
                                f"keep=obj-{int(pair_lock['keep_id'])-100} "
                                f"drop=obj-{int(pair_lock['remove_id'])-100} "
                                f"(basis={pair_lock.get('decision_basis')}, "
                                f"dino={pair_lock.get('dino_self_sim_a') if pair_lock.get('dino_self_sim_a') is not None else -1.0:.3f}/"
                                f"{pair_lock.get('dino_self_sim_b') if pair_lock.get('dino_self_sim_b') is not None else -1.0:.3f})"
                            )
                        pair_lock["newborn_immediate"] = bool(newborn_immediate)
                        pair_lock["newborn_a"] = bool(is_newborn_a)
                        pair_lock["newborn_b"] = bool(is_newborn_b)
                        pair_lock["newborn_age_a"] = None if age_a is None else int(age_a)
                        pair_lock["newborn_age_b"] = None if age_b is None else int(age_b)
                        pair_lock["birth_frame_a"] = None if birth_a is None else int(birth_a)
                        pair_lock["birth_frame_b"] = None if birth_b is None else int(birth_b)

                        if newborn_immediate:
                            forced_keep_id: Optional[int] = None
                            forced_remove_id: Optional[int] = None
                            if is_newborn_a and not is_newborn_b:
                                forced_keep_id, forced_remove_id = int(obj_id_b), int(obj_id_a)
                            elif is_newborn_b and not is_newborn_a:
                                forced_keep_id, forced_remove_id = int(obj_id_a), int(obj_id_b)
                            elif is_newborn_a and is_newborn_b:
                                if birth_a is not None and birth_b is not None and int(birth_a) != int(birth_b):
                                    if int(birth_a) < int(birth_b):
                                        forced_keep_id, forced_remove_id = int(obj_id_a), int(obj_id_b)
                                    else:
                                        forced_keep_id, forced_remove_id = int(obj_id_b), int(obj_id_a)
                            if forced_keep_id is not None and forced_remove_id is not None:
                                pair_lock["keep_id"] = int(forced_keep_id)
                                pair_lock["remove_id"] = int(forced_remove_id)
                                pair_lock["decision_basis"] = "newborn_immediate"
                                print(
                                    f"[Hotrack] F{self.frame_idx} POST_DUP_NEWBORN_OVERRIDE: "
                                    f"pair=({int(obj_id_a)-100},{int(obj_id_b)-100}) "
                                    f"keep=obj-{int(forced_keep_id)-100} drop=obj-{int(forced_remove_id)-100} "
                                    f"(age={-1 if age_a is None else int(age_a)}/{-1 if age_b is None else int(age_b)}, "
                                    f"confirm={int(confirm_frames)})"
                                )

                        print(
                            f"[Hotrack] F{self.frame_idx} POST_DUP_OVERLAP: "
                            f"obj-{int(obj_id_a)-100} vs obj-{int(obj_id_b)-100} "
                            f"(IoA: {ioa_a_to_b:.2f}/{ioa_b_to_a:.2f}) streak={streak} "
                            f"confirm={int(confirm_frames)}"
                        )
                        if streak < int(confirm_frames):
                            self.events.append({
                                "type": "post_duplicate_overlap",
                                "frame": int(self.frame_idx),
                                "obj_a": int(obj_id_a),
                                "obj_b": int(obj_id_b),
                                "ioa_a_to_b": float(ioa_a_to_b),
                                "ioa_b_to_a": float(ioa_b_to_a),
                                "streak": int(streak),
                                "locked_keep_id": int(pair_lock.get("keep_id")),
                                "locked_remove_id": int(pair_lock.get("remove_id")),
                                "locked_start_frame": int(pair_lock.get("start_frame", self.frame_idx)),
                                "confirm_frames": int(confirm_frames),
                                "newborn_immediate": bool(newborn_immediate),
                                "newborn_a": bool(is_newborn_a),
                                "newborn_b": bool(is_newborn_b),
                                "newborn_age_a": None if age_a is None else int(age_a),
                                "newborn_age_b": None if age_b is None else int(age_b),
                                "removed": False,
                            })
                            continue

                        keep_id = int(pair_lock.get("keep_id", int(obj_id_a)))
                        remove_id = int(pair_lock.get("remove_id", int(obj_id_b)))
                        if int(remove_id) in removed_object_ids:
                            continue
                        remove_mask = tracked_masks.get(int(remove_id))
                        if remove_mask is None or np.count_nonzero(remove_mask) == 0:
                            continue
                        if int(keep_id) not in tracked_masks:
                            continue

                        dino_sim_a = pair_lock.get("dino_self_sim_a")
                        dino_sim_b = pair_lock.get("dino_self_sim_b")
                        stab_a = float(pair_lock.get("stability_a", 0.0))
                        stab_b = float(pair_lock.get("stability_b", 0.0))
                        decision_basis = f"locked_{str(pair_lock.get('decision_basis', 'temporal_stability'))}"
                        start_frame = int(pair_lock.get("start_frame", self.frame_idx))

                        removed_object_ids.add(int(remove_id))
                        self._register_recent_post_duplicate(int(remove_id), int(keep_id), remove_mask)
                        self._register_confusion_object(
                            int(remove_id),
                            remove_mask,
                            reason=f"post_duplicate_with_{int(keep_id)}",
                            frame_bgr=image_bgr,
                        )
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, int(remove_id), strict=False, need_output=False
                        )
                        tracked_masks.pop(int(remove_id), None)
                        tracked_boxes.pop(int(remove_id), None)
                        self.mask_history.pop(int(remove_id), None)
                        lost_reason_by_id[int(remove_id)] = "post_duplicate"

                        if int(remove_id) in object_ids:
                            # Keep local list coherent for downstream passes in this frame.
                            object_ids = [int(x) for x in object_ids if int(x) != int(remove_id)]
                        self.post_dup_history.pop(pair_key_i, None)
                        self.post_dup_locked_decisions.pop(pair_key_i, None)
                        active_dup_pairs.discard(pair_key_i)

                        print(
                            f"[Hotrack] F{self.frame_idx} EVENT_POST_DUPLICATE: "
                            f"remove obj-{int(remove_id)-100}, keep obj-{int(keep_id)-100} "
                            f"(streak={streak}, start={start_frame}, basis={decision_basis}, "
                            f"dino={dino_sim_a if dino_sim_a is not None else -1.0:.3f}/"
                            f"{dino_sim_b if dino_sim_b is not None else -1.0:.3f}, "
                            f"stab={stab_a:.3f}/{stab_b:.3f})"
                        )
                        self.events.append({
                            "type": "post_duplicate",
                            "frame": int(self.frame_idx),
                            "removed_id": int(remove_id),
                            "kept_id": int(keep_id),
                            "streak": int(streak),
                            "ioa_a_to_b": float(ioa_a_to_b),
                            "ioa_b_to_a": float(ioa_b_to_a),
                            "stability_a": float(stab_a),
                            "stability_b": float(stab_b),
                            "dino_self_sim_a": None if dino_sim_a is None else float(dino_sim_a),
                            "dino_self_sim_b": None if dino_sim_b is None else float(dino_sim_b),
                            "decision_basis": str(decision_basis),
                            "locked_start_frame": int(start_frame),
                            "confirm_frames": int(confirm_frames),
                            "newborn_immediate": bool(pair_lock.get("newborn_immediate", False)),
                            "newborn_a": bool(pair_lock.get("newborn_a", False)),
                            "newborn_b": bool(pair_lock.get("newborn_b", False)),
                            "newborn_age_a": pair_lock.get("newborn_age_a"),
                            "newborn_age_b": pair_lock.get("newborn_age_b"),
                            "removed": True,
                        })

            # Decay pair streaks when overlap is not active in current frame.
            stale_pairs = [k for k in self.post_dup_history.keys() if k not in active_dup_pairs]
            for k in stale_pairs:
                self.post_dup_history.pop(k, None)
                self.post_dup_locked_decisions.pop(k, None)
        else:
            # Keep internal lock/history clean when post-duplicate pass is off.
            self.post_dup_history = {}
            self.post_dup_locked_decisions = {}

        # Update lost objects history
        prev_frame_bgr = None
        if self.backfill_window > 1 and len(self._frame_history) >= 2:
            prev_frame_bgr = self._frame_history[-2]
        self._update_confusion_objects_history(
            prev_object_ids,
            object_ids,
            prev_object_masks,
            prev_frame_bgr,
            lost_reason_by_id=lost_reason_by_id,
        )
        
        timing["sam2_tracking"] = time.time() - t_track_start
        
        # 5. Hand Tracking 처리 (strict duplicate removal)
        t_hand_start = time.time()
        new_hands: List[Dict[str, Any]] = []
        hand_sam2_input_debug: List[Dict[str, Any]] = []
        
        if bool(getattr(self, "track_hand_masks", True)) and len(detected_hands) > len(hand_ids):
            hand_candidates = []  # (hand_info, bbox, temp_id, mask)
            
            for hand in detected_hands:
                hand_bbox = hand["bbox_xyxy"]
                box_np = np.array(hand_bbox, dtype=np.float32)
                temp_hand_id = self.next_hand_id
                self.next_hand_id += 1
                
                _, self.all_ids, temp_mask_logits, self.inference_state = self.sam2_tracker.add_new_points_or_box(
                    self.inference_state,
                    frame_idx=self.frame_idx,
                    obj_id=temp_hand_id,
                    box=box_np,
                )
                
                if temp_mask_logits is None or len(temp_mask_logits) == 0:
                    continue
                
                new_mask = (temp_mask_logits[-1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze()
                if new_mask.shape[0] != height or new_mask.shape[1] != width:
                    new_mask = cv2.resize(new_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                new_mask = new_mask.astype(bool)

                centers: List[Tuple[int, int]] = []
                # Refine hand mask using uniformly distributed in-mask points.
                if bool(self.enable_prompt_refinement):
                    centers = self._mask_uniform_points(
                        new_mask,
                        max_points=self.dt_point_count,
                        min_dist=self.dt_point_min_dist,
                    )
                    if not centers:
                        centers = self._mask_center_points_dt(
                            new_mask,
                            max_points=self.dt_point_count,
                            min_dist=self.dt_point_min_dist,
                        )
                    if centers:
                        try:
                            pts = np.array(centers, dtype=np.float32)
                            labels = np.ones(len(centers), dtype=np.int32)
                            _, self.all_ids, pt_mask_logits, self.inference_state = self.sam2_tracker.add_new_points_or_box(
                                self.inference_state,
                                frame_idx=self.frame_idx,
                                obj_id=temp_hand_id,
                                points=pts,
                                labels=labels,
                                clear_old_points=True,
                            )
                            if pt_mask_logits is not None and len(pt_mask_logits) > 0:
                                pt_mask = (pt_mask_logits[-1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze()
                                if pt_mask.shape[0] != height or pt_mask.shape[1] != width:
                                    pt_mask = cv2.resize(pt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                if np.count_nonzero(pt_mask) > 0:
                                    new_mask = pt_mask.astype(bool)
                        except Exception:
                            pass
                hand_sam2_input_debug.append({
                    "box": [int(v) for v in hand_bbox] if hand_bbox is not None else None,
                    "points": [list(p) for p in centers],
                    "temp_id": int(temp_hand_id),
                })

                new_mask = self._keep_best_matching_component(new_mask, temp_hand_id)
                
                hand_candidates.append((hand, hand_bbox, temp_hand_id, new_mask))
            
            # Strict duplicate removal for hands
            accepted_hand_masks = {}
            for hand, hand_bbox, hand_id, new_mask in hand_candidates:
                ambiguous = False
                for prev_box in self._prev_object_boxes:
                    if bbox_iou(hand_bbox, prev_box) >= self.ambiguous_bbox_iou_threshold:
                        ambiguous = True
                        break
                if not ambiguous and object_ids:
                    for obj_id in object_ids:
                        obj_mask = tracked_masks.get(obj_id)
                        if obj_mask is None:
                            continue
                        _, ioa_hand_to_obj, ioa_obj_to_hand = calculate_ioa_bidirectional(new_mask, obj_mask)
                        if max(ioa_hand_to_obj, ioa_obj_to_hand) >= self.ambiguous_mask_ioa_threshold:
                            ambiguous = True
                            break
                if ambiguous:
                    decision = self._update_ambiguous_region(hand_bbox, "hand")
                    if decision != "hand":
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, hand_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "hand",
                            "stage": "ambiguous_gate",
                            "reason": "ambiguous_region",
                            "box": [int(v) for v in hand_bbox] if hand_bbox is not None else None,
                            "temp_id": int(hand_id),
                        })
                        continue

                # Check against already tracked hands
                is_dup = False
                for tracked_id in hand_ids:
                    if tracked_id in tracked_masks:
                        if is_duplicate_mask(new_mask, tracked_masks[tracked_id], ioa_threshold=0.5):
                            iou, ioa_new_to_exist, ioa_exist_to_new = calculate_ioa_bidirectional(
                                new_mask, tracked_masks[tracked_id]
                            )
                            print(
                                f"[Hotrack] F{self.frame_idx} HAND_DUP: hand-{hand_id} vs hand-{tracked_id} "
                                f"IoU={iou:.2f} IoA(new->exist)={ioa_new_to_exist:.2f} "
                                f"IoA(exist->new)={ioa_exist_to_new:.2f}"
                            )
                            is_dup = True
                            break

                # Reject if new hand overlaps too much with existing hands
                if not is_dup:
                    max_ioa = 0.0
                    for tracked_id in hand_ids:
                        if tracked_id in tracked_masks:
                            _, ioa_e2n, ioa_n2e = calculate_ioa_bidirectional(tracked_masks[tracked_id], new_mask)
                            max_ioa = max(max_ioa, ioa_e2n, ioa_n2e)
                    if max_ioa >= self.new_mask_reject_ioa:
                        is_dup = True
                        print(f"[Hotrack] F{self.frame_idx} HAND_REJECT: hand-{hand_id} high_ioa_{max_ioa:.2f}")
                        reject_events.append({
                            "kind": "hand",
                            "stage": "overlap_gate",
                            "reason": "high_ioa",
                            "value": float(max_ioa),
                            "box": [int(v) for v in hand_bbox] if hand_bbox is not None else None,
                            "temp_id": int(hand_id),
                        })
                
                # Check against already accepted candidates in this frame
                if not is_dup:
                    for acc_mask in accepted_hand_masks.values():
                        if is_duplicate_mask(new_mask, acc_mask, ioa_threshold=0.5):
                            iou, ioa_new_to_acc, ioa_acc_to_new = calculate_ioa_bidirectional(new_mask, acc_mask)
                            print(
                                f"[Hotrack] F{self.frame_idx} HAND_DUP: hand-{hand_id} vs new_hand "
                                f"IoU={iou:.2f} IoA(new->acc)={ioa_new_to_acc:.2f} "
                                f"IoA(acc->new)={ioa_acc_to_new:.2f}"
                            )
                            is_dup = True
                            break
                
                if is_dup:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, hand_id, strict=False, need_output=False
                    )
                    reject_events.append({
                        "kind": "hand",
                        "stage": "duplicate_gate",
                        "reason": "duplicate",
                        "box": [int(v) for v in hand_bbox] if hand_bbox is not None else None,
                        "temp_id": int(hand_id),
                    })
                    print(f"[Hotrack] F{self.frame_idx} EVENT_REJECT: hand-{hand_id} duplicate")
                else:
                    tracked_masks[hand_id] = new_mask
                    tracked_boxes[hand_id] = self._get_mask_bbox(new_mask)
                    accepted_hand_masks[hand_id] = new_mask
                    new_hands.append({"obj_id": hand_id, "bbox": hand_bbox, "hand_info": hand})
                    area = np.sum(new_mask)
                    print(f"[Hotrack] F{self.frame_idx} EVENT_NEW: hand-{hand_id} added (area={area} px)")
                    self.events.append({
                        "type": "new_hand",
                        "frame": self.frame_idx,
                        "obj_id": hand_id,
                        "area": int(area)
                    })
        
        timing["hand_processing"] = time.time() - t_hand_start
        
        # 6. Object Tracking 처리 (EXTREME rejection gates + pending confirmation)
        t_obj_start = time.time()
        new_objects: List[Dict[str, Any]] = []
        
        # Get current hand masks for gating
        if bool(getattr(self, "track_hand_masks", True)):
            hand_masks_list = [tracked_masks[hid] for hid in hand_ids if hid in tracked_masks]
        else:
            hand_masks_list = list(current_hand_proxy_masks or [])
        hand_area_ref = 0.0
        if hand_masks_list:
            hand_areas = [float(np.sum(m)) for m in hand_masks_list if m is not None]
            if hand_areas:
                hand_area_ref = max(hand_areas)
        
        # Get current object masks/boxes for duplicate checking
        object_masks = {oid: tracked_masks[oid] for oid in object_ids if oid in tracked_masks}
        object_boxes = {oid: tracked_boxes[oid] for oid in object_ids if oid in tracked_boxes}
        candidate_log_rows: List[Dict[str, Any]] = []
        raw_object_candidates: List[Dict[str, Any]] = []
        raw_object_candidate_by_temp: Dict[int, Dict[str, Any]] = {}

        sam2_input_debug: List[Dict[str, Any]] = []
        if len(target_boxes) > 0:
            # Generate candidates
            candidates: List[Dict[str, Any]] = []
            
            for box in target_boxes:
                # Skip segmentation/tracking if detection bbox is too large (likely background)
                if box is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    img_area = float(height * width)
                    box_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
                    if hand_area_ref > 0.0:
                        ratio_hand = box_area / hand_area_ref
                        if ratio_hand > self.max_obj_vs_hand_ratio:
                            reject_events.append({
                                "kind": "object",
                                "stage": "bbox_gate",
                                "reason": "bbox_vs_hand_ratio",
                                "value": float(ratio_hand),
                                "box": [int(v) for v in box],
                            })
                            print(
                                f"[Hotrack] F{self.frame_idx} OBJ_SKIP: det_box "
                                f"reason=bbox_vs_hand_ratio_{ratio_hand:.2f}"
                            )
                            continue
                    if img_area > 0:
                        ratio_img = box_area / img_area
                        if ratio_img > self.max_obj_area_ratio:
                            reject_events.append({
                                "kind": "object",
                                "stage": "bbox_gate",
                                "reason": "bbox_ratio",
                                "value": float(ratio_img),
                                "box": [int(v) for v in box],
                            })
                            print(
                                f"[Hotrack] F{self.frame_idx} OBJ_SKIP: det_box "
                                f"reason=bbox_ratio_{ratio_img:.2f}"
                            )
                            continue
                    if bool(getattr(self, "skip_existing_object_box_prompts", False)) and object_boxes:
                        best_existing_iou = 0.0
                        best_existing_cover = 0.0
                        best_existing_id = None
                        box_area_safe = float(max(1.0, box_area))
                        for oid_existing, existing_box in list(object_boxes.items()):
                            if not isinstance(existing_box, (list, tuple)) or len(existing_box) != 4:
                                continue
                            try:
                                ex1, ey1, ex2, ey2 = [int(v) for v in list(existing_box)[:4]]
                            except Exception:
                                continue
                            iou_val = float(bbox_iou([bx1, by1, bx2, by2], [ex1, ey1, ex2, ey2]))
                            ix1 = max(int(bx1), int(ex1))
                            iy1 = max(int(by1), int(ey1))
                            ix2 = min(int(bx2), int(ex2))
                            iy2 = min(int(by2), int(ey2))
                            inter_area = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
                            cover_val = float(inter_area / box_area_safe)
                            if (iou_val, cover_val) > (best_existing_iou, best_existing_cover):
                                best_existing_iou = float(iou_val)
                                best_existing_cover = float(cover_val)
                                best_existing_id = int(oid_existing)
                        if (
                            best_existing_iou >= float(self.existing_object_box_iou_skip_th)
                            or best_existing_cover >= float(self.existing_object_box_cover_skip_th)
                        ):
                            reject_events.append({
                                "kind": "object",
                                "stage": "existing_track_box_gate",
                                "reason": "matched_existing_track",
                                "box": [int(v) for v in box],
                                "matched_obj_id": int(best_existing_id) if best_existing_id is not None else None,
                                "bbox_iou": float(best_existing_iou),
                                "bbox_cover": float(best_existing_cover),
                            })
                            continue
                box_np = np.array(box, dtype=np.float32)
                temp_obj_id = self.next_obj_id
                self.next_obj_id += 1
                
                _, temp_all_ids, temp_mask_logits, temp_state = self.sam2_tracker.add_new_points_or_box(
                    self.inference_state,
                    frame_idx=self.frame_idx,
                    obj_id=temp_obj_id,
                    box=box_np,
                )
                self.all_ids = temp_all_ids
                self.inference_state = temp_state
                
                if temp_mask_logits is None or len(temp_mask_logits) == 0:
                    continue
                
                new_mask = (temp_mask_logits[-1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze()
                if new_mask.shape[0] != height or new_mask.shape[1] != width:
                    new_mask = cv2.resize(new_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                new_mask = new_mask.astype(bool)
                raw_mask = new_mask.copy()

                centers: List[Tuple[int, int]] = []
                # Refine mask using uniformly distributed in-mask points.
                if bool(self.enable_prompt_refinement):
                    centers = self._mask_uniform_points(
                        new_mask,
                        max_points=self.dt_point_count,
                        min_dist=self.dt_point_min_dist,
                    )
                    if not centers:
                        centers = self._mask_center_points_dt(
                            new_mask,
                            max_points=self.dt_point_count,
                            min_dist=self.dt_point_min_dist,
                        )
                    if centers:
                        try:
                            pts = np.array(centers, dtype=np.float32)
                            labels = np.ones(len(centers), dtype=np.int32)
                            _, self.all_ids, pt_mask_logits, self.inference_state = self.sam2_tracker.add_new_points_or_box(
                                self.inference_state,
                                frame_idx=self.frame_idx,
                                obj_id=temp_obj_id,
                                points=pts,
                                labels=labels,
                                clear_old_points=True,
                            )
                            if pt_mask_logits is not None and len(pt_mask_logits) > 0:
                                pt_mask = (pt_mask_logits[-1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze()
                                if pt_mask.shape[0] != height or pt_mask.shape[1] != width:
                                    pt_mask = cv2.resize(pt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                if np.count_nonzero(pt_mask) > 0:
                                    new_mask = pt_mask.astype(bool)
                        except Exception:
                            pass
                self._current_target_points.append(list(centers))
                sam2_input_debug.append({
                    "box": [int(v) for v in box] if box is not None else None,
                    "points": [list(p) for p in centers],
                    "temp_id": int(temp_obj_id),
                })
                raw_mask_u8 = raw_mask.astype(np.uint8)
                refined_mask_u8 = new_mask.astype(np.uint8)
                raw_entry = {
                    "temp_obj_id": int(temp_obj_id),
                    "box": [int(v) for v in box] if box is not None else None,
                    "points": [list(p) for p in centers],
                    # Primary candidate mask before point-refine.
                    "mask": raw_mask_u8.copy(),
                    "mask_pre_refine": raw_mask_u8,
                    # Post point-refine mask before later clipping/gating tweaks.
                    "mask_post_refine": refined_mask_u8.copy(),
                    # Mask actually used by overlap/inclusion gate (updated if clipped).
                    "mask_decision": refined_mask_u8,
                    "status": "pending",
                    "reason": "pending",
                }
                raw_object_candidates.append(raw_entry)
                raw_object_candidate_by_temp[int(temp_obj_id)] = raw_entry

                # Keep all components at candidate stage; split/refine handles pruning later.
                target_mask_for_split = new_mask.copy()
                split_mask_appended = False
                new_bbox = self._get_mask_bbox(new_mask)

                # Reject if detection bbox is too large (likely background)
                img_area = float(height * width)
                if box is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    box_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
                else:
                    box_area = 0.0
                if hand_area_ref > 0.0:
                    ratio_hand = box_area / hand_area_ref
                    if ratio_hand > self.max_obj_vs_hand_ratio:
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "object",
                            "stage": "bbox_gate",
                            "reason": "bbox_vs_hand_ratio",
                            "value": float(ratio_hand),
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                        })
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                            f"reason=bbox_vs_hand_ratio_{ratio_hand:.2f}"
                        )
                        continue
                if img_area > 0:
                    ratio_img = box_area / img_area
                    if ratio_img > self.max_obj_area_ratio:
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "object",
                            "stage": "bbox_gate",
                            "reason": "bbox_ratio",
                            "value": float(ratio_img),
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                        })
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                            f"reason=bbox_ratio_{ratio_img:.2f}"
                        )
                        continue

                mask_area = float(np.sum(new_mask))
                if hand_area_ref > 0.0:
                    ratio_hand = mask_area / hand_area_ref
                    if ratio_hand > self.max_obj_vs_hand_ratio:
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "object",
                            "stage": "mask_gate",
                            "reason": "mask_vs_hand_ratio",
                            "value": float(ratio_hand),
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                        })
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                            f"reason=mask_vs_hand_ratio_{ratio_hand:.2f}"
                        )
                        continue
                if img_area > 0:
                    ratio_img = mask_area / img_area
                    if ratio_img > self.max_obj_area_ratio:
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "object",
                            "stage": "mask_gate",
                            "reason": "mask_ratio",
                            "value": float(ratio_img),
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                        })
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                            f"reason=mask_ratio_{ratio_img:.2f}"
                        )
                        continue

                # Reject if bbox is too large (likely background)

                det_box_force_new = False
                if (
                    bool(getattr(self, "force_new_when_det_box_isolated", True))
                    and isinstance(box, (list, tuple))
                    and len(box) == 4
                    and object_masks
                ):
                    try:
                        dbx1, dby1, dbx2, dby2 = [int(v) for v in list(box)[:4]]
                        dbx1 = max(0, min(width - 1, dbx1))
                        dbx2 = max(0, min(width - 1, dbx2))
                        dby1 = max(0, min(height - 1, dby1))
                        dby2 = max(0, min(height - 1, dby2))
                        if dbx2 > dbx1 and dby2 > dby1:
                            det_box_i = [int(dbx1), int(dby1), int(dbx2), int(dby2)]
                            det_box_area = float(max(0, dbx2 - dbx1) * max(0, dby2 - dby1))
                            cand_in_box = float(self._mask_in_box_ratio(new_mask, det_box_i))
                            max_exist_in_box = 0.0
                            max_exist_box_iou = 0.0
                            for exist_id, exist_mask in object_masks.items():
                                if exist_mask is None or np.count_nonzero(exist_mask) <= 0:
                                    continue
                                max_exist_in_box = max(
                                    float(max_exist_in_box),
                                    float(self._mask_in_box_ratio(exist_mask, det_box_i)),
                                )
                                exist_box = object_boxes.get(int(exist_id))
                                if isinstance(exist_box, (list, tuple)) and len(exist_box) == 4:
                                    try:
                                        exb = [int(v) for v in list(exist_box)[:4]]
                                        max_exist_box_iou = max(
                                            float(max_exist_box_iou),
                                            float(bbox_iou(det_box_i, exb)),
                                        )
                                    except Exception:
                                        pass
                            iso_iou_th = float(
                                np.clip(float(getattr(self, "force_new_det_box_isolation_iou_th", 0.05)), 0.0, 1.0)
                            )
                            iso_mask_th = float(
                                np.clip(float(getattr(self, "force_new_det_box_isolation_mask_in_th", 0.05)), 0.0, 1.0)
                            )
                            min_cand_in_box = float(
                                np.clip(float(getattr(self, "force_new_det_box_min_candidate_in_box_ratio", 0.01)), 0.0, 1.0)
                            )
                            if (
                                float(max_exist_in_box) <= float(iso_mask_th)
                                and float(max_exist_box_iou) <= float(iso_iou_th)
                                and float(cand_in_box) >= float(min_cand_in_box)
                            ):
                                clipped_mask = np.zeros_like(new_mask, dtype=bool)
                                clipped_mask[dby1:dby2, dbx1:dbx2] = new_mask[dby1:dby2, dbx1:dbx2]
                                clipped_area = int(np.count_nonzero(clipped_mask))
                                min_clipped_area = int(max(16, round(det_box_area * 0.02)))
                                if clipped_area >= int(min_clipped_area):
                                    det_box_force_new = True
                                    new_mask = clipped_mask.astype(bool)
                                    target_mask_for_split = new_mask.copy()
                                    new_bbox = self._get_mask_bbox(new_mask)
                                    raw_entry["det_box_force_new"] = True
                                    raw_entry["det_box_force_new_stats"] = {
                                        "cand_in_box": float(cand_in_box),
                                        "max_exist_in_box": float(max_exist_in_box),
                                        "max_exist_box_iou": float(max_exist_box_iou),
                                        "clipped_area": int(clipped_area),
                                        "min_clipped_area": int(min_clipped_area),
                                    }
                    except Exception:
                        pass
                raw_entry["mask_decision"] = new_mask.astype(np.uint8)

                # Reject if new mask overlaps existing objects too much
                max_ioa = 0.0
                max_ioa_obj_id: Optional[int] = None
                max_ioa_e2n = 0.0
                max_ioa_e2n_obj_id: Optional[int] = None
                max_ioa_n2e = 0.0
                max_ioa_n2e_obj_id: Optional[int] = None
                for exist_id, exist_mask in object_masks.items():
                    _, ioa_e2n, ioa_n2e = calculate_ioa_bidirectional(exist_mask, new_mask)
                    if ioa_e2n > max_ioa_e2n:
                        max_ioa_e2n = float(ioa_e2n)
                        max_ioa_e2n_obj_id = int(exist_id)
                    if ioa_n2e > max_ioa_n2e:
                        max_ioa_n2e = float(ioa_n2e)
                        max_ioa_n2e_obj_id = int(exist_id)
                    bid = float(max(ioa_e2n, ioa_n2e))
                    if bid > max_ioa:
                        max_ioa = float(bid)
                        max_ioa_obj_id = int(exist_id)
                if max_ioa >= self.new_mask_reject_ioa or max_ioa_n2e >= self.new_mask_inclusion_reject_ioa:
                    # Rescue path: when full SAM2 mask leaks outside det-box, re-check overlap
                    # with det-box-clipped mask. If clipped mask is isolated, keep as new.
                    if (not det_box_force_new) and isinstance(box, (list, tuple)) and len(box) == 4:
                        try:
                            dbx1, dby1, dbx2, dby2 = [int(v) for v in list(box)[:4]]
                            dbx1 = max(0, min(width - 1, dbx1))
                            dbx2 = max(0, min(width - 1, dbx2))
                            dby1 = max(0, min(height - 1, dby1))
                            dby2 = max(0, min(height - 1, dby2))
                            if dbx2 > dbx1 and dby2 > dby1:
                                clipped_mask = np.zeros_like(new_mask, dtype=bool)
                                clipped_mask[dby1:dby2, dbx1:dbx2] = new_mask[dby1:dby2, dbx1:dbx2]
                                clipped_area = int(np.count_nonzero(clipped_mask))
                                if clipped_area > 0:
                                    max_clip_ioa = 0.0
                                    max_clip_e2n = 0.0
                                    max_clip_n2e = 0.0
                                    max_clip_obj_id: Optional[int] = None
                                    max_clip_n2e_obj_id: Optional[int] = None
                                    for exist_id, exist_mask in object_masks.items():
                                        _, ioa_e2n_clip, ioa_n2e_clip = calculate_ioa_bidirectional(exist_mask, clipped_mask)
                                        if ioa_e2n_clip > max_clip_e2n:
                                            max_clip_e2n = float(ioa_e2n_clip)
                                        if ioa_n2e_clip > max_clip_n2e:
                                            max_clip_n2e = float(ioa_n2e_clip)
                                            max_clip_n2e_obj_id = int(exist_id)
                                        bid_clip = float(max(ioa_e2n_clip, ioa_n2e_clip))
                                        if bid_clip > max_clip_ioa:
                                            max_clip_ioa = float(bid_clip)
                                            max_clip_obj_id = int(exist_id)
                                    if (
                                        max_clip_ioa < float(self.new_mask_reject_ioa)
                                        and max_clip_n2e < float(self.new_mask_inclusion_reject_ioa)
                                    ):
                                        det_box_force_new = True
                                        new_mask = clipped_mask.astype(bool)
                                        target_mask_for_split = new_mask.copy()
                                        new_bbox = self._get_mask_bbox(new_mask)
                                        raw_entry["mask_decision"] = new_mask.astype(np.uint8)
                                        raw_entry["det_box_force_new"] = True
                                        raw_entry["det_box_rescue_new"] = True
                                        raw_entry["det_box_force_new_stats"] = {
                                            "mode": "overlap_rescue",
                                            "clipped_area": int(clipped_area),
                                            "clip_max_ioa": float(max_clip_ioa),
                                            "clip_max_e2n": float(max_clip_e2n),
                                            "clip_max_n2e": float(max_clip_n2e),
                                            "clip_overlap_obj_id": int(max_clip_obj_id) if max_clip_obj_id is not None else None,
                                            "clip_inclusion_obj_id": int(max_clip_n2e_obj_id) if max_clip_n2e_obj_id is not None else None,
                                        }
                        except Exception:
                            pass
                    if det_box_force_new:
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_FORCE_NEW_DETBOX: {temp_obj_id} "
                            f"(override overlap max_ioa={max_ioa:.2f} max_ioa_n2e={max_ioa_n2e:.2f})"
                        )
                    else:
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reason = f"high_ioa_{max_ioa:.2f}"
                        component_dup_parent = None
                        component_dup_score = 0.0
                        component_dup_other_overlap = 0.0
                        if max_ioa_n2e >= self.new_mask_inclusion_reject_ioa:
                            reason = f"inclusion_ioa_{max_ioa_n2e:.2f}"
                            # Inclusion rejection can still provide valid component-split evidence:
                            # keep this mask for split matching only when it duplicates a component
                            # (both-direction IoA high) and that component is not already
                            # represented by another tracked object.
                            best_parent = None
                            best_comp_mask = None
                            best_comp_score = 0.0
                            for exist_id, exist_mask in object_masks.items():
                                if exist_mask is None or np.count_nonzero(exist_mask) == 0:
                                    continue
                                mask_uint8 = exist_mask.astype(np.uint8)
                                num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                                if num_labels <= 2:
                                    continue
                                for label in range(1, num_labels):
                                    comp_mask = (labels == label)
                                    if np.count_nonzero(comp_mask) == 0:
                                        continue
                                    _, ioa_c_to_n, ioa_n_to_c = calculate_ioa_bidirectional(comp_mask, target_mask_for_split)
                                    if ioa_c_to_n < duplicate_ioa_threshold or ioa_n_to_c < duplicate_ioa_threshold:
                                        continue
                                    score = float(min(ioa_c_to_n, ioa_n_to_c))
                                    if score > best_comp_score:
                                        best_comp_score = score
                                        best_parent = int(exist_id)
                                        best_comp_mask = comp_mask
                            if best_parent is not None and best_comp_mask is not None:
                                best_other_overlap = 0.0
                                for other_id, other_mask in object_masks.items():
                                    if int(other_id) == int(best_parent):
                                        continue
                                    if other_mask is None or np.count_nonzero(other_mask) == 0:
                                        continue
                                    _, ioa_o_to_c, ioa_c_to_o = calculate_ioa_bidirectional(other_mask, best_comp_mask)
                                    best_other_overlap = max(best_other_overlap, ioa_o_to_c, ioa_c_to_o)
                                if best_other_overlap < self.new_mask_inclusion_reject_ioa:
                                    split_copy = target_mask_for_split.copy()
                                    self._current_target_masks.append(split_copy)
                                    self._current_split_target_masks.append(split_copy.copy())
                                    split_mask_appended = True
                                    component_dup_parent = int(best_parent)
                                    component_dup_score = float(best_comp_score)
                                    component_dup_other_overlap = float(best_other_overlap)
                        else:
                            self._current_target_masks.append(target_mask_for_split.copy())
                            split_mask_appended = True
                        # Keep detector-guided mask for attach/merge signal even when
                        # this temporary object is rejected as a new tracked object.
                        if not split_mask_appended:
                            self._current_target_masks.append(target_mask_for_split.copy())
                            split_mask_appended = True

                        reject_events.append({
                            "kind": "object",
                            "stage": "overlap_gate",
                            "reason": reason,
                            "value": float(max_ioa),
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                            "target_mask_kept_for_signal": bool(split_mask_appended),
                            "component_dup_parent": component_dup_parent,
                            "component_dup_score": float(component_dup_score),
                            "component_dup_other_overlap": float(component_dup_other_overlap),
                            "overlap_obj_id": int(max_ioa_obj_id) if max_ioa_obj_id is not None else None,
                            "overlap_e2n_obj_id": int(max_ioa_e2n_obj_id) if max_ioa_e2n_obj_id is not None else None,
                            "inclusion_obj_id": int(max_ioa_n2e_obj_id) if max_ioa_n2e_obj_id is not None else None,
                            "overlap_max_ioa": float(max_ioa),
                            "overlap_e2n": float(max_ioa_e2n),
                            "overlap_n2e": float(max_ioa_n2e),
                        })
                        overlap_txt = (
                            f" overlap_obj={int(max_ioa_obj_id)}"
                            if max_ioa_obj_id is not None
                            else ""
                        )
                        incl_txt = (
                            f" inclusion_obj={int(max_ioa_n2e_obj_id)}"
                            if max_ioa_n2e_obj_id is not None
                            else ""
                        )
                        print(
                            f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                            f"reason={reason}{overlap_txt}{incl_txt}"
                        )
                        continue
                if not split_mask_appended:
                    self._current_target_masks.append(target_mask_for_split.copy())
                    split_mask_appended = True

                ambiguous = False
                for prev_box in self._prev_hand_boxes:
                    if bbox_iou(new_bbox, prev_box) >= self.ambiguous_bbox_iou_threshold:
                        ambiguous = True
                        break
                if not ambiguous and hand_masks_list:
                    for hand_mask in hand_masks_list:
                        _, ioa_obj_to_hand, ioa_hand_to_obj = calculate_ioa_bidirectional(new_mask, hand_mask)
                        if max(ioa_obj_to_hand, ioa_hand_to_obj) >= self.ambiguous_mask_ioa_threshold:
                            ambiguous = True
                            break
                if ambiguous:
                    decision = self._update_ambiguous_region(new_bbox, "object")
                    if decision != "object":
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, temp_obj_id, strict=False, need_output=False
                        )
                        reject_events.append({
                            "kind": "object",
                            "stage": "ambiguous_gate",
                            "reason": "ambiguous_region",
                            "box": [int(v) for v in box] if box is not None else None,
                            "temp_id": int(temp_obj_id),
                        })
                        continue

                # Soft scoring: near-hand/duplicate/overlap become features, not hard gates.
                near_hand = self._is_near_hand_dilated(new_mask, hand_masks_list) if hand_masks_list else False
                is_dup, dup_reason = self._check_duplicate(
                    new_mask,
                    new_bbox,
                    object_masks,
                    object_boxes,
                    ioa_threshold=duplicate_ioa_threshold,
                )
                quality = self._compute_quality_score(new_mask, new_bbox, hand_masks_list)
                features = self._build_candidate_features(
                    mask=new_mask,
                    bbox=new_bbox,
                    det_box=[int(v) for v in box] if box is not None else None,
                    image_h=height,
                    image_w=width,
                    hand_masks=hand_masks_list,
                    object_masks=object_masks,
                    hand_area_ref=hand_area_ref,
                    quality=quality,
                    near_hand=near_hand,
                    duplicate_flag=is_dup,
                )
                score, score_raw, score_source = self._score_candidate_features(features)
                if is_dup:
                    features["duplicate_reason_len"] = float(len(str(dup_reason)))
                candidates.append({
                    "box": [int(v) for v in box] if box is not None else None,
                    "temp_obj_id": int(temp_obj_id),
                    "mask": new_mask,
                    "bbox": new_bbox,
                    "quality": float(quality),
                    "score": float(score),
                    "score_raw": float(score_raw),
                    "score_source": str(score_source),
                    "features": features,
                })
            
            # Soft decision stage: sort by score, enrich top-P with backfill stability.
            if not self.enable_candidate_scorer:
                for cand in candidates:
                    cand["score"] = float(np.clip(cand.get("quality", 0.0), 0.0, 1.0))
                    cand["score_source"] = "quality_only"

            candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            top_p = min(len(candidates), max(0, int(self.candidate_backfill_top_p)))
            for cand in candidates[:top_p]:
                bf = self._candidate_backfill_metrics(
                    candidate_mask=cand.get("mask"),
                    hand_masks_current=hand_masks_list,
                )
                if bf:
                    cand["features"].update(bf)
                    bf_score = float(bf.get("backfill_stability_score", 0.0))
                    cand["score"] = float(np.clip(0.75 * float(cand.get("score", 0.0)) + 0.25 * bf_score, 0.0, 1.0))
                    cand["score_source"] = f"{cand.get('score_source', 'unknown')}+backfill"
            candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

            accepted_in_frame: Dict[int, np.ndarray] = {}
            resolved_temp_ids: Set[int] = set()

            def _log_decision(cand: Dict[str, Any], status: str, reason: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
                row = {
                    "frame_idx": int(self.frame_idx),
                    "temp_obj_id": int(cand.get("temp_obj_id", -1)),
                    "score": float(cand.get("score", 0.0)),
                    "score_raw": float(cand.get("score_raw", 0.0)),
                    "score_source": str(cand.get("score_source", "")),
                    "status": str(status),
                    "reason": str(reason),
                    "box": cand.get("box"),
                    "bbox": [int(v) for v in (cand.get("bbox") or [0, 0, 0, 0])],
                    "features": dict(cand.get("features", {})),
                }
                if extra:
                    row.update(extra)
                candidate_log_rows.append(row)
                temp_id = int(cand.get("temp_obj_id", -1))
                raw_entry = raw_object_candidate_by_temp.get(temp_id)
                if raw_entry is not None:
                    raw_entry["status"] = str(status)
                    raw_entry["reason"] = str(reason)
                    if extra:
                        if "recovered_id" in extra and extra.get("recovered_id") is not None:
                            raw_entry["recovered_id"] = int(extra.get("recovered_id"))
                        if "target_id" in extra and extra.get("target_id") is not None:
                            raw_entry["target_id"] = int(extra.get("target_id"))

            for cand in candidates:
                box = cand.get("box")
                temp_obj_id = int(cand.get("temp_obj_id"))
                new_mask = cand.get("mask")
                new_bbox = cand.get("bbox")
                quality = float(cand.get("quality", 0.0))
                score = float(cand.get("score", 0.0))

                if score < float(self.candidate_score_threshold):
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, temp_obj_id, strict=False, need_output=False
                    )
                    resolved_temp_ids.add(int(temp_obj_id))
                    reject_events.append({
                        "kind": "object",
                        "stage": "score_gate",
                        "reason": "low_score",
                        "value": float(score),
                        "box": [int(v) for v in box] if box is not None else None,
                        "temp_id": int(temp_obj_id),
                    })
                    _log_decision(cand, "rejected", "low_score")
                    continue

                is_dup_frame = False
                for acc_id, acc_mask in accepted_in_frame.items():
                    _, ioa_acc_to_new, ioa_new_to_acc = calculate_ioa_bidirectional(acc_mask, new_mask)
                    if ioa_acc_to_new >= duplicate_ioa_threshold and ioa_new_to_acc >= duplicate_ioa_threshold:
                        is_dup_frame = True
                        print(f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} reason=dup_in_frame_with_{acc_id}")
                        break

                if is_dup_frame:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, temp_obj_id, strict=False, need_output=False
                    )
                    resolved_temp_ids.add(int(temp_obj_id))
                    reject_events.append({
                        "kind": "object",
                        "stage": "in_frame_dup",
                        "reason": "dup_in_frame",
                        "box": [int(v) for v in box] if box is not None else None,
                        "temp_id": int(temp_obj_id),
                    })
                    _log_decision(cand, "rejected", "dup_in_frame")
                    continue

                cooldown_hit = self._duplicate_cooldown_hit(new_mask)
                if cooldown_hit is not None:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, temp_obj_id, strict=False, need_output=False
                    )
                    resolved_temp_ids.add(int(temp_obj_id))
                    reject_events.append({
                        "kind": "object",
                        "stage": "duplicate_cooldown",
                        "reason": "recent_post_duplicate",
                        "box": [int(v) for v in box] if box is not None else None,
                        "temp_id": int(temp_obj_id),
                        "removed_id": int(cooldown_hit.get("removed_id", -1)),
                        "kept_id": int(cooldown_hit.get("kept_id")) if cooldown_hit.get("kept_id") is not None else None,
                        "age": int(cooldown_hit.get("age", -1)),
                        "max_ioa": float(cooldown_hit.get("max_ioa", 0.0)),
                    })
                    _log_decision(cand, "rejected", "recent_post_duplicate")
                    print(
                        f"[Hotrack] F{self.frame_idx} OBJ_REJECT: {temp_obj_id} "
                        f"reason=recent_post_duplicate_{int(cooldown_hit.get('removed_id', -1))}"
                    )
                    continue

                # Check if this matches a recently lost object (ID recovery)
                recovered_id = self._check_lost_object_recovery(image_bgr, new_mask)
                if recovered_id is not None and int(recovered_id) in self._retired_struct_ids:
                    reject_events.append({
                        "kind": "object",
                        "stage": "id_recovery_gate",
                        "reason": "retired_struct_id",
                        "temp_id": int(temp_obj_id),
                        "recovered_id": int(recovered_id),
                    })
                    recovered_id = None
                if recovered_id is not None and self._is_id_recovery_blocked(int(recovered_id), int(self.frame_idx)):
                    reject_events.append({
                        "kind": "object",
                        "stage": "id_recovery_gate",
                        "reason": "recovery_blocked_after_struct_op",
                        "temp_id": int(temp_obj_id),
                        "recovered_id": int(recovered_id),
                        "block_until": int(self._struct_id_recovery_block_until.get(int(recovered_id), int(self.frame_idx))),
                    })
                    recovered_id = None
                if recovered_id is not None:
                    area = np.sum(new_mask)
                    print(
                        f"[Hotrack] F{self.frame_idx} EVENT_ID_RECOVERY: "
                        f"obj-{temp_obj_id-100} -> obj-{recovered_id-100} "
                        f"(DINO >= {self.dino_recovery_sim_threshold:.2f}, area={area} px)"
                    )
                    self.events.append({
                        "type": "id_recovery",
                        "frame": self.frame_idx,
                        "temp_id": temp_obj_id,
                        "recovered_id": recovered_id,
                        "area": int(area)
                    })

                    # If another active ID already occupies this region, treat it as duplicate
                    # and keep the recovered/original ID.
                    conflict_id = None
                    conflict_ioa = 0.0
                    for oid, exist_mask in tracked_masks.items():
                        oid_int = int(oid)
                        if oid_int < 100:
                            continue
                        if oid_int in {int(temp_obj_id), int(recovered_id)}:
                            continue
                        if exist_mask is None or np.count_nonzero(exist_mask) == 0:
                            continue
                        _, ioa_e2n, ioa_n2e = calculate_ioa_bidirectional(exist_mask, new_mask)
                        max_ioa = float(max(ioa_e2n, ioa_n2e))
                        if max_ioa > float(conflict_ioa):
                            conflict_ioa = float(max_ioa)
                            conflict_id = int(oid_int)

                    if conflict_id is not None and float(conflict_ioa) >= float(self.post_track_dup_ioa_threshold):
                        conflict_mask = tracked_masks.get(int(conflict_id))
                        if conflict_mask is not None and np.count_nonzero(conflict_mask) > 0:
                            self._register_recent_post_duplicate(int(conflict_id), int(recovered_id), conflict_mask)
                        self.all_ids, _ = self.sam2_tracker.remove_object(
                            self.inference_state, int(conflict_id), strict=False, need_output=False
                        )
                        tracked_masks.pop(int(conflict_id), None)
                        tracked_boxes.pop(int(conflict_id), None)
                        object_masks.pop(int(conflict_id), None)
                        object_boxes.pop(int(conflict_id), None)
                        accepted_in_frame.pop(int(conflict_id), None)
                        self.mask_history.pop(int(conflict_id), None)
                        print(
                            f"[Hotrack] F{self.frame_idx} EVENT_ID_RECOVERY_DUP: "
                            f"remove obj-{int(conflict_id)-100}, keep recovered obj-{int(recovered_id)-100} "
                            f"(max_ioa={float(conflict_ioa):.2f})"
                        )
                        self.events.append({
                            "type": "id_recovery_duplicate",
                            "frame": int(self.frame_idx),
                            "removed_id": int(conflict_id),
                            "recovered_id": int(recovered_id),
                            "max_ioa": float(conflict_ioa),
                        })

                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, temp_obj_id, strict=False, need_output=False
                    )
                    resolved_temp_ids.add(int(temp_obj_id))

                    # Re-seed recovered/original ID with the current new mask directly.
                    ret = self.sam2_tracker.add_new_mask(
                        self.inference_state,
                        frame_idx=self.frame_idx,
                        obj_id=recovered_id,
                        mask=new_mask,
                    )
                    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                        self.all_ids = ret[1]

                    recovered_mask_bool = new_mask.astype(bool).copy()
                    recovered_bbox = self._get_mask_bbox(recovered_mask_bool)
                    tracked_masks[int(recovered_id)] = recovered_mask_bool
                    tracked_boxes[int(recovered_id)] = recovered_bbox
                    object_masks[int(recovered_id)] = recovered_mask_bool
                    object_boxes[int(recovered_id)] = recovered_bbox
                    accepted_in_frame[int(recovered_id)] = recovered_mask_bool
                    new_objects.append({
                        "obj_id": int(recovered_id),
                        "bbox": recovered_bbox,
                        "source": "id_recovery",
                    })
                    if int(recovered_id) in self.confusion_objects_history:
                        del self.confusion_objects_history[int(recovered_id)]
                    if self.use_dino_id:
                        rec_embed = self._dino_encode(image_bgr, recovered_mask_bool)
                        if rec_embed is not None:
                            self._update_dino_embedding_history(int(recovered_id), rec_embed)

                    _log_decision(
                        cand,
                        "accepted",
                        "id_recovery",
                        extra={
                            "recovered_id": int(recovered_id),
                            "conflict_removed_id": int(conflict_id) if conflict_id is not None else None,
                            "conflict_max_ioa": float(conflict_ioa),
                        },
                    )
                    if self._is_new_accept_cap_reached(accepted_in_frame):
                        break
                    continue

                temporal_match = self._find_temporal_reassign_target(new_mask, object_masks)
                if temporal_match is not None:
                    target_id = int(temporal_match["obj_id"])
                    self._apply_temporal_reassign(
                        target_id=target_id,
                        replacement_mask=new_mask,
                        tracked_masks=tracked_masks,
                        tracked_boxes=tracked_boxes,
                        object_masks=object_masks,
                        object_boxes=object_boxes,
                        accepted_in_frame=accepted_in_frame,
                        temp_obj_id=int(temp_obj_id),
                        source="new_candidate",
                        metrics={
                            "ioa_prev_to_new": float(temporal_match.get("ioa_prev_to_new", 0.0)),
                            "ioa_new_to_prev": float(temporal_match.get("ioa_new_to_prev", 0.0)),
                            "prev_area": float(temporal_match.get("prev_area", 0.0)),
                            "cur_area": float(temporal_match.get("cur_area", 0.0)),
                            "new_area": float(temporal_match.get("new_area", 0.0)),
                        },
                    )
                    resolved_temp_ids.add(int(temp_obj_id))
                    _log_decision(cand, "accepted", "temporal_reassign", extra={"target_id": int(target_id)})
                    continue

                tracked_masks[temp_obj_id] = new_mask
                tracked_boxes[temp_obj_id] = new_bbox
                object_masks[temp_obj_id] = new_mask
                object_boxes[temp_obj_id] = new_bbox
                accepted_in_frame[temp_obj_id] = new_mask
                resolved_temp_ids.add(int(temp_obj_id))
                new_objects.append({
                    "obj_id": int(temp_obj_id),
                    "bbox": box,
                    "source": "new_object",
                    "det_box_count": int(len(list(self._current_target_boxes or []))),
                })
                self._log_dino_similarity(image_bgr, temp_obj_id, new_mask, tracked_masks)

                area = np.sum(new_mask)
                print(
                    f"[Hotrack] F{self.frame_idx} EVENT_NEW: obj-{temp_obj_id-100} "
                    f"(score={score:.2f}, q={quality:.2f}, area={area} px)"
                )
                self.events.append({
                    "type": "new_object",
                    "frame": self.frame_idx,
                    "obj_id": temp_obj_id,
                    "area": int(area),
                    "quality": float(quality),
                    "score": float(score),
                })
                _log_decision(cand, "accepted", "new_object")
                if self._is_new_accept_cap_reached(accepted_in_frame):
                    break

            # Remove unresolved temporary IDs that were never selected.
            for cand in candidates:
                tid = int(cand.get("temp_obj_id"))
                if tid in resolved_temp_ids:
                    continue
                if tid in self.all_ids:
                    self.all_ids, _ = self.sam2_tracker.remove_object(
                        self.inference_state, tid, strict=False, need_output=False
                    )
                reject_events.append({
                    "kind": "object",
                    "stage": "topk_prune",
                    "reason": "not_selected",
                    "temp_id": int(tid),
                })
                _log_decision(cand, "rejected", "not_selected")

            # Mark raw candidates that were rejected before soft-scoring (hard gates).
            reject_by_temp: Dict[int, Dict[str, Any]] = {}
            for ev in reject_events:
                if not isinstance(ev, dict):
                    continue
                if str(ev.get("kind", "")) != "object":
                    continue
                tid_val = ev.get("temp_id")
                if tid_val is None:
                    continue
                try:
                    tid_i = int(tid_val)
                except Exception:
                    continue
                if tid_i not in reject_by_temp:
                    reject_by_temp[tid_i] = ev
            for raw_entry in raw_object_candidates:
                if str(raw_entry.get("status", "")) != "pending":
                    continue
                tid_i = int(raw_entry.get("temp_obj_id", -1))
                ev = reject_by_temp.get(tid_i)
                if ev is None:
                    continue
                stage = str(ev.get("stage", "gate"))
                reason = str(ev.get("reason", "rejected"))
                raw_entry["status"] = "rejected"
                raw_entry["reason"] = f"{stage}:{reason}"
                if stage == "overlap_gate":
                    overlap_obj_id = ev.get("overlap_obj_id")
                    inclusion_obj_id = ev.get("inclusion_obj_id")
                    try:
                        if overlap_obj_id is not None:
                            raw_entry["overlap_obj_id"] = int(overlap_obj_id)
                    except Exception:
                        pass
                    try:
                        if inclusion_obj_id is not None:
                            raw_entry["inclusion_obj_id"] = int(inclusion_obj_id)
                    except Exception:
                        pass
                    for k in ("overlap_max_ioa", "overlap_e2n", "overlap_n2e"):
                        if ev.get(k) is None:
                            continue
                        try:
                            raw_entry[k] = float(ev.get(k))
                        except Exception:
                            continue
                    id_for_reason = raw_entry.get("inclusion_obj_id", raw_entry.get("overlap_obj_id"))
                    if id_for_reason is not None:
                        try:
                            raw_entry["reason"] = f"{stage}:{reason}@obj{int(id_for_reason)}"
                        except Exception:
                            pass
        
        self._append_candidate_logs(candidate_log_rows)
        timing["object_processing"] = time.time() - t_obj_start

        # Add new objects from detection masks if they moderately overlap but are not duplicates
        if self._current_target_masks:
            current_det_box_count = int(len(list(self._current_target_boxes or [])))
            object_ids = [i for i in self.all_ids if i >= 100]
            component_promote_by_parent: Dict[int, List[Dict[str, Any]]] = {}
            for det_mask in self._current_target_masks:
                if det_mask is None or np.count_nonzero(det_mask) == 0:
                    continue
                component_ioa = 0.0
                component_parent = None
                component_mask = None
                component_label = None
                if bool(self.enable_component_promote_from_det):
                    for oid in object_ids:
                        obj_mask = tracked_masks.get(oid)
                        if obj_mask is None or np.count_nonzero(obj_mask) == 0:
                            continue
                        mask_uint8 = obj_mask.astype(np.uint8)
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                        if num_labels <= 2:
                            continue
                        raw_areas = [
                            int(stats[label, cv2.CC_STAT_AREA])
                            for label in range(1, int(num_labels))
                            if int(stats[label, cv2.CC_STAT_AREA]) > 0
                        ]
                        if len(raw_areas) < 2:
                            continue
                        largest_area = int(max(raw_areas))
                        min_area_by_ratio = int(max(1.0, float(largest_area) * float(self.split_min_area_ratio)))
                        min_valid_area = int(max(int(self.min_mask_area), int(min_area_by_ratio)))
                        valid_labels = [
                            int(label)
                            for label in range(1, int(num_labels))
                            if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_valid_area)
                        ]
                        if len(valid_labels) < 2:
                            continue
                        for label in valid_labels:
                            comp_mask = (labels == label)
                            if np.count_nonzero(comp_mask) < int(min_valid_area):
                                continue
                            _, ioa_c_to_d, ioa_d_to_c = calculate_ioa_bidirectional(comp_mask, det_mask)
                            # Component promotion should require near-duplicate
                            # agreement in both directions.
                            ioa = min(ioa_c_to_d, ioa_d_to_c)
                            if ioa > component_ioa:
                                component_ioa = ioa
                                component_parent = int(oid)
                                component_mask = comp_mask
                                component_label = int(label)
                component_promote_th = float(
                    max(
                        float(self.detect_component_add_ioa_threshold),
                        float(self.split_component_duplicate_ioa_th),
                    )
                )
                promote_from_component = bool(
                    bool(self.enable_component_promote_from_det)
                    and component_parent is not None
                    and component_mask is not None
                    and component_ioa >= component_promote_th
                )

                new_id = int(self.next_obj_id)
                best_ioa = 0.0
                best_ioa_other = 0.0
                best_other_obj_id = None
                for oid in object_ids:
                    obj_mask = tracked_masks.get(oid)
                    if obj_mask is None:
                        continue
                    _, ioa_o_to_d, ioa_d_to_o = calculate_ioa_bidirectional(obj_mask, det_mask)
                    pair_ioa = max(ioa_o_to_d, ioa_d_to_o)
                    best_ioa = max(best_ioa, pair_ioa)
                    if (
                        promote_from_component
                        and component_parent is not None
                        and int(oid) != int(component_parent)
                        and pair_ioa > best_ioa_other
                    ):
                        best_ioa_other = float(pair_ioa)
                        best_other_obj_id = int(oid)
                # Promote-from-component should only happen when that component is
                # not already represented by another tracked object.
                if promote_from_component and best_ioa_other >= self.new_mask_inclusion_reject_ioa:
                    reject_events.append({
                        "kind": "object",
                        "stage": "component_promote_gate",
                        "reason": f"overlap_existing_obj_{int(best_other_obj_id) if best_other_obj_id is not None else -1}",
                        "value": float(best_ioa_other),
                        "temp_id": None,
                    })
                    continue
                force_promote_from_component = bool(promote_from_component)

                if (not force_promote_from_component) and best_ioa >= self.new_mask_reject_ioa:
                    continue
                if (not force_promote_from_component) and best_ioa < self.detect_add_ioa_threshold:
                    continue

                if not bool(force_promote_from_component):
                    temporal_match = self._find_temporal_reassign_target(det_mask, object_masks)
                    if temporal_match is not None:
                        target_id = int(temporal_match["obj_id"])
                        self._apply_temporal_reassign(
                            target_id=target_id,
                            replacement_mask=det_mask,
                            tracked_masks=tracked_masks,
                            tracked_boxes=tracked_boxes,
                            object_masks=object_masks,
                            object_boxes=object_boxes,
                            temp_obj_id=None,
                            source="det_mask",
                            metrics={
                                "ioa_prev_to_new": float(temporal_match.get("ioa_prev_to_new", 0.0)),
                                "ioa_new_to_prev": float(temporal_match.get("ioa_new_to_prev", 0.0)),
                                "prev_area": float(temporal_match.get("prev_area", 0.0)),
                                "cur_area": float(temporal_match.get("cur_area", 0.0)),
                                "new_area": float(temporal_match.get("new_area", 0.0)),
                            },
                        )
                        continue

                cooldown_hit = self._duplicate_cooldown_hit(det_mask)
                if cooldown_hit is not None:
                    reject_events.append({
                        "kind": "object",
                        "stage": "duplicate_cooldown",
                        "reason": "recent_post_duplicate",
                        "temp_id": None,
                        "removed_id": int(cooldown_hit.get("removed_id", -1)),
                        "kept_id": int(cooldown_hit.get("kept_id")) if cooldown_hit.get("kept_id") is not None else None,
                        "age": int(cooldown_hit.get("age", -1)),
                        "max_ioa": float(cooldown_hit.get("max_ioa", 0.0)),
                    })
                    continue

                if (
                    bool(force_promote_from_component)
                    and bool(self.enable_structural_ops)
                    and component_parent is not None
                ):
                    if current_det_box_count <= 1:
                        reject_events.append({
                            "kind": "object",
                            "stage": "component_promote_split_gate",
                            "reason": "single_det_box_hard_gate",
                            "parent_id": int(component_parent),
                            "det_box_count": int(current_det_box_count),
                        })
                        continue
                    component_promote_by_parent.setdefault(int(component_parent), []).append({
                        "mask": det_mask.astype(bool).copy(),
                        "score": float(max(best_ioa, component_ioa)),
                        "overlap_ioa": float(best_ioa),
                        "component_ioa": float(component_ioa),
                        "component_label": int(component_label) if component_label is not None else None,
                    })

                self.next_obj_id += 1
                ret = self.sam2_tracker.add_new_mask(
                    self.inference_state,
                    frame_idx=self.frame_idx,
                    obj_id=new_id,
                    mask=det_mask,
                )
                if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                    self.all_ids = ret[1]
                tracked_masks[new_id] = det_mask
                tracked_boxes[new_id] = self._get_mask_bbox(det_mask)
                object_masks[new_id] = det_mask
                object_boxes[new_id] = tracked_boxes[new_id]
                self.mask_history[new_id] = deque([(self.frame_idx, det_mask.copy())], maxlen=self.mask_history_length)
                new_objects.append({
                    "obj_id": int(new_id),
                    "bbox": tracked_boxes[new_id],
                    "source": "det",
                    "promote_from_component": bool(force_promote_from_component),
                    "component_parent": int(component_parent) if component_parent is not None else None,
                    "det_box_count": int(current_det_box_count),
                })
                self._log_dino_similarity(image_bgr, new_id, det_mask, tracked_masks)
                print(
                    f"[Hotrack] F{self.frame_idx} EVENT_NEW_FROM_DET: obj-{new_id-100} "
                    f"(overlap_ioa={best_ioa:.2f}, mode={'component' if bool(force_promote_from_component) else 'normal'})"
                )
                self.events.append({
                    "type": "new_object_from_det",
                    "frame": self.frame_idx,
                    "obj_id": new_id,
                    "overlap_ioa": float(best_ioa),
                    "promote_from_component": bool(force_promote_from_component),
                    "component_parent": int(component_parent) if component_parent is not None else None,
                    "component_ioa": float(component_ioa),
                })

            if bool(self.enable_structural_ops):
                for parent_id, rows in list(component_promote_by_parent.items()):
                    self.events.append({
                        "type": "component_promote_seed",
                        "frame": int(self.frame_idx),
                        "parent": int(parent_id),
                        "candidate_count": int(len(list(rows or []))),
                    })

        # 7. Log frame summary
        current_hand_ids = [i for i in self.all_ids if i < 100]
        current_object_ids = [i for i in self.all_ids if i >= 100]
        print(f"[Hotrack] F{self.frame_idx} SUMMARY: {len(current_hand_ids)} hands, {len(current_object_ids)} objects tracked")

        # 7.2 Background motion estimation (t-1 -> t)
        prev_gray = self._prev_gray
        fg_union = None
        for m in tracked_masks.values():
            fg_union = m if fg_union is None else np.logical_or(fg_union, m)
        if prev_gray is None:
            A_bg, bg_motion_ok, bg_inlier_ratio, bg_num_matches = None, False, 0.0, 0
        else:
            A_bg, bg_motion_ok, bg_inlier_ratio, bg_num_matches = self._estimate_bg_motion(
                prev_gray, gray, self._prev_fg_mask, fg_union
            )
        self._prev_fg_mask = fg_union

        # 7.25 Pending split logic for multi-component objects
        if bool(getattr(self, "track_hand_masks", True)):
            hand_masks_list = [tracked_masks[hid] for hid in self.all_ids if hid < 100 and hid in tracked_masks]
        else:
            hand_masks_list = list(current_hand_proxy_masks or [])
        split_parent_ids_frame: Set[int] = set()
        for oid in [i for i in self.all_ids if i >= 100]:
            mask = tracked_masks.get(oid)
            if mask is None or np.count_nonzero(mask) == 0:
                self._pending_splits.pop(oid, None)
                continue
            mask_uint8 = mask.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            if num_labels <= 2:
                self._pending_splits.pop(oid, None)
                continue

            # Build component info
            comp_info = []
            for label in range(1, num_labels):
                comp_mask = (labels == label)
                area = int(np.sum(comp_mask))
                if area == 0:
                    continue
                ys, xs = np.where(comp_mask)
                cx = float(xs.mean())
                cy = float(ys.mean())
                comp_info.append({
                    "label": label,
                    "mask": comp_mask,
                    "area": area,
                    "centroid": (cx, cy),
                })
            if not comp_info:
                continue

            # Determine component overlap to previous mask (for fallback and tie-break).
            prev_mask = self._prev_tracked_masks.get(oid)
            if prev_mask is not None and prev_mask.shape != mask.shape:
                prev_mask = cv2.resize(
                    prev_mask.astype(np.uint8),
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            for comp in comp_info:
                if prev_mask is None:
                    comp["overlap_prev"] = 0
                    comp["ioa"] = 0.0
                else:
                    overlap = int(np.sum(np.logical_and(comp["mask"], prev_mask)))
                    comp["overlap_prev"] = int(overlap)
                    comp["ioa"] = overlap / max(comp["area"], 1)
            # Baseline: keep original ID on the component with the highest IoA.
            # Tie-breakers: absolute overlap pixels, then area.
            comp_info.sort(
                key=lambda x: (
                    float(x.get("ioa", 0.0)),
                    int(x.get("overlap_prev", 0)),
                    int(x.get("area", 0)),
                ),
                reverse=True,
            )
            main = comp_info[0]

            # Prefer DINO similarity over IoA when split components compete for the same old ID.
            if self.use_dino_id:
                ref_embed = None
                if prev_frame_bgr is not None and prev_mask is not None and np.count_nonzero(prev_mask) > 0:
                    ref_embed = self._dino_encode(prev_frame_bgr, prev_mask)
                if ref_embed is None:
                    ref_embed = self._dino_embedding_history.get(int(oid))

                if ref_embed is not None:
                    largest_area = max(int(c.get("area", 0)) for c in comp_info) if comp_info else 0
                    min_area_by_ratio = int(max(1.0, float(largest_area) * float(self.split_min_area_ratio)))
                    dino_candidates = [
                        c for c in comp_info
                        if int(c.get("area", 0)) >= int(self.min_mask_area)
                        and int(c.get("area", 0)) >= int(min_area_by_ratio)
                    ]
                    if not dino_candidates:
                        dino_candidates = list(comp_info)

                    best_dino_comp = None
                    best_dino_key = None
                    for comp in dino_candidates:
                        emb = self._dino_encode(image_bgr, comp["mask"])
                        if emb is None:
                            comp["dino_sim"] = -1.0
                            continue
                        sim = float(self._cosine_sim(ref_embed, emb))
                        comp["dino_sim"] = sim
                        comp["dino_embed"] = emb
                        key = (
                            float(sim),
                            float(comp.get("ioa", 0.0)),
                            int(comp.get("overlap_prev", 0)),
                            int(comp.get("area", 0)),
                        )
                        if best_dino_key is None or key > best_dino_key:
                            best_dino_key = key
                            best_dino_comp = comp

                    if best_dino_comp is not None:
                        main = best_dino_comp
                        emb_main = best_dino_comp.get("dino_embed")
                        if emb_main is not None:
                            self._update_dino_embedding_history(int(oid), emb_main)
                        print(
                            f"[Hotrack] F{self.frame_idx} SPLIT_MAIN_BY_DINO: "
                            f"obj-{oid-100} keep label={int(main['label'])} sim={float(main.get('dino_sim', -1.0)):.2f}"
                        )

            main_area = float(main["area"])

            # Filter tiny components (noise)
            valid = [c for c in comp_info if c["area"] >= self.min_mask_area and c["area"] >= main_area * self.split_min_area_ratio]
            if not valid:
                tracked_masks[oid] = main["mask"]
                tracked_boxes[oid] = self._get_mask_bbox(main["mask"])
                continue

            # If any component overlaps another object's mask (duplicate),
            # remove it from this object and re-seed with the parent ID.
            duplicate_labels: Set[int] = set()
            overlap_targets: Dict[int, int] = {}
            for comp in valid:
                if comp["label"] == main["label"]:
                    continue
                other_masks = {
                    int(other_id): other_mask
                    for other_id, other_mask in tracked_masks.items()
                    if int(other_id) >= 100
                    and int(other_id) != int(oid)
                    and other_mask is not None
                    and np.count_nonzero(other_mask) > 0
                }
                overlap_obj_id = self._find_duplicate_object(
                    comp["mask"],
                    other_masks,
                    ioa_threshold=duplicate_ioa_threshold,
                )
                if overlap_obj_id is None:
                    continue
                duplicate_labels.add(int(comp["label"]))
                overlap_targets[int(comp["label"])] = int(overlap_obj_id)


            # Default: keep union of valid components (excluding duplicates)
            union_mask = np.zeros_like(mask_uint8, dtype=bool)
            for comp in valid:
                if int(comp["label"]) in duplicate_labels:
                    continue
                union_mask |= comp["mask"]
            tracked_masks[oid] = union_mask
            tracked_boxes[oid] = self._get_mask_bbox(union_mask)
            self.mask_history[oid] = deque(
                [(self.frame_idx, union_mask.copy())],
                maxlen=self.mask_history_length,
            )
            self.sam2_tracker.add_new_mask(
                self.inference_state,
                frame_idx=self.frame_idx,
                obj_id=oid,
                mask=union_mask,
            )
            for label, other_id in overlap_targets.items():
                self.events.append({
                    "type": "component_overlap_refine",
                    "frame": self.frame_idx,
                    "obj_id": int(oid),
                    "overlap_obj_id": int(other_id),
                    "component_label": int(label),
                })
                print(
                    f"[Hotrack] F{self.frame_idx} OBJ_COMP_OVERLAP: obj-{oid-100} "
                    f"removed component {label} overlapping obj-{other_id-100}"
                )

            # Precompute detector-level duplicate matches for each component label.
            det_duplicate_by_label: Dict[int, Dict[str, float]] = {}
            split_signal_masks = _collect_split_signal_masks_current_frame()
            comp_dup_th = float(np.clip(self.split_component_duplicate_ioa_th, 0.0, 1.0))
            for candidate in valid:
                cand_label = int(candidate["label"])
                if cand_label in duplicate_labels:
                    continue
                best_det_idx = None
                best_det_score = -1.0
                best_source = "none"

                # Primary match: mask-level bi-directional IoA.
                if split_signal_masks:
                    for det_idx, det_mask in enumerate(split_signal_masks):
                        if det_mask is None:
                            continue
                        _, ioa_c_to_d, ioa_d_to_c = calculate_ioa_bidirectional(candidate["mask"], det_mask)
                        if ioa_c_to_d < comp_dup_th or ioa_d_to_c < comp_dup_th:
                            continue
                        score = float(min(ioa_c_to_d, ioa_d_to_c))
                        if score > best_det_score:
                            best_det_score = score
                            best_det_idx = int(det_idx)
                            best_source = "mask_ioa"

                # Relaxed fallback: expanded detector box containment for elongated components.
                if (
                    best_det_idx is None
                    and bool(self.split_component_allow_box_fallback)
                    and self._current_target_boxes
                ):
                    det_box_match = self._best_det_box_match_for_component(
                        candidate["mask"],
                        list(self._current_target_boxes or []),
                    )
                    if bool(det_box_match.get("matched", False)):
                        best_det_idx = int(det_box_match.get("det_idx", -1))
                        best_det_score = float(det_box_match.get("score", 0.0))
                        best_source = str(det_box_match.get("source", "strict_box"))

                if best_det_idx is not None and best_det_idx >= 0:
                    det_duplicate_by_label[cand_label] = {
                        "det_idx": int(best_det_idx),
                        "score": float(best_det_score),
                        "source": str(best_source),
                    }
            # Split evidence policy:
            # - Prefer direct detector-backed split labels (non-main component matched to det mask).
            # - If no direct split label exists but any component is detector-backed,
            #   allow fallback split promotion for non-main components.
            direct_split_labels: Set[int] = {
                int(lbl)
                for lbl in det_duplicate_by_label.keys()
                if int(lbl) != int(main["label"]) and int(lbl) not in duplicate_labels
            }
            has_direct_split_evidence = bool(len(direct_split_labels) > 0)
            any_component_det_backed = bool(len(det_duplicate_by_label) > 0)
            fallback_det_info: Optional[Dict[str, float]] = det_duplicate_by_label.get(int(main["label"]))
            if fallback_det_info is None and det_duplicate_by_label:
                fallback_det_info = max(
                    det_duplicate_by_label.values(),
                    key=lambda r: float((r or {}).get("score", 0.0)),
                )

            pending = self._pending_splits.get(oid, {})
            updated_pending: Dict[str, Any] = {}

            split_applied = False
            for comp in valid:
                if int(comp["label"]) in duplicate_labels:
                    continue
                if comp["label"] == main["label"]:
                    continue
                det_info_comp = det_duplicate_by_label.get(int(comp["label"]))
                fallback_split_evidence = bool(
                    bool(self.split_allow_fallback_evidence)
                    and det_info_comp is None
                    and not bool(has_direct_split_evidence)
                    and bool(any_component_det_backed)
                )
                split_evidence = bool(det_info_comp is not None or fallback_split_evidence)

                # Match to existing pending entry by IoA
                best_key = None
                best_ioa = 0.0
                for key, entry in pending.items():
                    pmask = entry.get("mask")
                    if pmask is None or pmask.shape != comp["mask"].shape:
                        continue
                    overlap = int(np.sum(np.logical_and(pmask, comp["mask"])))
                    ioa = overlap / max(int(np.sum(comp["mask"])), 1)
                    if ioa > best_ioa:
                        best_ioa = ioa
                        best_key = key

                if best_key is None or best_ioa < 0.3:
                    key = f"{oid}:{comp['label']}:{self.frame_idx}"
                    entry = {
                        "frames": 0,
                        "mask": comp["mask"].copy(),
                        "centroid": comp["centroid"],
                        "split_score": 0,
                    }
                else:
                    key = best_key
                    entry = pending[best_key]

                if split_evidence:
                    entry["frames"] = entry.get("frames", 0) + 1
                    entry["split_score"] = entry.get("split_score", 0) + 1
                else:
                    entry["frames"] = 0
                    entry["split_score"] = 0
                entry["mask"] = comp["mask"].copy()
                entry["centroid"] = comp["centroid"]

                confirm_frames = max(1, int(self.split_confirm_frames))
                det_box_count = int(len(list(self._current_target_boxes or [])))
                if bool(self.split_multicomponent_instant_new):
                    split_ready = bool(split_evidence)
                else:
                    # Single-det-box scenes are prone to over-splitting from transient
                    # multi-component noise. Require stronger temporal persistence.
                    required_frames_local = int(confirm_frames)
                    if int(det_box_count) <= 1:
                        required_frames_local = int(
                            max(
                                int(required_frames_local),
                                int(max(1, int(self.split_new_det_pair_persist_frames))),
                            )
                        )
                    split_ready = bool(
                        split_evidence
                        and int(entry.get("frames", 0)) >= int(required_frames_local)
                    )
                if split_ready:
                    det_match = comp
                    det_info = det_info_comp if det_info_comp is not None else fallback_det_info
                    if det_info is None:
                        updated_pending[key] = entry
                        continue
                    det_dup_score = float(det_info.get("score", 0.0))
                    det_match_det_idx = int(det_info.get("det_idx", -1))
                    det_match_source = str(det_info.get("source", "mask_ioa"))

                    if det_match_det_idx < 0 and bool(self.split_component_allow_box_fallback):
                        split_box_match = self._best_det_box_match_for_component(
                            det_match["mask"],
                            list(self._current_target_boxes or []),
                        )
                        if bool(split_box_match.get("matched", False)):
                            det_match_det_idx = int(split_box_match.get("det_idx", -1))
                            det_dup_score = float(max(float(det_dup_score), float(split_box_match.get("score", 0.0))))
                            det_match_source = str(split_box_match.get("source", "strict_box"))

                    # Existing ID stays on `main` component (DINO-first, IoA fallback).
                    keep_comp = main
                    keep_det_info = det_duplicate_by_label.get(int(keep_comp["label"]))
                    keep_box_match = self._best_det_box_match_for_component(
                        keep_comp["mask"],
                        list(self._current_target_boxes or []),
                    )
                    keep_comp_det_idx = int(keep_box_match.get("det_idx", -1)) if bool(keep_box_match.get("matched", False)) else -1
                    if isinstance(keep_det_info, dict):
                        keep_det_idx_hint = int(keep_det_info.get("det_idx", -1))
                        if keep_det_idx_hint >= 0:
                            keep_comp_det_idx = int(keep_det_idx_hint)

                    if det_match_det_idx < 0:
                        updated_pending[key] = entry
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_gate",
                            "reason": "split_det_required",
                            "obj_id": int(oid),
                            "split_label": int(det_match["label"]),
                            "keep_label": int(keep_comp["label"]),
                            "split_det_idx": int(det_match_det_idx),
                            "keep_det_idx": int(keep_comp_det_idx),
                            "split_det_source": str(det_match_source),
                        })
                        continue

                    # Strict distinct-box gate is enforced only when >=2 detector boxes exist.
                    enforce_two_box_gate = bool(self.split_require_two_det_boxes) and int(det_box_count) >= 2
                    if enforce_two_box_gate:
                        if keep_comp_det_idx < 0:
                            keep_reason = "keep_det_strict_required" if not isinstance(keep_det_info, dict) else "keep_det_required"
                            updated_pending[key] = entry
                            reject_events.append({
                                "kind": "object",
                                "stage": "split_gate",
                                "reason": str(keep_reason),
                                "obj_id": int(oid),
                                "split_label": int(det_match["label"]),
                                "keep_label": int(keep_comp["label"]),
                                "split_det_idx": int(det_match_det_idx),
                                "keep_det_idx": int(keep_comp_det_idx),
                                "keep_det_source": str(keep_box_match.get("source", "none")),
                            })
                            continue
                        if int(keep_comp_det_idx) == int(det_match_det_idx):
                            updated_pending[key] = entry
                            reject_events.append({
                                "kind": "object",
                                "stage": "split_gate",
                                "reason": "same_det_box_occlusion",
                                "obj_id": int(oid),
                                "split_label": int(det_match["label"]),
                                "keep_label": int(keep_comp["label"]),
                                "det_idx": int(det_match_det_idx),
                                "split_det_source": str(det_match_source),
                                "keep_det_source": str(keep_box_match.get("source", "none")),
                            })
                            continue
                    elif int(det_box_count) >= 2 and keep_comp_det_idx >= 0 and int(keep_comp_det_idx) == int(det_match_det_idx):
                        updated_pending[key] = entry
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_gate",
                            "reason": "same_det_box_occlusion",
                            "obj_id": int(oid),
                            "split_label": int(det_match["label"]),
                            "keep_label": int(keep_comp["label"]),
                            "det_idx": int(det_match_det_idx),
                            "split_det_source": str(det_match_source),
                            "keep_det_source": str(keep_box_match.get("source", "none")),
                        })
                        continue

                    # If split component still overlaps another tracked object, do not split.
                    max_other_overlap = 0.0
                    max_other_overlap_id = None
                    for other_id, other_mask in object_masks.items():
                        other_id_i = int(other_id)
                        if other_id_i == int(oid):
                            continue
                        other_mask_b = _to_bool_mask(other_mask)
                        if other_mask_b is None or np.count_nonzero(other_mask_b) <= 0:
                            continue
                        _, ioa_o_to_s, ioa_s_to_o = calculate_ioa_bidirectional(other_mask_b, det_match["mask"])
                        ov = float(max(ioa_o_to_s, ioa_s_to_o))
                        if ov > max_other_overlap:
                            max_other_overlap = float(ov)
                            max_other_overlap_id = int(other_id_i)
                    if max_other_overlap >= float(self.split_other_overlap_reject_ioa):
                        updated_pending[key] = entry
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_gate",
                            "reason": f"overlap_existing_obj_{int(max_other_overlap_id) if max_other_overlap_id is not None else -1}",
                            "obj_id": int(oid),
                            "split_label": int(det_match["label"]),
                            "overlap_ioa": float(max_other_overlap),
                            "threshold": float(self.split_other_overlap_reject_ioa),
                        })
                        continue

                    cooldown_hit = self._duplicate_cooldown_hit(det_match["mask"])
                    if cooldown_hit is not None:
                        updated_pending[key] = entry
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_gate",
                            "reason": "recent_post_duplicate",
                            "obj_id": int(oid),
                            "split_label": int(det_match["label"]),
                            "removed_id": int(cooldown_hit.get("removed_id", -1)),
                            "kept_id": int(cooldown_hit.get("kept_id")) if cooldown_hit.get("kept_id") is not None else None,
                            "age": int(cooldown_hit.get("age", -1)),
                            "max_ioa": float(cooldown_hit.get("max_ioa", 0.0)),
                        })
                        continue

                    split_child_mask = det_match["mask"]
                    keep_child_mask = np.zeros_like(split_child_mask, dtype=bool)
                    for keep_comp_cand in list(valid):
                        lbl_keep = int(keep_comp_cand.get("label", -1))
                        if int(lbl_keep) == int(det_match.get("label", -1)):
                            continue
                        if int(lbl_keep) in duplicate_labels:
                            continue
                        km = _to_bool_mask(keep_comp_cand.get("mask"))
                        if km is None or np.count_nonzero(km) <= 0:
                            continue
                        keep_child_mask |= km
                    keep_child_mask = np.logical_and(
                        keep_child_mask.astype(bool),
                        np.logical_not(split_child_mask.astype(bool)),
                    )
                    if np.count_nonzero(keep_child_mask) <= 0:
                        updated_pending[key] = entry
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_gate",
                            "reason": "keep_component_empty_after_split",
                            "obj_id": int(oid),
                            "split_label": int(det_match["label"]),
                        })
                        continue

                    keep_det_idx_for_seed = int(keep_comp_det_idx)
                    if (
                        0 <= int(det_match_det_idx) < len(split_signal_masks)
                        and int(det_match_det_idx) != int(keep_det_idx_for_seed)
                    ):
                        split_seed = _to_bool_mask(split_signal_masks[int(det_match_det_idx)])
                        if split_seed is not None and np.count_nonzero(split_seed) > 0:
                            split_child_mask = split_seed.astype(bool).copy()

                    split_reason = "split_confirm_fallback" if bool(fallback_split_evidence) else "split_confirm"
                    split_ev = self._apply_keep_and_new_transaction(
                        parent_id=int(oid),
                        keep_mask=keep_child_mask,
                        new_mask=split_child_mask,
                        reason=str(split_reason),
                        score=float(det_dup_score),
                        frame_idx=int(self.frame_idx),
                        tracked_masks=tracked_masks,
                        tracked_boxes=tracked_boxes,
                        object_masks=object_masks,
                        object_boxes=object_boxes,
                        new_objects=new_objects,
                        new_obj_meta={
                            "source": "multicomponent_split",
                            "det_box_count": int(det_box_count),
                            "split_det_idx": int(det_match_det_idx),
                            "keep_det_idx": int(keep_comp_det_idx),
                        },
                        emit_struct_split=False,
                    )
                    if split_ev is None:
                        updated_pending[key] = entry
                        continue
                    new_child = int(split_ev.get("child", -1))
                    if new_child >= 100:
                        self._pending_split_from_new[int(new_child)] = {
                            "parent_id": int(oid),
                            "start_frame": int(self.frame_idx),
                            "last_frame": int(self.frame_idx),
                            "score": float(det_dup_score),
                            "reason": str(split_reason),
                        }
                    print(
                        f"[Hotrack] F{self.frame_idx} OBJ_SPLIT_NEW_CANDIDATE: obj-{oid-100} "
                        f"new=obj-{new_child-100 if new_child >= 100 else -1} "
                        f"keep_label={keep_comp['label']} split_label={det_match['label']} "
                        f"dup={det_dup_score:.2f} det_pair=({det_match_det_idx},{keep_comp_det_idx}) "
                        f"mode={'fallback' if bool(fallback_split_evidence) else 'direct'} "
                        f"(candidate_confirm={int(entry.get('frames', 0))})"
                    )
                    split_parent_ids_frame.add(int(oid))
                    split_applied = True
                    break

                updated_pending[key] = entry

            if split_applied:
                self._pending_splits.pop(int(oid), None)
            else:
                self._pending_splits[oid] = updated_pending

        # Seed split candidates from any newly created objects in this frame.
        self._seed_pending_split_from_new_objects(
            new_objects=new_objects,
            tracked_masks=tracked_masks,
        )

        # 7.26 Split confirmation from newly created objects (window-based).
        # Candidate trigger: new object creation first, then temporal validation.
        if bool(self.enable_structural_ops) and self._pending_split_from_new:
            max_age = int(max(1, int(self.split_candidate_max_age_frames)))
            fail_drop = int(max(1, int(self.split_candidate_fail_drop_frames)))
            if self._split_verify_cache:
                live_children = {int(k) for k in self._pending_split_from_new.keys()}
                self._split_verify_cache = {
                    k: v
                    for k, v in list(self._split_verify_cache.items())
                    if isinstance(k, tuple) and len(k) >= 2 and int(k[1]) in live_children
                }

            split_confirm_pass_rows: List[Dict[str, Any]] = []

            def _cluster_key_from_candidate(crow: Dict[str, Any]) -> Tuple[int, int]:
                start_f = int(crow.get("start_frame", self.frame_idx))
                try:
                    seed_parent = int(crow.get("seed_parent_id", crow.get("parent_id", -1)))
                except Exception:
                    seed_parent = int(crow.get("parent_id", -1))
                return (int(start_f), int(seed_parent))

            # Explicit group key for "same-frame, same-parent, multi new objects from det mask".
            det_multi_children_by_cluster: Dict[Tuple[int, int], Set[int]] = {}
            if isinstance(new_objects, list):
                for nrow in list(new_objects):
                    if not isinstance(nrow, dict):
                        continue
                    source_txt = str(nrow.get("source") or "").strip().lower()
                    if source_txt != "det":
                        continue
                    try:
                        child_new_i = int(nrow.get("obj_id", -1))
                    except Exception:
                        child_new_i = -1
                    try:
                        parent_new_i = int(nrow.get("component_parent", -1))
                    except Exception:
                        parent_new_i = -1
                    if int(child_new_i) < 100 or int(parent_new_i) < 100:
                        continue
                    key_new = (int(self.frame_idx), int(parent_new_i))
                    det_multi_children_by_cluster.setdefault(key_new, set()).add(int(child_new_i))
            det_multi_children_by_cluster = {
                (int(k[0]), int(k[1])): {int(x) for x in list(v)}
                for k, v in list(det_multi_children_by_cluster.items())
                if isinstance(k, tuple)
                and len(k) == 2
                and int(k[1]) >= 100
                and len(set(int(x) for x in list(v))) >= 2
            }

            for child_id, cand in list(self._pending_split_from_new.items()):
                try:
                    child_i = int(child_id)
                    parent_i = int(cand.get("parent_id"))
                except Exception:
                    reject_events.append({
                        "kind": "object",
                        "stage": "split_confirm_input_gate",
                        "reason": "invalid_candidate_row",
                        "child_id_raw": child_id,
                        "candidate": dict(cand or {}) if isinstance(cand, dict) else None,
                    })
                    self._pending_split_from_new.pop(int(child_id), None)
                    continue

                age = int(self.frame_idx - int(cand.get("start_frame", self.frame_idx)))
                cand_cluster_key = _cluster_key_from_candidate(cand)
                det_multi_children = set(det_multi_children_by_cluster.get(cand_cluster_key, set()))
                in_multi_det_parent_group = bool(cand.get("det_multi_group", False))
                if (not bool(in_multi_det_parent_group)) and bool(
                    len(det_multi_children) >= 2 and int(child_i) in det_multi_children
                ):
                    in_multi_det_parent_group = True
                    cand["det_multi_group"] = True
                    cand["det_multi_group_parent"] = int(cand_cluster_key[1])
                    self._pending_split_from_new[int(child_i)] = cand
                if bool(in_multi_det_parent_group) and int(cand_cluster_key[1]) >= 100:
                    parent_i = int(cand_cluster_key[1])
                    cand["parent_id"] = int(parent_i)
                    self._pending_split_from_new[int(child_i)] = cand

                def _log_split_confirm_reject(stage: str, reason: str, **extra: Any) -> None:
                    row: Dict[str, Any] = {
                        "kind": "object",
                        "stage": str(stage),
                        "reason": str(reason),
                        "parent_id": int(parent_i),
                        "child_id": int(child_i),
                        "frame_idx": int(self.frame_idx),
                        "start_frame": int(cand.get("start_frame", self.frame_idx)),
                        "candidate_age": int(age),
                        "verify_fail_streak": int(cand.get("verify_fail_streak", 0)),
                    }
                    try:
                        row["candidate_score"] = float(cand.get("score", 0.0))
                    except Exception:
                        pass
                    for k, v in dict(extra or {}).items():
                        row[k] = v
                    reject_events.append(row)

                if int(age) > int(max_age):
                    _log_split_confirm_reject(
                        "split_confirm_age_gate",
                        "candidate_expired",
                        max_age=int(max_age),
                    )
                    if bool(self.split_candidate_revert_on_expire):
                        self._revert_unconfirmed_split_candidate(
                            parent_id=int(parent_i),
                            child_id=int(child_i),
                            frame_idx=int(self.frame_idx),
                            tracked_masks=tracked_masks,
                            tracked_boxes=tracked_boxes,
                            object_masks=object_masks,
                            object_boxes=object_boxes,
                            new_objects=new_objects,
                            reason="candidate_expired",
                        )
                    self._pending_split_from_new.pop(int(child_i), None)
                    continue

                pair_key = (min(int(parent_i), int(child_i)), max(int(parent_i), int(child_i)))
                if pair_key in self._emitted_struct_split_pairs:
                    _log_split_confirm_reject(
                        "split_confirm_pair_gate",
                        "pair_already_emitted",
                        pair_key=[int(pair_key[0]), int(pair_key[1])],
                    )
                    self._pending_split_from_new.pop(int(child_i), None)
                    continue

                child_mask = _to_bool_mask(tracked_masks.get(int(child_i)))
                if (
                    child_mask is None
                    or np.count_nonzero(child_mask) <= 0
                ):
                    cand["last_frame"] = int(self.frame_idx)
                    cand["last_verify_pass"] = False
                    cand["last_verify_reason"] = "child_missing"
                    cand["verify_fail_streak"] = int(cand.get("verify_fail_streak", 0)) + 1
                    _log_split_confirm_reject(
                        "split_confirm_child_gate",
                        "child_missing",
                        fail_drop=int(fail_drop),
                        next_verify_fail_streak=int(cand.get("verify_fail_streak", 0)),
                    )
                    self._pending_split_from_new[int(child_i)] = cand
                    if int(cand.get("verify_fail_streak", 0)) >= int(fail_drop):
                        if bool(self.split_candidate_revert_on_fail):
                            self._revert_unconfirmed_split_candidate(
                                parent_id=int(parent_i),
                                child_id=int(child_i),
                                frame_idx=int(self.frame_idx),
                                tracked_masks=tracked_masks,
                                tracked_boxes=tracked_boxes,
                                object_masks=object_masks,
                                object_boxes=object_boxes,
                                new_objects=new_objects,
                                reason="verify_fail_streak:child_missing",
                            )
                        self._pending_split_from_new.pop(int(child_i), None)
                    continue

                split_start_fi = int(cand.get("start_frame", self.frame_idx))
                split_cache_key = (int(parent_i), int(child_i), int(split_start_fi))
                split_verify: Optional[Dict[str, Any]] = None
                cached_verify = self._split_verify_cache.get(split_cache_key)
                if isinstance(cached_verify, dict):
                    split_verify = dict(cached_verify)
                    split_verify["cache_hit"] = True
                if split_verify is None:
                    split_verify = self._validate_split_union_vs_parent_forward_track(
                        parent_id=int(parent_i),
                        child_id=int(child_i),
                        split_start_frame=int(split_start_fi),
                        split_mask_now=child_mask.astype(bool),
                        tracked_masks_now=tracked_masks,
                    )
                    self._split_verify_cache[split_cache_key] = dict(split_verify or {})
                if not isinstance(split_verify, dict):
                    split_verify = {"pass": False, "reason": "split_verify_invalid"}
                cache_hit_verify = bool(split_verify.get("cache_hit", False))

                cand["last_frame"] = int(self.frame_idx)
                cand["last_verify_pass"] = bool(split_verify.get("pass", False))
                cand["last_verify_reason"] = str(split_verify.get("reason", ""))
                if split_verify.get("debug_dir") is not None:
                    cand["last_verify_debug_dir"] = str(split_verify.get("debug_dir"))
                cand["verify_fail_streak"] = int(cand.get("verify_fail_streak", 0))
                self._pending_split_from_new[int(child_i)] = cand

                split_verify_pass = bool(split_verify.get("pass", False))
                verify_reason_txt = str(split_verify.get("reason", ""))
                if not bool(split_verify_pass):
                    fail_streak_next = int(cand.get("verify_fail_streak", 0)) + 1
                    _log_split_confirm_reject(
                        "split_confirm_verify_gate",
                        str(verify_reason_txt or "split_verify_failed"),
                        cache_hit=bool(cache_hit_verify),
                        verify_fail_streak_next=int(fail_streak_next),
                        non_blocking_for_multi_det_group=bool(in_multi_det_parent_group),
                        split_verify=(
                            {
                                "pass": bool(split_verify.get("pass", False)),
                                "reason": str(split_verify.get("reason", "")),
                                "matched_parent_id": split_verify.get("matched_parent_id"),
                                "candidate_parent_tail_now": split_verify.get("candidate_parent_tail_now"),
                                "candidate_parent_max_tail": split_verify.get("candidate_parent_max_tail"),
                                "candidate_parent_pass_count": split_verify.get("candidate_parent_pass_count"),
                                "candidate_parent_max_tail_pass": split_verify.get("candidate_parent_max_tail_pass"),
                            }
                            if isinstance(split_verify, dict)
                            else None
                        ),
                    )
                    if not bool(in_multi_det_parent_group):
                        cand["verify_fail_streak"] = int(fail_streak_next)
                        self._pending_split_from_new[int(child_i)] = cand
                        if int(cand.get("verify_fail_streak", 0)) >= int(fail_drop):
                            if bool(self.split_candidate_revert_on_fail):
                                self._revert_unconfirmed_split_candidate(
                                    parent_id=int(parent_i),
                                    child_id=int(child_i),
                                    frame_idx=int(self.frame_idx),
                                    tracked_masks=tracked_masks,
                                    tracked_boxes=tracked_boxes,
                                    object_masks=object_masks,
                                    object_boxes=object_boxes,
                                    new_objects=new_objects,
                                    reason=f"verify_fail_streak:{str(split_verify.get('reason', 'fail'))}",
                                )
                            self._pending_split_from_new.pop(int(child_i), None)
                        continue
                    cand["verify_fail_streak"] = 0
                    self._pending_split_from_new[int(child_i)] = cand

                matched_parent_i = int(parent_i)
                matched_parent_id = split_verify.get("matched_parent_id")
                if bool(split_verify_pass) and (not bool(in_multi_det_parent_group)) and matched_parent_id is not None:
                    try:
                        matched_parent_cand = int(matched_parent_id)
                    except Exception:
                        matched_parent_cand = int(parent_i)
                    if int(matched_parent_cand) >= 100 and int(matched_parent_cand) != int(parent_i):
                        matched_parent_mask_now = _to_bool_mask(tracked_masks.get(int(matched_parent_cand)))
                        matched_parent_present_now = bool(
                            matched_parent_mask_now is not None
                            and np.count_nonzero(matched_parent_mask_now) > 0
                        )
                        if matched_parent_present_now:
                            matched_parent_i = int(matched_parent_cand)
                            cand["matched_parent_id"] = int(matched_parent_i)
                            cand["parent_update_reason"] = "split_verify_matched_parent_present"
                        else:
                            cand["matched_parent_id"] = int(matched_parent_cand)
                            cand["parent_update_reason"] = "split_verify_matched_parent_not_present_now"
                        self._pending_split_from_new[int(child_i)] = cand

                parent_mask_now = _to_bool_mask(tracked_masks.get(int(matched_parent_i)))
                child_mask_now = _to_bool_mask(tracked_masks.get(int(child_i)))
                if (
                    parent_mask_now is None
                    or child_mask_now is None
                    or np.count_nonzero(parent_mask_now) <= 0
                    or np.count_nonzero(child_mask_now) <= 0
                ):
                    cand["last_frame"] = int(self.frame_idx)
                    cand["last_verify_pass"] = False
                    cand["last_verify_reason"] = "parent_or_child_missing_after_verify"
                    cand["verify_fail_streak"] = int(cand.get("verify_fail_streak", 0)) + 1
                    _log_split_confirm_reject(
                        "split_confirm_parent_child_gate",
                        "parent_or_child_missing_after_verify",
                        matched_parent_id=int(matched_parent_i),
                        parent_mask_present=bool(parent_mask_now is not None and np.count_nonzero(parent_mask_now) > 0),
                        child_mask_present=bool(child_mask_now is not None and np.count_nonzero(child_mask_now) > 0),
                    )
                    self._pending_split_from_new[int(child_i)] = cand
                    continue

                # Split-confirm from new-object follows replay/containment verification.
                # Detector-box matching below is diagnostic only (non-blocking).
                det_signal_masks = _collect_split_signal_masks_current_frame()
                split_det_match_th = float(
                    max(
                        float(self.detect_component_add_ioa_threshold),
                        float(self.split_component_duplicate_ioa_th),
                    )
                )
                keep_mask_now = np.logical_and(
                    parent_mask_now.astype(bool),
                    np.logical_not(child_mask_now.astype(bool)),
                )

                def _best_det_bidir_match(mask_in: np.ndarray) -> Dict[str, Any]:
                    if mask_in is None or np.count_nonzero(mask_in) <= 0:
                        return {"det_idx": -1, "score": 0.0}
                    best_idx = -1
                    best_score = 0.0
                    for det_idx, det_mask in enumerate(det_signal_masks):
                        if det_mask is None or np.count_nonzero(det_mask) <= 0:
                            continue
                        _, ioa_m_to_d, ioa_d_to_m = calculate_ioa_bidirectional(mask_in, det_mask)
                        score = float(min(ioa_m_to_d, ioa_d_to_m))
                        if score > best_score:
                            best_score = float(score)
                            best_idx = int(det_idx)
                    return {"det_idx": int(best_idx), "score": float(best_score)}

                keep_match = _best_det_bidir_match(keep_mask_now)
                child_match = _best_det_bidir_match(child_mask_now.astype(bool))
                keep_det_idx = int(keep_match.get("det_idx", -1))
                child_det_idx = int(child_match.get("det_idx", -1))
                keep_det_score = float(keep_match.get("score", 0.0))
                child_det_score = float(child_match.get("score", 0.0))
                strict_det_gate_pass_raw = bool(
                    keep_det_idx >= 0
                    and child_det_idx >= 0
                    and keep_det_idx != child_det_idx
                    and keep_det_score >= split_det_match_th
                    and child_det_score >= split_det_match_th
                )
                child_det_gate_pass_raw = bool(
                    child_det_idx >= 0
                    and child_det_score >= split_det_match_th
                )
                det_gate_for_candidate_raw = (
                    bool(child_det_gate_pass_raw)
                    if bool(in_multi_det_parent_group)
                    else bool(strict_det_gate_pass_raw)
                )
                if not bool(det_gate_for_candidate_raw):
                    reject_events.append({
                        "kind": "object",
                        "stage": "split_confirm_det_gate_diag",
                        "reason": "child_det_missing" if bool(in_multi_det_parent_group) else "insufficient_distinct_det_match",
                        "parent_id": int(matched_parent_i),
                        "child_id": int(child_i),
                        "keep_det_idx": int(keep_det_idx),
                        "child_det_idx": int(child_det_idx),
                        "keep_det_score": float(keep_det_score),
                        "child_det_score": float(child_det_score),
                        "threshold": float(split_det_match_th),
                        "det_signal_count": int(len(det_signal_masks)),
                        "enforced": False,
                    })
                cand["verify_fail_streak"] = 0
                cand["last_verify_pass"] = bool(split_verify_pass)
                cand["last_verify_reason"] = (
                    str(split_verify.get("reason") or "ok")
                    if bool(split_verify_pass)
                    else str(split_verify.get("reason") or "soft")
                )
                self._pending_split_from_new[int(child_i)] = cand

                score_val = cand.get("score", 0.0)
                try:
                    score_f = float(score_val)
                except Exception:
                    score_f = 0.0
                reason_base = str(cand.get("reason") or "split_candidate")
                verify_reason = str(split_verify.get("reason") or ("ok" if bool(split_verify_pass) else "soft"))
                split_reason = f"split_from_new_object:{reason_base}:{verify_reason}"

                try:
                    seed_parent_i_row = int(cand.get("seed_parent_id", parent_i))
                except Exception:
                    seed_parent_i_row = int(parent_i)
                split_confirmed_tail = {
                    "event_type": "split",
                    "confirm_mode": "replay_verify",
                    "parent_id": int(parent_i),
                    "matched_parent_id": int(matched_parent_i),
                    "child_id": int(child_i),
                    "start_frame": int(split_start_fi),
                    "candidate_frames": int(cand.get("frames", 0) or 0),
                    "candidate_age": int(age),
                    "required_frames": int(split_verify.get("same_parent_required_frames", split_verify.get("required_frames", 0)) or 0),
                    "candidate_score": float(score_f),
                    "verify_reason": str(verify_reason),
                    "cache_hit": bool(cache_hit_verify),
                    "split_eval_direction": str(split_verify.get("split_eval_direction") or ""),
                    "candidate_parent_id": int(split_verify.get("candidate_parent_id", parent_i) or parent_i),
                    "candidate_parent_tail_now": int(split_verify.get("candidate_parent_tail_now", 0) or 0),
                    "candidate_parent_max_tail": int(split_verify.get("candidate_parent_max_tail", 0) or 0),
                    "candidate_parent_pass_count": int(split_verify.get("candidate_parent_pass_count", 0) or 0),
                    "candidate_parent_max_tail_pass": int(split_verify.get("candidate_parent_max_tail_pass", 0) or 0),
                    "matched_parent_tail_now": int(split_verify.get("matched_parent_tail_now", 0) or 0),
                    "matched_parent_max_tail": int(split_verify.get("matched_parent_max_tail", 0) or 0),
                    "matched_parent_mean_ratio": float(split_verify.get("matched_parent_mean_ratio", 0.0) or 0.0),
                    "frame_indices": [int(x) for x in list(split_verify.get("frame_indices", []) or [])],
                }
                split_confirm_pass_rows.append({
                    "child_id": int(child_i),
                    "parent_id": int(parent_i),
                    "matched_parent_id": int(matched_parent_i),
                    "start_frame": int(split_start_fi),
                    "seed_parent_id": int(seed_parent_i_row),
                    "cluster_key": [int(x) for x in _cluster_key_from_candidate(cand)],
                    "candidate_score": float(score_f),
                    "split_reason": str(split_reason),
                    "verify_reason": str(verify_reason),
                    "cache_hit": bool(cache_hit_verify),
                    "keep_det_idx": int(keep_det_idx),
                    "child_det_idx": int(child_det_idx),
                    "keep_det_score": float(keep_det_score),
                    "child_det_score": float(child_det_score),
                    "det_threshold": float(split_det_match_th),
                    "strict_det_gate_pass_raw": bool(strict_det_gate_pass_raw),
                    "child_det_gate_pass_raw": bool(child_det_gate_pass_raw),
                    "det_gate_for_candidate_raw": bool(det_gate_for_candidate_raw),
                    "det_gate_enforced": False,
                    "split_verify_pass": bool(split_verify_pass),
                    "in_multi_det_parent_group": bool(in_multi_det_parent_group),
                    "confirmed_tail": dict(split_confirmed_tail),
                })

            pass_rows_by_cluster: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for row in list(split_confirm_pass_rows):
                ckey = (int(row["cluster_key"][0]), int(row["cluster_key"][1]))
                pass_rows_by_cluster.setdefault(ckey, []).append(row)

            cluster_consumed_children: Set[int] = set()
            overlap_drop_th = float(np.clip(self.new_mask_reject_ioa, 0.0, 1.0))

            for ckey, rows in list(pass_rows_by_cluster.items()):
                if len(rows) < 2:
                    continue

                start_fi, seed_parent_i = int(ckey[0]), int(ckey[1])
                cluster_multi_det = bool(
                    any(bool(r.get("in_multi_det_parent_group", False)) for r in list(rows))
                )
                parent_groups: Dict[int, List[Dict[str, Any]]] = {}
                if bool(cluster_multi_det):
                    forced_parent_i = int(seed_parent_i)
                    if int(forced_parent_i) < 100:
                        parent_votes: Dict[int, int] = {}
                        for row in list(rows):
                            try:
                                spid = int(row.get("seed_parent_id", -1))
                            except Exception:
                                spid = -1
                            if int(spid) < 100:
                                continue
                            parent_votes[int(spid)] = int(parent_votes.get(int(spid), 0)) + 1
                        if parent_votes:
                            forced_parent_i = int(
                                sorted(parent_votes.items(), key=lambda it: (int(it[1]), int(it[0])), reverse=True)[0][0]
                            )
                    forced_rows = [
                        r for r in list(rows)
                        if int(r.get("seed_parent_id", -1)) == int(forced_parent_i)
                        and int(r.get("child_id", -1)) >= 100
                    ]
                    if int(forced_parent_i) >= 100 and forced_rows:
                        parent_groups[int(forced_parent_i)] = forced_rows
                else:
                    for row in list(rows):
                        pid = int(row.get("matched_parent_id", -1))
                        if int(pid) < 100:
                            continue
                        parent_groups.setdefault(int(pid), []).append(row)

                selected_parent = -1
                selected_rows: List[Dict[str, Any]] = []
                selected_det_count = 0
                selected_total_score = 0.0

                for pid, prow in list(parent_groups.items()):
                    best_by_det: Dict[int, Dict[str, Any]] = {}
                    for row in list(prow):
                        det_idx = int(row.get("child_det_idx", -1))
                        if int(det_idx) < 0:
                            continue
                        prev_row = best_by_det.get(int(det_idx))
                        if prev_row is None:
                            best_by_det[int(det_idx)] = row
                            continue
                        prev_rank = (
                            float(prev_row.get("child_det_score", 0.0)),
                            float(prev_row.get("candidate_score", 0.0)),
                            -int(prev_row.get("child_id", -1)),
                        )
                        row_rank = (
                            float(row.get("child_det_score", 0.0)),
                            float(row.get("candidate_score", 0.0)),
                            -int(row.get("child_id", -1)),
                        )
                        if row_rank > prev_rank:
                            best_by_det[int(det_idx)] = row
                    dedup_rows = list(best_by_det.values())
                    if len(dedup_rows) < 2:
                        # Do not require distinct detector boxes for cluster split.
                        # Fallback to score-based rows when det indices are missing/same.
                        dedup_rows = sorted(
                            [
                                r for r in list(prow)
                                if int(r.get("child_id", -1)) >= 100
                            ],
                            key=lambda r: (
                                -float(r.get("candidate_score", 0.0)),
                                -float(r.get("child_det_score", 0.0)),
                                int(r.get("child_id", -1)),
                            ),
                        )
                    if len(dedup_rows) < 2:
                        continue
                    if (not bool(cluster_multi_det)) and (not any(bool(r.get("split_verify_pass", False)) for r in dedup_rows)):
                        continue
                    det_count = int(len({
                        int(r.get("child_det_idx", -1))
                        for r in list(dedup_rows)
                        if int(r.get("child_det_idx", -1)) >= 0
                    }))
                    total_score = float(sum(float(r.get("candidate_score", 0.0)) for r in dedup_rows))
                    if (
                        not selected_rows
                        or det_count > selected_det_count
                        or (
                            det_count == selected_det_count
                            and total_score > selected_total_score
                        )
                    ):
                        selected_parent = int(pid)
                        selected_rows = dedup_rows
                        selected_det_count = int(det_count)
                        selected_total_score = float(total_score)

                cluster_children = sorted({int(r.get("child_id", -1)) for r in list(rows) if int(r.get("child_id", -1)) >= 100})

                if int(selected_parent) >= 100 and len(selected_rows) >= 2:
                    selected_rows = sorted(
                        list(selected_rows),
                        key=lambda r: (
                            -float(r.get("candidate_score", 0.0)),
                            -float(r.get("child_det_score", 0.0)),
                            int(r.get("child_id", -1)),
                        ),
                    )
                    selected_child_ids = sorted({int(r.get("child_id", -1)) for r in list(selected_rows) if int(r.get("child_id", -1)) >= 100})
                    cluster_score = float(max((float(r.get("candidate_score", 0.0)) for r in selected_rows), default=0.0))
                    cluster_reason = (
                        "split_from_new_cluster:"
                        f"start={int(start_fi)}:"
                        f"seed={int(seed_parent_i)}:"
                        f"parent={int(selected_parent)}"
                    )
                    cluster_confirmed_tail = {
                        "event_type": "split",
                        "confirm_mode": "cluster_existing_children",
                        "parent_id": int(selected_parent),
                        "seed_parent_id": int(seed_parent_i),
                        "start_frame": int(start_fi),
                        "cluster_children": [int(x) for x in cluster_children],
                        "selected_children": [int(x) for x in selected_child_ids],
                        "dropped_children": [int(x) for x in sorted(set(cluster_children) - set(selected_child_ids))],
                        "required_children": 2,
                        "candidate_rows": [
                            {
                                "child_id": int(r.get("child_id", -1)),
                                "matched_parent_id": int(r.get("matched_parent_id", -1)),
                                "start_frame": int(r.get("start_frame", start_fi)),
                                "candidate_frames": int(r.get("candidate_frames", 0) or 0),
                                "candidate_age": int(r.get("candidate_age", 0) or 0),
                                "required_frames": int((r.get("confirmed_tail") or {}).get("required_frames", 0) or 0),
                                "candidate_score": float(r.get("candidate_score", 0.0) or 0.0),
                                "verify_reason": str(r.get("verify_reason") or ""),
                                "matched_parent_tail_now": int((r.get("confirmed_tail") or {}).get("matched_parent_tail_now", 0) or 0),
                                "matched_parent_max_tail": int((r.get("confirmed_tail") or {}).get("matched_parent_max_tail", 0) or 0),
                            }
                            for r in list(selected_rows)
                        ],
                    }
                    split_ev = self._apply_split_confirm_existing_children(
                        parent_id=int(selected_parent),
                        child_ids=list(selected_child_ids),
                        reason=str(cluster_reason),
                        score=float(cluster_score),
                        frame_idx=int(self.frame_idx),
                        tracked_masks=tracked_masks,
                        tracked_boxes=tracked_boxes,
                        object_masks=object_masks,
                        object_boxes=object_boxes,
                        new_objects=new_objects,
                        confirmed_tail=cluster_confirmed_tail,
                    )
                    if split_ev is None:
                        reject_events.append({
                            "kind": "object",
                            "stage": "split_confirm_cluster",
                            "decision": "apply_failed",
                            "reason": "cluster_apply_returned_none",
                            "frame_idx": int(self.frame_idx),
                            "start_frame": int(start_fi),
                            "seed_parent_id": int(seed_parent_i),
                            "matched_parent_id": int(selected_parent),
                            "cluster_children": [int(x) for x in cluster_children],
                            "selected_children": [int(x) for x in selected_child_ids],
                            "rows": [
                                {
                                    "child_id": int(r.get("child_id", -1)),
                                    "matched_parent_id": int(r.get("matched_parent_id", -1)),
                                    "child_det_idx": int(r.get("child_det_idx", -1)),
                                    "child_det_score": float(r.get("child_det_score", 0.0)),
                                    "verify_reason": str(r.get("verify_reason", "")),
                                }
                                for r in list(rows)
                            ],
                        })
                        continue

                    drop_children = sorted(set(cluster_children) - set(selected_child_ids))
                    for drop_cid in list(drop_children):
                        self._drop_object_id(
                            obj_id=int(drop_cid),
                            frame_idx=int(self.frame_idx),
                            tracked_masks=tracked_masks,
                            tracked_boxes=tracked_boxes,
                            object_masks=object_masks,
                            object_boxes=object_boxes,
                            new_objects=new_objects,
                            reason="split_cluster_non_selected_child",
                        )

                    for cid in list(selected_child_ids):
                        pair_key_final = (min(int(selected_parent), int(cid)), max(int(selected_parent), int(cid)))
                        self._emitted_struct_split_pairs.add(pair_key_final)

                    split_confirm_emitted_rows.append({
                        "frame_idx": int(self.frame_idx),
                        "confirm_mode": "cluster_existing_children",
                        "parent_id": int(selected_parent),
                        "child_ids": [int(x) for x in selected_child_ids],
                        "start_frame": int(start_fi),
                        "decision_delay_frames": int(max(0, int(self.frame_idx) - int(start_fi))),
                        "required_children": 2,
                        "score": float(cluster_score),
                        "reason": str(cluster_reason),
                        "emitted_children": [int(x) for x in list(split_ev.get("children", []) or []) if int(x) >= 100],
                    })
                    self._pending_split_from_new = {
                        int(k): v
                        for k, v in list(self._pending_split_from_new.items())
                        if _cluster_key_from_candidate(v) != (int(start_fi), int(seed_parent_i))
                    }
                    for cid in list(cluster_children):
                        cluster_consumed_children.add(int(cid))

                    reject_events.append({
                        "kind": "object",
                        "stage": "split_confirm_cluster",
                        "decision": "split",
                        "reason": "cluster_confirmed",
                        "frame_idx": int(self.frame_idx),
                        "start_frame": int(start_fi),
                        "seed_parent_id": int(seed_parent_i),
                        "matched_parent_id": int(selected_parent),
                        "cluster_children": [int(x) for x in cluster_children],
                        "selected_children": [int(x) for x in selected_child_ids],
                        "dropped_children": [int(x) for x in drop_children],
                    })
                    print(
                        f"[Hotrack] F{self.frame_idx} OBJ_SPLIT_CLUSTER_CONFIRMED: "
                        f"parent=obj-{selected_parent-100} "
                        f"children={[int(x)-100 for x in selected_child_ids]} "
                        f"drop={[int(x)-100 for x in drop_children]} "
                        f"cluster=({start_fi},{seed_parent_i})"
                    )
                    continue

                overlap_parent_i = -1
                if parent_groups:
                    overlap_parent_i = int(max(parent_groups.items(), key=lambda it: len(it[1]))[0])
                overlap_parent_mask = _to_bool_mask(tracked_masks.get(int(overlap_parent_i)))
                dropped_children: List[int] = []
                kept_children: List[int] = []
                for row in list(rows):
                    cid = int(row.get("child_id", -1))
                    if int(cid) < 100:
                        continue
                    drop_child = False
                    if (
                        overlap_parent_mask is not None
                        and np.count_nonzero(overlap_parent_mask) > 0
                    ):
                        cm = _to_bool_mask(tracked_masks.get(int(cid)))
                        if cm is not None and np.count_nonzero(cm) > 0:
                            _, ioa_p_to_c, ioa_c_to_p = calculate_ioa_bidirectional(overlap_parent_mask, cm)
                            if float(max(ioa_p_to_c, ioa_c_to_p)) >= float(overlap_drop_th):
                                drop_child = True
                    if drop_child:
                        self._drop_object_id(
                            obj_id=int(cid),
                            frame_idx=int(self.frame_idx),
                            tracked_masks=tracked_masks,
                            tracked_boxes=tracked_boxes,
                            object_masks=object_masks,
                            object_boxes=object_boxes,
                            new_objects=new_objects,
                            reason="split_cluster_fail_overlap_with_parent",
                        )
                        dropped_children.append(int(cid))
                    else:
                        kept_children.append(int(cid))
                    self._pending_split_from_new.pop(int(cid), None)
                    cluster_consumed_children.add(int(cid))

                reject_events.append({
                    "kind": "object",
                    "stage": "split_confirm_cluster",
                    "decision": "no_split",
                    "reason": "cluster_not_confirmed",
                    "frame_idx": int(self.frame_idx),
                    "start_frame": int(start_fi),
                    "seed_parent_id": int(seed_parent_i),
                    "matched_parent_candidates": [int(x) for x in sorted(parent_groups.keys())],
                    "cluster_children": [int(x) for x in cluster_children],
                    "kept_children": [int(x) for x in kept_children],
                    "dropped_children": [int(x) for x in dropped_children],
                    "rows": [
                        {
                            "child_id": int(r.get("child_id", -1)),
                            "matched_parent_id": int(r.get("matched_parent_id", -1)),
                            "child_det_idx": int(r.get("child_det_idx", -1)),
                            "child_det_score": float(r.get("child_det_score", 0.0)),
                            "verify_reason": str(r.get("verify_reason", "")),
                        }
                        for r in list(rows)
                    ],
                })

            # Single-child (or non-cluster) pass path: keep previous behavior.
            for row in list(split_confirm_pass_rows):
                child_i = int(row.get("child_id", -1))
                parent_i = int(row.get("matched_parent_id", row.get("parent_id", -1)))
                if int(child_i) in cluster_consumed_children:
                    continue
                if bool(row.get("in_multi_det_parent_group", False)):
                    continue
                if int(child_i) not in self._pending_split_from_new:
                    continue
                split_ev = self._apply_split_reid_from_candidate(
                    parent_id=int(parent_i),
                    child_id=int(child_i),
                    reason=str(row.get("split_reason", "split_from_new_object:ok")),
                    score=float(row.get("candidate_score", 0.0)),
                    frame_idx=int(self.frame_idx),
                    tracked_masks=tracked_masks,
                    tracked_boxes=tracked_boxes,
                    object_masks=object_masks,
                    object_boxes=object_boxes,
                    new_objects=new_objects,
                    confirmed_tail=(dict(row.get("confirmed_tail")) if isinstance(row.get("confirmed_tail"), dict) else None),
                )
                if split_ev is None:
                    reject_events.append({
                        "kind": "object",
                        "stage": "split_confirm_apply_gate",
                        "reason": "apply_split_reid_returned_none",
                        "parent_id": int(parent_i),
                        "child_id": int(child_i),
                        "split_reason": str(row.get("split_reason", "")),
                        "cache_hit": bool(row.get("cache_hit", False)),
                        "verify_reason": str(row.get("verify_reason", "")),
                    })
                    continue

                self._pending_split_from_new = {
                    int(k): v
                    for k, v in list(self._pending_split_from_new.items())
                    if int(k) != int(child_i) and int((v or {}).get("parent_id", -1)) != int(parent_i)
                }
                pair_key_final = (min(int(parent_i), int(child_i)), max(int(parent_i), int(child_i)))
                self._emitted_struct_split_pairs.add(pair_key_final)
                new_child_ids = [int(x) for x in list(split_ev.get("children", []) or []) if int(x) >= 100]
                confirmed_tail_row = dict(row.get("confirmed_tail")) if isinstance(row.get("confirmed_tail"), dict) else {}
                split_confirm_emitted_rows.append({
                    "frame_idx": int(self.frame_idx),
                    "confirm_mode": str(confirmed_tail_row.get("confirm_mode") or "replay_verify"),
                    "parent_id": int(parent_i),
                    "child_id": int(child_i),
                    "start_frame": int(row.get("start_frame", self.frame_idx)),
                    "decision_delay_frames": int(max(0, int(self.frame_idx) - int(row.get("start_frame", self.frame_idx)))),
                    "required_frames": int(confirmed_tail_row.get("required_frames", 0) or 0),
                    "candidate_frames": int(confirmed_tail_row.get("candidate_frames", 0) or 0),
                    "candidate_age": int(confirmed_tail_row.get("candidate_age", 0) or 0),
                    "score": float(row.get("candidate_score", 0.0) or 0.0),
                    "reason": str(row.get("split_reason", "")),
                    "verify_reason": str(row.get("verify_reason", "")),
                    "emitted_children": [int(x) for x in new_child_ids],
                })
                print(
                    f"[Hotrack] F{self.frame_idx} OBJ_SPLIT_CONFIRMED_FROM_NEW: "
                    f"parent=obj-{parent_i-100} child=obj-{child_i-100} "
                    f"-> new={[int(x)-100 for x in new_child_ids]} reason={str(row.get('split_reason', ''))}"
                )

        # 7.3 Geometry summary per id
        geom: Dict[int, Dict[str, Any]] = {}
        for obj_id, mask in tracked_masks.items():
            bb = tracked_boxes.get(obj_id)
            if mask is not None and np.count_nonzero(mask) > 0:
                ys, xs = np.where(mask)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(mask.sum())
            elif bb is not None:
                x1, y1, x2, y2 = bb
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                area = int(max(0, (x2 - x1 + 1) * (y2 - y1 + 1)))
            else:
                continue
            geom[int(obj_id)] = {
                "centroid": (float(cx), float(cy)),
                "area": int(area),
                "bbox": [int(x) for x in (bb if bb is not None else [0, 0, 0, 0])],
            }

        # 7.35 Remove static hands (likely misclassified objects)
        if bg_motion_ok and A_bg is not None:
            t_bg = np.array([float(A_bg[0, 2]), float(A_bg[1, 2])], dtype=np.float32)
        else:
            t_bg = None
        if t_bg is not None:
            hands_to_remove = []
            for hid in [i for i in self.all_ids if i < 100]:
                if hid not in geom:
                    continue
                cx, cy = geom[hid]["centroid"]
                prev = self.prev_hand_centroids.get(hid)
                if prev is None:
                    self.prev_hand_centroids[hid] = (cx, cy)
                    self.hand_static_counts[hid] = 0
                    continue
                delta = np.array([cx - prev[0], cy - prev[1]], dtype=np.float32)
                res = delta - t_bg
                res_norm = float(np.linalg.norm(res))
                if res_norm < self.hand_static_threshold:
                    self.hand_static_counts[hid] = self.hand_static_counts.get(hid, 0) + 1
                else:
                    self.hand_static_counts[hid] = 0
                self.prev_hand_centroids[hid] = (cx, cy)
                if self.hand_static_counts[hid] >= self.hand_static_frames:
                    hands_to_remove.append(hid)
            for hid in hands_to_remove:
                self.all_ids, _ = self.sam2_tracker.remove_object(
                    self.inference_state, hid, strict=False, need_output=False
                )
                if hid in tracked_masks:
                    del tracked_masks[hid]
                if hid in tracked_boxes:
                    del tracked_boxes[hid]
                if hid in self.mask_history:
                    del self.mask_history[hid]
                if hid in geom:
                    del geom[hid]
                self.hand_static_counts.pop(hid, None)
                self.prev_hand_centroids.pop(hid, None)
                print(f"[Hotrack] F{self.frame_idx} EVENT_REMOVE: hand-{hid} static_like_object")

        # 7.4 Per-object motion (optical flow) for tracking JSON
        motion: Dict[int, Dict[str, Any]] = {}
        for obj_id, mask in tracked_masks.items():
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            est = self._estimate_component_motion(prev_gray, gray, mask, A_bg, bool(bg_motion_ok))
            if est is None:
                motion[int(obj_id)] = {"t": [0.0, 0.0], "w": 0.0, "ok": False}
            else:
                t, w = est
                motion[int(obj_id)] = {"t": [float(t[0]), float(t[1])], "w": float(w), "ok": True}

        # Cache per-frame masks/boxes/geom for backfill visualization
        if self.backfill_window > 0:
            self._tracked_masks_history.append({
                int(k): (v.copy() if v is not None else v) for k, v in tracked_masks.items()
            })
            self._target_boxes_history.append([
                [int(v) for v in b]
                for b in list(target_boxes or [])
                if isinstance(b, (list, tuple)) and len(b) == 4
            ])
            self._target_masks_history.append([
                m.copy()
                for m in list(self._current_target_masks or [])
                if m is not None and np.count_nonzero(m) > 0
            ])
            if self.backfill_debug:
                self._tracked_boxes_history.append({
                    int(k): (list(v) if v is not None else v) for k, v in tracked_boxes.items()
                })
                self._geom_history.append({
                    int(k): dict(v) for k, v in geom.items()
                })
                self._bg_history.append({
                    "A_bg": None if A_bg is None else A_bg.copy(),
                    "bg_ok": bool(bg_motion_ok),
                })

        # 7.5 Backfill masks for surviving new objects (optional, reverse tracking)
        surviving_new_object_ids: List[int] = []
        seen_new_ids: Set[int] = set()
        for new_obj in new_objects:
            obj_id_val = new_obj.get("obj_id") if isinstance(new_obj, dict) else None
            if obj_id_val is None:
                continue
            try:
                obj_id = int(obj_id_val)
            except (TypeError, ValueError):
                continue
            if obj_id in seen_new_ids:
                continue
            if obj_id not in self.all_ids:
                continue
            mask = tracked_masks.get(obj_id)
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            surviving_new_object_ids.append(int(obj_id))
            seen_new_ids.add(int(obj_id))

        backfill_masks: Dict[int, Dict[int, np.ndarray]] = {}
        backfill_frame_indices: List[int] = list(self._frame_idx_history)
        attach_proxy_signal = self._attach_proxy_signal_from_boxes(
            target_boxes=target_boxes,
            target_masks=list(self._current_target_masks or []),
            tracked_masks=tracked_masks,
            object_ids=[i for i in self.all_ids if int(i) >= 100],
        )
        attach_proxy_backfill: Dict[str, Any] = {
            "frame_indices": [],
            "proxies": [],
            "signal": attach_proxy_signal,
            "signal_history": [],
            "start_seed_frame_idx": None,
            "start_seed_signal_boxes": [],
            "proxies_start_seed": [],
            "start_seed_replay": [],
            "pair_signal_tail": [],
            "pair_persist_required": int(max(1, int(self.attach_proxy_signal_persist_frames))),
            "attach_sessions": [],
        }
        if self.backfill_window > 1 and len(self._frame_history) >= 2 and len(surviving_new_object_ids) > 0:
            frames_list = list(self._frame_history)
            max_backfill = min(self.backfill_window - 1, len(frames_list) - 1)
            # Detach backfill only needs new-object trajectories.
            seed_object_ids: Set[int] = set(int(x) for x in surviving_new_object_ids)
            seed_masks: Dict[int, np.ndarray] = {}
            for obj_id in sorted(seed_object_ids):
                mask = tracked_masks.get(int(obj_id))
                if mask is None or np.count_nonzero(mask) == 0:
                    continue
                seed_masks[int(obj_id)] = mask
            if seed_masks:
                if self.backfill_debug:
                    for sid, smask in seed_masks.items():
                        seed_name = f"backfill_seed_f{self.frame_idx:06d}_obj{sid-100}.png"
                        seed_path = os.path.join(self.backfill_debug_dir, seed_name)
                        cv2.imwrite(seed_path, (smask.astype(np.uint8) * 255))
                masks_by_id = self.backfill_track_backward(
                    frames_list,
                    seed_masks,
                    max_backfill=max_backfill,
                )
                if masks_by_id:
                    any_list = next(iter(masks_by_id.values()))
                    frame_indices = backfill_frame_indices[-len(any_list):]
                    for sid, masks_list in masks_by_id.items():
                        if int(sid) < 100:
                            continue
                        backfill_masks[int(sid)] = {
                            int(fi): m for fi, m in zip(frame_indices, masks_list) if m is not None
                        }
                    if self.backfill_debug:
                        k_count = len(frame_indices)
                        idx_map = {int(fi): idx for idx, fi in enumerate(backfill_frame_indices)}
                        palette = [
                            (0, 255, 0),
                            (255, 0, 0),
                            (0, 0, 255),
                            (255, 255, 0),
                            (255, 0, 255),
                            (0, 255, 255),
                            (255, 128, 0),
                            (128, 0, 255),
                        ]
                        for fi in frame_indices:
                            frame_idx = int(fi)
                            hist_idx = idx_map.get(frame_idx)
                            if hist_idx is None or hist_idx >= len(frames_list):
                                continue
                            base = frames_list[hist_idx].copy()
                            frame_geom = {}
                            frame_bg = {"A_bg": None, "bg_ok": False}
                            if hist_idx < len(self._geom_history):
                                frame_geom = self._geom_history[hist_idx] or {}
                            if hist_idx < len(self._bg_history):
                                frame_bg = self._bg_history[hist_idx] or frame_bg

                            masks_list_vis = []
                            boxes_list_vis = []
                            labels_list = []
                            colors_list = []
                            for sid, masks_list in masks_by_id.items():
                                if int(sid) < 100:
                                    continue
                                if hist_idx >= len(masks_list):
                                    continue
                                m = masks_list[hist_idx]
                                if m is None or np.count_nonzero(m) == 0:
                                    continue
                                masks_list_vis.append(m)
                                boxes_list_vis.append(None)
                                labels_list.append(f"obj-{int(sid)-100}")
                                colors_list.append(palette[int(sid) % len(palette)])

                            name = f"backfill_f{self.frame_idx:06d}_k{k_count}_t{int(fi):06d}.png"
                            path = os.path.join(self.backfill_debug_dir, name)
                            if not masks_list_vis:
                                cv2.imwrite(path, base)
                                continue

                            vis = visualize_segmentation(
                                base,
                                masks_list_vis,
                                boxes_list_vis,
                                labels=labels_list,
                                colors=colors_list,
                                outline_only=True,
                            )
                            geom_vis = {int(k): dict(v) for k, v in frame_geom.items()}
                            overlay_motion_debug(
                                vis,
                                geom_vis,
                                {},
                                frame_bg.get("A_bg"),
                                bool(frame_bg.get("bg_ok")),
                            )
                            cv2.imwrite(path, vis)

        # 7.6 Attach replay sessions (tail-end trigger + reverse-to-earliest-AB + forward inclusion count)
        if (
            bool(self.attach_proxy_backfill_enabled)
            and self.backfill_window > 1
            and len(self._frame_history) >= 2
        ):
            replay_candidate_pairs = 0
            replay_qualified_pairs = 0
            replay_cache_hits = 0
            replay_cache_misses = 0
            replay_exec_count = 0
            replay_exec_time_ms = 0.0
            frames_list = list(self._frame_history)
            frame_indices_all = [int(x) for x in list(self._frame_idx_history)]
            idx_by_frame = {int(fi): int(idx) for idx, fi in enumerate(frame_indices_all)}
            target_boxes_hist_all = list(self._target_boxes_history)
            target_masks_hist_all = list(self._target_masks_history)
            tracked_masks_hist_all = list(self._tracked_masks_history)
            frames_by_index_all: Dict[int, np.ndarray] = {
                int(fi): frames_list[idx]
                for idx, fi in enumerate(frame_indices_all)
                if idx < len(frames_list)
            }
            target_boxes_by_frame_all: Dict[int, List[List[int]]] = {}
            for jj, fi in enumerate(frame_indices_all):
                boxes_j = []
                if jj < len(target_boxes_hist_all):
                    boxes_j = list(target_boxes_hist_all[jj] or [])
                cleaned_j: List[List[int]] = []
                seen_j: Set[Tuple[int, int, int, int]] = set()
                for box_j in boxes_j:
                    if not isinstance(box_j, (list, tuple)) or len(box_j) != 4:
                        continue
                    key_j = (int(box_j[0]), int(box_j[1]), int(box_j[2]), int(box_j[3]))
                    if key_j in seen_j:
                        continue
                    seen_j.add(key_j)
                    cleaned_j.append([int(v) for v in box_j])
                target_boxes_by_frame_all[int(fi)] = cleaned_j
            tracked_masks_by_frame_all: Dict[int, Dict[int, np.ndarray]] = {}
            for jj, ffi in enumerate(frame_indices_all):
                if jj < len(tracked_masks_hist_all):
                    tracked_masks_by_frame_all[int(ffi)] = dict(tracked_masks_hist_all[jj] or {})

            signal_hist_rows: List[Dict[str, Any]] = []
            for hist_idx, fi in enumerate(frame_indices_all):
                boxes_h = []
                det_masks_h: List[np.ndarray] = []
                if hist_idx < len(target_boxes_hist_all):
                    boxes_h = list(target_boxes_hist_all[hist_idx] or [])
                if hist_idx < len(target_masks_hist_all):
                    det_masks_h = list(target_masks_hist_all[hist_idx] or [])
                tracked_masks_h = {}
                if hist_idx < len(tracked_masks_hist_all):
                    tracked_masks_h = dict(tracked_masks_hist_all[hist_idx] or {})
                object_ids_h = [
                    int(oid)
                    for oid, mm in tracked_masks_h.items()
                    if int(oid) >= 100 and mm is not None and np.count_nonzero(mm) > 0
                ]
                sig_h = self._attach_proxy_signal_from_boxes(
                    target_boxes=boxes_h,
                    target_masks=det_masks_h,
                    tracked_masks=tracked_masks_h,
                    object_ids=object_ids_h,
                )
                signal_hist_rows.append({
                    "frame_idx": int(fi),
                    "signal": bool(sig_h.get("signal", False)),
                    "mode": str(sig_h.get("mode") or "unknown"),
                    "fallback_used": bool(str(sig_h.get("mode") or "") == "det_box_fallback"),
                    "signal_boxes": [[int(v) for v in b] for b in list(sig_h.get("signal_boxes", []) or [])],
                    "signal_pairs": [[int(p[0]), int(p[1])] for p in list(sig_h.get("signal_pairs", []) or []) if isinstance(p, (list, tuple)) and len(p) >= 2],
                    "box_rows": list(sig_h.get("box_rows", []) or []),
                })
            attach_proxy_backfill["frame_indices"] = [int(fi) for fi in frame_indices_all]
            attach_proxy_backfill["signal_history"] = signal_hist_rows

            pair_tail_rows_live: List[Dict[str, Any]] = []
            pair_req = int(max(1, int(self.attach_proxy_signal_persist_frames)))
            if signal_hist_rows:
                last_pairs = [
                    (min(int(p[0]), int(p[1])), max(int(p[0]), int(p[1])))
                    for p in list(signal_hist_rows[-1].get("signal_pairs", []) or [])
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]
                for pair in sorted(set(last_pairs)):
                    # Strict consecutive requirement:
                    # the SAME pair must be present in strictly consecutive frame indices.
                    tail = 0
                    tail_indices: List[int] = []
                    prev_fi: Optional[int] = None
                    for ridx in range(len(signal_hist_rows) - 1, -1, -1):
                        row = signal_hist_rows[ridx]
                        row_pairs = {
                            (min(int(pp[0]), int(pp[1])), max(int(pp[0]), int(pp[1])))
                            for pp in list(row.get("signal_pairs", []) or [])
                            if isinstance(pp, (list, tuple)) and len(pp) >= 2
                        }
                        if pair not in row_pairs:
                            break
                        try:
                            fi = int(row.get("frame_idx", -1))
                        except Exception:
                            fi = -1
                        if fi < 0:
                            break
                        if prev_fi is not None and int(fi) != int(prev_fi) - 1:
                            break
                        tail += 1
                        tail_indices.append(int(ridx))
                        prev_fi = int(fi)
                    if tail <= 0:
                        continue
                    candidate_start_idx = int(min(tail_indices))
                    candidate_start_frame_idx = int(
                        signal_hist_rows[candidate_start_idx].get("frame_idx", frame_indices_all[0])
                    )
                    replay_offset = int(max(0, int(getattr(self, "attach_replay_start_offset_frames", 2))))
                    # Tail-start is the replay anchor after tail ends.
                    start_idx = int(candidate_start_idx)
                    start_frame_idx = int(candidate_start_frame_idx)
                    qualified = bool(tail >= pair_req)
                    if qualified:
                        if pair not in self._attach_pair_locked_start:
                            self._attach_pair_locked_start[pair] = int(start_frame_idx)
                        else:
                            start_frame_idx = int(self._attach_pair_locked_start[pair])
                            if int(start_frame_idx) in idx_by_frame:
                                start_idx = int(idx_by_frame[int(start_frame_idx)])
                    tail_indices_sorted = sorted(int(x) for x in tail_indices)
                    hit_frames = [
                        int(signal_hist_rows[j].get("frame_idx", -1))
                        for j in tail_indices_sorted
                    ]
                    hit_modes = [
                        str(signal_hist_rows[j].get("mode") or "unknown")
                        for j in tail_indices_sorted
                    ]
                    det_mask_tail_count = int(sum(1 for m in hit_modes if str(m) == "det_mask_reuse"))
                    fallback_tail_count = int(sum(1 for m in hit_modes if str(m) == "det_box_fallback"))
                    pair_tail_rows_live.append({
                        "pair": [int(pair[0]), int(pair[1])],
                        "tail_count": int(tail),
                        "required": int(pair_req),
                        "qualified": bool(qualified),
                        "candidate_start_idx": int(candidate_start_idx),
                        "candidate_start_frame_idx": int(candidate_start_frame_idx),
                        "start_idx": int(start_idx),
                        "start_frame_idx": int(start_frame_idx),
                        "replay_start_offset_frames": int(replay_offset),
                        "hit_frames": hit_frames,
                        "mode_history": hit_modes,
                        "det_mask_tail_count": int(det_mask_tail_count),
                        "fallback_tail_count": int(fallback_tail_count),
                        "det_mask_tail_ratio": float(
                            float(det_mask_tail_count) / float(max(1, len(hit_modes)))
                        ),
                        "start_mode": str(signal_hist_rows[start_idx].get("mode") or "unknown"),
                        "fallback_in_tail": bool(fallback_tail_count > 0),
                        "tail_ended": False,
                        "tail_end_frame_idx": None,
                    })

            pair_row_map_live: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for row in pair_tail_rows_live:
                pair_raw = list(row.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                pair_key = (
                    min(int(pair_raw[0]), int(pair_raw[1])),
                    max(int(pair_raw[0]), int(pair_raw[1])),
                )
                pair_row_map_live[pair_key] = row

            ended_tail_rows: List[Dict[str, Any]] = []
            for pair_key, prev_q in list(self._attach_pair_prev_qualified.items()):
                row_now = pair_row_map_live.get(pair_key)
                q_now = bool(row_now.get("qualified", False)) if isinstance(row_now, dict) else False
                if bool(prev_q) and not bool(q_now):
                    last_row = self._attach_pair_last_qualified.get(pair_key)
                    if isinstance(last_row, dict):
                        ended_row = dict(last_row)
                        hit_frames_last = [
                            int(x)
                            for x in list(ended_row.get("hit_frames", []) or [])
                            if x is not None
                        ]
                        tail_end_fi = max(hit_frames_last) if hit_frames_last else None
                        ended_row["qualified"] = True
                        ended_row["tail_ended"] = True
                        ended_row["tail_end_frame_idx"] = (
                            None if tail_end_fi is None else int(tail_end_fi)
                        )
                        ended_row["end_trigger_frame_idx"] = int(self.frame_idx)
                        ended_tail_rows.append(ended_row)
                    self._attach_pair_last_qualified.pop(pair_key, None)
                    self._attach_pair_locked_start.pop(pair_key, None)

            next_prev_qualified: Dict[Tuple[int, int], bool] = {}
            for pair_key, row in pair_row_map_live.items():
                q_now = bool(row.get("qualified", False))
                next_prev_qualified[pair_key] = bool(q_now)
                if q_now:
                    self._attach_pair_last_qualified[pair_key] = dict(row)
            self._attach_pair_prev_qualified = next_prev_qualified

            ended_dedup: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
            for row in ended_tail_rows:
                pair_raw = list(row.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                key = (
                    min(int(pair_raw[0]), int(pair_raw[1])),
                    max(int(pair_raw[0]), int(pair_raw[1])),
                    int(row.get("start_frame_idx", -1)),
                )
                if key not in ended_dedup:
                    ended_dedup[key] = row
            pair_tail_rows_ended = sorted(
                list(ended_dedup.values()),
                key=lambda r: (
                    int((r.get("pair") or [0, 0])[0]),
                    int((r.get("pair") or [0, 0])[1]),
                    int(r.get("start_frame_idx", -1)),
                ),
            )

            live_dedup: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
            for row in pair_tail_rows_live:
                if not isinstance(row, dict):
                    continue
                if not bool(row.get("qualified", False)):
                    continue
                pair_raw = list(row.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                key = (
                    min(int(pair_raw[0]), int(pair_raw[1])),
                    max(int(pair_raw[0]), int(pair_raw[1])),
                    int(row.get("start_frame_idx", -1)),
                )
                if int(key[2]) < 0:
                    continue
                prev_live = live_dedup.get(key)
                if prev_live is None or int(row.get("tail_count", 0) or 0) > int(prev_live.get("tail_count", 0) or 0):
                    row_live = dict(row)
                    row_live["tail_ended"] = False
                    row_live["tail_end_frame_idx"] = None
                    live_dedup[key] = row_live

            pair_tail_rows_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
            for row in list(live_dedup.values()):
                pair_raw = list(row.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                key = (
                    min(int(pair_raw[0]), int(pair_raw[1])),
                    max(int(pair_raw[0]), int(pair_raw[1])),
                    int(row.get("start_frame_idx", -1)),
                )
                if int(key[2]) < 0:
                    continue
                pair_tail_rows_map[key] = dict(row)
            for row in list(pair_tail_rows_ended):
                pair_raw = list(row.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                key = (
                    min(int(pair_raw[0]), int(pair_raw[1])),
                    max(int(pair_raw[0]), int(pair_raw[1])),
                    int(row.get("start_frame_idx", -1)),
                )
                if int(key[2]) < 0:
                    continue
                # Prefer ended row for same key (more stable tail interval closure metadata).
                pair_tail_rows_map[key] = dict(row)

            pair_tail_rows = sorted(
                list(pair_tail_rows_map.values()),
                key=lambda r: (
                    int((r.get("pair") or [0, 0])[0]),
                    int((r.get("pair") or [0, 0])[1]),
                    int(r.get("start_frame_idx", -1)),
                ),
            )

            attach_proxy_backfill["pair_signal_tail"] = pair_tail_rows
            attach_proxy_backfill["pair_signal_tail_live"] = sorted(
                list(live_dedup.values()),
                key=lambda r: (
                    int((r.get("pair") or [0, 0])[0]),
                    int((r.get("pair") or [0, 0])[1]),
                    int(r.get("start_frame_idx", -1)),
                ),
            )
            attach_proxy_backfill["pair_signal_tail_ended"] = list(pair_tail_rows_ended)
            attach_proxy_backfill["pair_persist_required"] = int(pair_req)

            qualified_pairs_live = {
                (min(int(r["pair"][0]), int(r["pair"][1])), max(int(r["pair"][0]), int(r["pair"][1])))
                for r in pair_tail_rows_live
                if isinstance(r, dict) and bool(r.get("qualified", False))
            }
            for pair in list(self._attach_pair_locked_start.keys()):
                if pair not in qualified_pairs_live:
                    self._attach_pair_locked_start.pop(pair, None)

            self._prune_attach_sessions(int(self.frame_idx))

            start_proxies_payload: List[Dict[str, Any]] = []
            replay_rows: List[Dict[str, Any]] = []
            attach_sessions_payload: List[Dict[str, Any]] = []
            start_seed_boxes_all: List[List[int]] = []
            start_seed_frames_all: List[int] = []

            def _append_session_payload(
                srow_in: Dict[str, Any],
                pair_in: Tuple[int, int],
                start_frame_idx_in: int,
            ) -> None:
                srow = dict(srow_in or {})
                attach_sessions_payload.append({
                    "pair": [int(pair_in[0]), int(pair_in[1])],
                    "start_frame_idx": int(start_frame_idx_in),
                    "seed_box": [int(v) for v in list(srow.get("seed_box", []) or [])[:4]],
                    "seed_source": str(srow.get("seed_source") or "seed_box"),
                    "reverse": dict(srow.get("reverse", {}) or {}),
                    "forward": dict(srow.get("forward", {}) or {}),
                    "status": str(srow.get("status") or ""),
                    "cache_hit": bool(srow.get("cache_hit", False)),
                })
                sp = srow.get("_start_proxy")
                if isinstance(sp, dict):
                    start_proxies_payload.append(sp)
                    sb = sp.get("seed_box")
                    if isinstance(sb, (list, tuple)) and len(sb) == 4:
                        start_seed_boxes_all.append([int(v) for v in sb])
                    try:
                        start_seed_frames_all.append(int(sp.get("start_frame_idx", start_frame_idx_in)))
                    except Exception:
                        start_seed_frames_all.append(int(start_frame_idx_in))
                sr = srow.get("_start_replay")
                if isinstance(sr, dict):
                    replay_rows.append(sr)

            for pinfo in pair_tail_rows:
                replay_candidate_pairs += 1
                pair_raw = list(pinfo.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                pair = (min(int(pair_raw[0]), int(pair_raw[1])), max(int(pair_raw[0]), int(pair_raw[1])))
                qualified = bool(pinfo.get("qualified", False))
                if qualified:
                    replay_qualified_pairs += 1
                start_frame_idx = int(pinfo.get("start_frame_idx", -1))
                if int(start_frame_idx) in idx_by_frame:
                    start_idx = int(idx_by_frame[int(start_frame_idx)])
                else:
                    start_idx = int(pinfo.get("start_idx", -1))

                if qualified:
                    locked_start = self._attach_pair_locked_start.get(pair)
                    if locked_start is not None and int(locked_start) >= 0:
                        start_frame_idx = int(locked_start)
                        pinfo["start_frame_idx"] = int(start_frame_idx)
                        if int(start_frame_idx) in idx_by_frame:
                            start_idx = int(idx_by_frame[int(start_frame_idx)])
                            pinfo["start_idx"] = int(start_idx)

                    # Reuse existing replay session for this pair/start and skip heavy replay logic.
                    existing_key = self._attach_session_cache_key(pair, int(start_frame_idx))
                    existing_session = self._attach_sessions.get(existing_key)
                    if isinstance(existing_session, dict):
                        srow_keep = dict(existing_session)
                        # Tail-local replay policy:
                        # for the same pair/start(tail), run replay once and reuse the first result
                        # (passed or failed) while the tail is alive.
                        replay_cache_hits += 1
                        srow_keep["cache_hit"] = True
                        srow_keep["last_seen_frame"] = int(self.frame_idx)
                        self._attach_sessions[existing_key] = srow_keep
                        self._attach_replay_cache[existing_key] = srow_keep
                        _append_session_payload(srow_keep, pair, int(start_frame_idx))
                        continue

                if not (0 <= int(start_idx) < len(signal_hist_rows)):
                    continue

                if qualified:
                    # Replay-start policy (updated):
                    # - seed/replay start search is done only inside the qualified tail.
                    # - scan from tail-end -> tail-start and pick the first single-component A∪B.
                    # - replay starts at that seed frame and proceeds toward tail-end.
                    pinfo["reverse_replay_from_source"] = "seed_single_component_to_tail_end"

                    # Mandatory seed selection rule:
                    # choose seed only inside the detected tail interval.
                    # Scan [tail_end_idx .. tail_start_idx] in decreasing frame order and
                    # pick the first frame where union(A,B) is single-component.
                    seed_union_components = -1
                    seed_union_single_component = False
                    seed_union_frame_idx: Optional[int] = None
                    seed_union_hist_idx: Optional[int] = None
                    seed_mask_replay: Optional[np.ndarray] = None
                    tail_start_idx = int(start_idx)
                    tail_end_idx = int(start_idx)
                    tail_hit_frames = [
                        int(x)
                        for x in list(pinfo.get("hit_frames", []) or [])
                        if x is not None
                    ]
                    if tail_hit_frames:
                        tail_start_frame_idx = int(min(tail_hit_frames))
                        tail_end_frame_idx = int(max(tail_hit_frames))
                    else:
                        tail_start_frame_idx = int(start_frame_idx)
                        tail_end_frame_idx = int(start_frame_idx)
                    if pinfo.get("tail_end_frame_idx") is not None:
                        try:
                            tail_end_frame_idx = int(pinfo.get("tail_end_frame_idx"))
                        except Exception:
                            pass
                    if int(tail_start_frame_idx) in idx_by_frame:
                        tail_start_idx = int(idx_by_frame[int(tail_start_frame_idx)])
                    if int(tail_end_frame_idx) in idx_by_frame:
                        tail_end_idx = int(idx_by_frame[int(tail_end_frame_idx)])
                    if int(tail_end_idx) < int(tail_start_idx):
                        tail_start_idx, tail_end_idx = int(tail_end_idx), int(tail_start_idx)
                    tail_start_idx = int(max(0, min(int(tail_start_idx), len(frame_indices_all) - 1)))
                    tail_end_idx = int(max(0, min(int(tail_end_idx), len(frame_indices_all) - 1)))
                    pinfo["tail_start_frame_idx"] = int(frame_indices_all[tail_start_idx])
                    pinfo["tail_end_frame_idx"] = int(frame_indices_all[tail_end_idx])
                    replay_from_idx = int(tail_start_idx)
                    replay_from_frame_idx = int(frame_indices_all[replay_from_idx])
                    pinfo["reverse_replay_from_frame_idx"] = int(replay_from_frame_idx)
                    pinfo["reverse_replay_from_idx"] = int(replay_from_idx)

                    # Seed-X search range:
                    # tail interval only [tail_start .. tail_end].
                    # If no single-component A∪B exists in this interval, drop this attach pair.
                    seed_search_min_idx = int(tail_start_idx)
                    pinfo["seed_search_min_frame_idx"] = int(frame_indices_all[int(seed_search_min_idx)])
                    seed_det_union_in = 0.0
                    seed_det_in_union = 0.0
                    seed_a_in_det = 0.0
                    seed_b_in_det = 0.0

                    for cand_hist_idx in range(int(tail_end_idx), int(seed_search_min_idx) - 1, -1):
                        fi_cand = int(frame_indices_all[cand_hist_idx])
                        frame_seed_masks_gate = dict(tracked_masks_by_frame_all.get(int(fi_cand), {}) or {})
                        m_a_gate = _to_bool_mask(frame_seed_masks_gate.get(int(pair[0])))
                        m_b_gate = _to_bool_mask(frame_seed_masks_gate.get(int(pair[1])))
                        if (
                            m_a_gate is None or np.count_nonzero(m_a_gate) == 0
                            or m_b_gate is None or np.count_nonzero(m_b_gate) == 0
                        ):
                            continue
                        union_gate = np.logical_or(m_a_gate, m_b_gate).astype(bool)
                        if np.count_nonzero(union_gate) <= 0:
                            continue
                        num_labels_gate, _, _, _ = cv2.connectedComponentsWithStats(
                            union_gate.astype(np.uint8), connectivity=8
                        )
                        seed_union_components = max(0, int(num_labels_gate) - 1)
                        if int(seed_union_components) > 1:
                            continue

                        # Seed must be union(A,B) itself (not detector mask) and single-component.
                        a_in_det = 1.0
                        b_in_det = 1.0
                        union_in_det = 1.0
                        det_in_union = 1.0
                        seed_union_single_component = True
                        seed_union_frame_idx = int(fi_cand)
                        seed_union_hist_idx = int(cand_hist_idx)
                        seed_mask_replay = union_gate.astype(bool).copy()
                        seed_det_union_in = float(union_in_det)
                        seed_det_in_union = float(det_in_union)
                        seed_a_in_det = float(a_in_det)
                        seed_b_in_det = float(b_in_det)
                        break

                    pinfo["seed_union_components"] = int(seed_union_components)
                    pinfo["seed_union_single_component"] = bool(seed_union_single_component)
                    pinfo["seed_union_frame_idx"] = (
                        None if seed_union_frame_idx is None else int(seed_union_frame_idx)
                    )
                    pinfo["seed_a_in_det"] = float(seed_a_in_det)
                    pinfo["seed_b_in_det"] = float(seed_b_in_det)
                    pinfo["seed_union_in_det"] = float(seed_det_union_in)
                    pinfo["seed_det_in_union"] = float(seed_det_in_union)
                    if not bool(seed_union_single_component) or seed_union_hist_idx is None or seed_mask_replay is None:
                        pinfo["blocked_reason"] = "no_single_component_union_seed_in_tail"
                        pinfo["qualified"] = False
                        continue

                    pinfo["blocked_reason"] = ""
                    pinfo["reverse_seed_frame_idx"] = int(seed_union_frame_idx)
                    pinfo["reverse_seed_idx"] = int(seed_union_hist_idx)
                    # Effective reverse replay start is the seed frame (single-component A∪B frame).
                    replay_from_idx = int(seed_union_hist_idx)
                    replay_from_frame_idx = int(frame_indices_all[replay_from_idx])
                    # Reverse replay must move to the past (smaller frame numbers).
                    # Use the earliest available history frame as lower bound so that
                    # Y can be discovered before tail start.
                    replay_to_idx = 0
                    replay_to_frame_idx = int(frame_indices_all[0])
                    if int(replay_to_idx) > int(replay_from_idx):
                        replay_to_idx = int(replay_from_idx)
                        replay_to_frame_idx = int(frame_indices_all[replay_to_idx])
                    pinfo["reverse_replay_from_frame_idx"] = int(replay_from_frame_idx)
                    pinfo["reverse_replay_from_idx"] = int(replay_from_idx)
                    pinfo["reverse_replay_to_frame_idx"] = int(replay_to_frame_idx)
                    pinfo["reverse_replay_to_idx"] = int(replay_to_idx)

                    frame_for_seed = (
                        frames_list[int(seed_union_hist_idx)]
                        if 0 <= int(seed_union_hist_idx) < len(frames_list)
                        else None
                    )
                    img_area_seed = 0.0
                    if frame_for_seed is not None:
                        try:
                            hh_seed, ww_seed = frame_for_seed.shape[:2]
                            img_area_seed = float(max(1, int(hh_seed) * int(ww_seed)))
                        except Exception:
                            img_area_seed = 0.0
                    tracked_seed_masks = dict(tracked_masks_by_frame_all.get(int(seed_union_frame_idx), {}) or {})
                    hand_area_ref_seed = 0.0
                    if tracked_seed_masks:
                        hand_areas_seed = [
                            float(np.count_nonzero(_to_bool_mask(mm)))
                            for oid, mm in tracked_seed_masks.items()
                            if int(oid) < 100 and _to_bool_mask(mm) is not None and np.count_nonzero(_to_bool_mask(mm)) > 0
                        ]
                        if hand_areas_seed:
                            hand_area_ref_seed = float(max(hand_areas_seed))

                    seed_box_from_union = self._get_mask_bbox(seed_mask_replay)
                    if (
                        not isinstance(seed_box_from_union, (list, tuple))
                        or len(seed_box_from_union) != 4
                    ):
                        pinfo["blocked_reason"] = "seed_union_bbox_missing"
                        pinfo["qualified"] = False
                        continue
                    seed_box_union = [int(v) for v in seed_box_from_union]
                    box_area_seed = float(
                        max(0, int(seed_box_union[2]) - int(seed_box_union[0]))
                        * max(0, int(seed_box_union[3]) - int(seed_box_union[1]))
                    )
                    if hand_area_ref_seed > 0.0:
                        ratio_hand_seed = float(box_area_seed / max(hand_area_ref_seed, 1.0))
                        if ratio_hand_seed > float(self.max_obj_vs_hand_ratio):
                            pinfo["blocked_reason"] = "seed_union_box_too_large_vs_hand"
                            pinfo["qualified"] = False
                            continue
                    if img_area_seed > 0.0:
                        ratio_img_seed = float(box_area_seed / img_area_seed)
                        if ratio_img_seed > float(self.max_obj_area_ratio):
                            pinfo["blocked_reason"] = "seed_union_box_too_large_vs_image"
                            pinfo["qualified"] = False
                            continue

                    chosen_seed_boxes = [seed_box_union]

                    # Build reverse replay input range as [older .. seed] (ascending).
                    # backfill_replay_backward_from_seed_box() internally reverses input,
                    # so execution runs from seed toward older frames.
                    frames_start_asc: List[np.ndarray] = []
                    frame_indices_start_asc: List[int] = []
                    for global_j in range(int(replay_to_idx), int(replay_from_idx) + 1):
                        if 0 <= int(global_j) < len(frames_list):
                            frames_start_asc.append(frames_list[int(global_j)])
                            frame_indices_start_asc.append(int(frame_indices_all[int(global_j)]))
                    if not frames_start_asc or not frame_indices_start_asc:
                        continue
                    max_backfill_start = int(max(0, len(frames_start_asc) - 1))
                    max_steps_start = int(max(1, len(frames_start_asc) - 1))
                    target_boxes_start_asc: List[List[List[int]]] = []
                    for global_j in range(int(replay_to_idx), int(replay_from_idx) + 1):
                        boxes_j = []
                        if 0 <= int(global_j) < len(target_boxes_hist_all):
                            boxes_j = list(target_boxes_hist_all[global_j] or [])
                        cleaned_j: List[List[int]] = []
                        seen_j: Set[Tuple[int, int, int, int]] = set()
                        for box_j in boxes_j:
                            if not isinstance(box_j, (list, tuple)) or len(box_j) != 4:
                                continue
                            key_j = (int(box_j[0]), int(box_j[1]), int(box_j[2]), int(box_j[3]))
                            if key_j in seen_j:
                                continue
                            seen_j.add(key_j)
                            cleaned_j.append([int(v) for v in box_j])
                        target_boxes_start_asc.append(cleaned_j)

                    for seed_box in chosen_seed_boxes:
                        if not frames_start_asc:
                            continue
                        seed_box_replay = [int(v) for v in seed_box]
                        if (
                            int(seed_box_replay[2]) <= int(seed_box_replay[0])
                            or int(seed_box_replay[3]) <= int(seed_box_replay[1])
                        ):
                            continue

                        session_key = self._attach_session_cache_key(pair, int(start_frame_idx))
                        cached = self._attach_replay_cache.get(session_key)
                        cache_hit = False
                        if isinstance(cached, dict):
                            # Tail-local replay policy:
                            # cached session for same pair/start is always reusable.
                            cache_hit = True
                        if cache_hit:
                            replay_cache_hits += 1
                            session_row = dict(cached)
                            session_row["cache_hit"] = True
                            # Ensure replay debug artifacts are present even on cache-hit path.
                            if bool(self.attach_replay_debug):
                                try:
                                    seed_box_cached = session_row.get("seed_box")
                                    if not isinstance(seed_box_cached, (list, tuple)) or len(seed_box_cached) != 4:
                                        seed_box_cached = [int(v) for v in seed_box_replay]
                                    else:
                                        seed_box_cached = [int(v) for v in list(seed_box_cached)[:4]]
                                    start_frame_cached = int(session_row.get("start_frame_idx", start_frame_idx))
                                    sx1, sy1, sx2, sy2 = [int(v) for v in seed_box_cached]
                                    session_name_cached = (
                                        f"{int(start_frame_cached):06d}_{int(pair[0])}-{int(pair[1])}"
                                        f"_seed_{int(sx1)}_{int(sy1)}_{int(sx2)}_{int(sy2)}"
                                    )
                                    meta_path_cached = os.path.join(
                                        self.attach_replay_debug_dir,
                                        session_name_cached,
                                        "meta.json",
                                    )
                                    if not os.path.isfile(meta_path_cached):
                                        replay_cached = session_row.get("_start_replay")
                                        reverse_cached = dict(session_row.get("reverse", {}) or {})
                                        if isinstance(replay_cached, dict):
                                            self._save_attach_replay_debug_visuals(
                                                pair=(int(pair[0]), int(pair[1])),
                                                start_frame_idx=int(start_frame_cached),
                                                seed_box=[int(v) for v in seed_box_cached],
                                                replay=replay_cached,
                                                reverse_eval={
                                                    "y_first_frame": (
                                                        int(reverse_cached.get("y_first_frame"))
                                                        if reverse_cached.get("y_first_frame") is not None
                                                        else None
                                                    ),
                                                    "y_within_5": bool(reverse_cached.get("y_within_5", False)),
                                                    "xy_ab_match_pass": bool(reverse_cached.get("xy_ab_match_pass", False)),
                                                    "xy_ab_match_reason": str(reverse_cached.get("xy_ab_match_reason") or ""),
                                                    "xy_ab_match_score": float(reverse_cached.get("xy_ab_match_score", 0.0) or 0.0),
                                                    "xy_ab_match_assignment": str(reverse_cached.get("xy_ab_match_assignment") or ""),
                                                },
                                                forward_replay=None,
                                                session_status=str(session_row.get("status") or ""),
                                                pair_qualified=bool(qualified),
                                                frames_by_index=frames_by_index_all,
                                                target_boxes_by_frame=target_boxes_by_frame_all,
                                                tracked_masks_by_frame=tracked_masks_by_frame_all,
                                            )
                                except Exception:
                                    pass
                        else:
                            replay_cache_misses += 1
                            replay_exec_count += 1
                            miss_eval_t0 = time.perf_counter()
                            self._attach_proxy_id_counter += 1
                            proxy_id_row = int(920000 + self._attach_proxy_id_counter)
                            replay = self.backfill_replay_backward_from_seed_box(
                                frames_start_asc,
                                frame_indices_start_asc,
                                target_boxes_start_asc,
                                [int(v) for v in seed_box_replay],
                                seed_mask_t=seed_mask_replay,
                                proxy_obj_id=int(proxy_id_row),
                                a_id=int(pair[0]),
                                b_id=int(pair[1]),
                                tracked_masks_by_frame=tracked_masks_by_frame_all,
                                max_backfill=max_backfill_start,
                                max_steps=int(max_steps_start),
                                stop_on_first_new_object=False,
                                stop_on_first_matched_new_object=True,
                                post_new_object_extra_steps=int(
                                    max(
                                        0,
                                        int(
                                            getattr(
                                                self,
                                                "attach_reverse_xy_match_post_y_frames",
                                                2,
                                            )
                                        ),
                                    )
                                ),
                            )
                            replay["pair"] = [int(pair[0]), int(pair[1])]
                            replay["start_frame_idx"] = int(start_frame_idx)
                            replay["seed_frame_idx"] = int(seed_union_frame_idx)
                            replay["seed_hist_idx"] = int(seed_union_hist_idx)
                            replay["reverse_min_frame_idx"] = int(min(frame_indices_start_asc))
                            replay["reverse_max_frame_idx"] = int(max(frame_indices_start_asc))

                            y_first_frame = None
                            y_within_5 = False
                            forward_seed_frame = None
                            forward_replay: Optional[Dict[str, Any]] = None
                            xy_match = {
                                "pass": False,
                                "score": 0.0,
                                "threshold": float(self.attach_reverse_xy_match_th),
                                "frame_idx": None,
                                "y_id": None,
                                "assignment": "",
                                "a_in_x": 0.0,
                                "b_in_x": 0.0,
                                "a_in_y": 0.0,
                                "b_in_y": 0.0,
                                "reason": "",
                            }

                            forward = {
                                "window": int(self.attach_forward_window),
                                "count_on": 0,
                                "hit_frames": [],
                                "pass": False,
                                "rows": [],
                                "mode": "y_in_x",
                                "confirm_frame": None,
                            }
                            # Forward confirmation runs from Y-birth frame toward seed frame
                            # (future direction), not toward tail-end.
                            forward_end_idx = int(max(0, min(int(seed_union_hist_idx), len(frame_indices_all) - 1)))
                            forward_end_frame_idx = int(frame_indices_all[forward_end_idx])
                            status = "failed_reverse"
                            if bool(replay.get("ok", False)):
                                proxy_masks = dict(replay.get("proxy_masks_by_frame") or {})
                                new_masks_by_frame = dict(replay.get("new_masks_by_frame") or {})
                                reverse_rows = [
                                    r for r in list(replay.get("rows", []) or []) if isinstance(r, dict)
                                ]
                                candidate_y_rows: List[Tuple[int, int, Dict[str, Any]]] = []
                                reverse_rows_with_idx: List[Tuple[int, Dict[str, Any]]] = []
                                row_pos_by_frame: Dict[int, int] = {}
                                for pos_row, rrow in enumerate(reverse_rows):
                                    try:
                                        fi_row = int(rrow.get("frame_idx", -1))
                                    except Exception:
                                        continue
                                    if fi_row < 0:
                                        continue
                                    reverse_rows_with_idx.append((int(fi_row), rrow))
                                    row_pos_by_frame.setdefault(int(fi_row), int(pos_row))

                                y_birth_rows: List[Tuple[int, int]] = []
                                for fi_row, rrow in reverse_rows_with_idx:
                                    for yid_raw in list(rrow.get("accepted_new_ids", []) or []):
                                        try:
                                            y_birth_rows.append((int(fi_row), int(yid_raw)))
                                        except Exception:
                                            continue

                                if y_birth_rows:
                                    try:
                                        first_birth = min(
                                            y_birth_rows,
                                            key=lambda t: int(row_pos_by_frame.get(int(t[0]), 10**9)),
                                        )
                                        y_first_frame = int(first_birth[0])
                                    except Exception:
                                        pass

                                pair_th_local = float(
                                    xy_match.get("threshold", self.attach_reverse_xy_match_th)
                                    or self.attach_reverse_xy_match_th
                                )
                                post_y_frames = int(
                                    max(
                                        0,
                                        int(
                                            getattr(
                                                self,
                                                "attach_reverse_xy_match_post_y_frames",
                                                2,
                                            )
                                        ),
                                    )
                                )
                                for fi_birth, yid in y_birth_rows:
                                    pos_birth = row_pos_by_frame.get(int(fi_birth))
                                    if pos_birth is None:
                                        continue
                                    end_pos = int(
                                        min(
                                            len(reverse_rows_with_idx) - 1,
                                            int(pos_birth) + int(post_y_frames),
                                        )
                                    )
                                    for pos_chk in range(int(pos_birth), int(end_pos) + 1):
                                        fi_chk, _ = reverse_rows_with_idx[int(pos_chk)]
                                        frame_new_chk = dict(new_masks_by_frame.get(int(fi_chk), {}) or {})
                                        m_y_rel = _to_bool_mask(frame_new_chk.get(int(yid)))
                                        if m_y_rel is None or np.count_nonzero(m_y_rel) == 0:
                                            continue
                                        m_x_rel = _to_bool_mask(proxy_masks.get(int(fi_chk)))
                                        frame_masks_chk = dict(
                                            tracked_masks_by_frame_all.get(int(fi_chk), {}) or {}
                                        )
                                        m_a_rel = _to_bool_mask(frame_masks_chk.get(int(pair[0])))
                                        m_b_rel = _to_bool_mask(frame_masks_chk.get(int(pair[1])))
                                        has_a = bool(m_a_rel is not None and np.count_nonzero(m_a_rel) > 0)
                                        has_b = bool(m_b_rel is not None and np.count_nonzero(m_b_rel) > 0)
                                        if not bool(has_a or has_b):
                                            continue

                                        a_in_x = 0.0
                                        b_in_x = 0.0
                                        if m_x_rel is not None and np.count_nonzero(m_x_rel) > 0:
                                            if has_a:
                                                a_in_x = float(self._mask_in_mask_ratio(m_a_rel, m_x_rel))
                                            if has_b:
                                                b_in_x = float(self._mask_in_mask_ratio(m_b_rel, m_x_rel))

                                        a_in_y = 0.0
                                        b_in_y = 0.0
                                        a_overlap_y = 0.0
                                        b_overlap_y = 0.0
                                        if has_a:
                                            a_in_y = float(self._mask_in_mask_ratio(m_a_rel, m_y_rel))
                                            _, ioa_a_to_y, ioa_y_to_a = calculate_ioa_bidirectional(m_a_rel, m_y_rel)
                                            a_overlap_y = float(max(ioa_a_to_y, ioa_y_to_a))
                                        if has_b:
                                            b_in_y = float(self._mask_in_mask_ratio(m_b_rel, m_y_rel))
                                            _, ioa_b_to_y, ioa_y_to_b = calculate_ioa_bidirectional(m_b_rel, m_y_rel)
                                            b_overlap_y = float(max(ioa_b_to_y, ioa_y_to_b))

                                        pass_a = bool(float(a_overlap_y) >= float(pair_th_local))
                                        pass_b = bool(float(b_overlap_y) >= float(pair_th_local))
                                        if not bool(pass_a or pass_b):
                                            continue

                                        pair_score = float(max(a_overlap_y, b_overlap_y))
                                        assignment = "Y~(A|B)"
                                        if bool(pass_a) and bool(pass_b):
                                            match_mode = "y_overlaps_ab"
                                        else:
                                            match_mode = "y_overlaps_one_of_ab"
                                        candidate_y_rows.append((
                                            int(fi_chk),
                                            int(yid),
                                            {
                                                "cand_id": int(yid),
                                                "pair_score": float(pair_score),
                                                "pair_th": float(pair_th_local),
                                                "assignment": str(assignment),
                                                "match_mode": str(match_mode),
                                                "pass_xy_xA_yB": False,
                                                "pass_yx_xB_yA": False,
                                                "pass_y_overlaps_a": bool(pass_a),
                                                "pass_y_overlaps_b": bool(pass_b),
                                                "a_in_x": float(a_in_x),
                                                "b_in_x": float(b_in_x),
                                                "a_in_y": float(a_in_y),
                                                "b_in_y": float(b_in_y),
                                                "a_overlap_y": float(a_overlap_y),
                                                "b_overlap_y": float(b_overlap_y),
                                                "pass": True,
                                            },
                                        ))

                                if not candidate_y_rows:
                                    xy_match["reason"] = "no_pair_matched_y"
                                    status = "failed_reverse:no_pair_matched_y"
                                else:
                                    for fi_row, yid, eval_row in candidate_y_rows:
                                        frame_new = dict(new_masks_by_frame.get(int(fi_row), {}) or {})
                                        y_seed_mask = _to_bool_mask(frame_new.get(int(yid)))
                                        if y_seed_mask is None or np.count_nonzero(y_seed_mask) == 0:
                                            xy_match["reason"] = "y_mask_missing_for_candidate"
                                            status = "failed_forward:y_seed_missing"
                                            continue

                                        if y_first_frame is None:
                                            y_first_frame = int(fi_row)
                                        xy_pass = bool(eval_row.get("pass", True))
                                        xy_score = float(eval_row.get("pair_score", 0.0) or 0.0)
                                        xy_assignment = str(eval_row.get("assignment") or "")
                                        xy_match.update({
                                            "pass": bool(xy_pass),
                                            "score": float(xy_score),
                                            "frame_idx": int(fi_row),
                                            "y_id": int(yid),
                                            "assignment": str(xy_assignment),
                                            "a_in_x": float(eval_row.get("a_in_x", 0.0) or 0.0),
                                            "b_in_x": float(eval_row.get("b_in_x", 0.0) or 0.0),
                                            "a_in_y": float(eval_row.get("a_in_y", 0.0) or 0.0),
                                            "b_in_y": float(eval_row.get("b_in_y", 0.0) or 0.0),
                                            "reason": "ok" if bool(xy_pass) else "score_below_threshold",
                                        })
                                        if not bool(xy_pass):
                                            status = "failed_reverse:xy_ab_mismatch"
                                            continue

                                        split_idx = idx_by_frame.get(int(fi_row))
                                        if split_idx is None or not (0 <= int(split_idx) < len(frames_list)):
                                            status = "failed_forward:seed_frame_out_of_history"
                                            continue
                                        # Forward confirmation must cover from Y birth frame
                                        # up to the actual union-seed frame selected by backward scan.
                                        if int(split_idx) > int(forward_end_idx):
                                            status = "failed_forward:split_after_forward_end_frame"
                                            continue
                                        frames_forward = frames_list[int(split_idx): int(forward_end_idx) + 1]
                                        frame_indices_forward = frame_indices_all[int(split_idx): int(forward_end_idx) + 1]
                                        if not frames_forward or not frame_indices_forward:
                                            status = "failed_forward:no_frames"
                                            continue
                                        x_masks_forward: Dict[int, np.ndarray] = {}
                                        for ffi in frame_indices_forward:
                                            mm_x = _to_bool_mask(proxy_masks.get(int(ffi)))
                                            if mm_x is not None and np.count_nonzero(mm_x) > 0:
                                                x_masks_forward[int(ffi)] = mm_x
                                        if not x_masks_forward:
                                            status = "failed_forward:x_track_missing"
                                            continue

                                        forward_seed_frame = int(fi_row)
                                        forward_replay = self.replay_forward_y_in_x_from_seed_mask(
                                            frames_forward,
                                            frame_indices_forward,
                                            y_seed_mask,
                                            proxy_obj_id=int(proxy_id_row),
                                            x_masks_by_frame=x_masks_forward,
                                            max_forward=int(max(1, len(frames_forward))),
                                            count_on_th=int(self.attach_forward_count_on_th),
                                        )
                                        if bool(forward_replay.get("ok", False)):
                                            forward_hit_frames = [
                                                int(x) for x in list(forward_replay.get("hit_frames", []) or [])
                                            ]
                                            forward_tail_hit_frames = [
                                                int(x) for x in list(forward_replay.get("tail_hit_frames", []) or [])
                                            ]
                                            confirm_frame = (
                                                int(forward_replay.get("confirm_frame"))
                                                if forward_replay.get("confirm_frame") is not None
                                                else None
                                            )
                                            forward = {
                                                "window": int(forward_replay.get("window", len(frames_forward)) or len(frames_forward)),
                                                "count_on": int(forward_replay.get("count_on", 0) or 0),
                                                "tail_on": int(forward_replay.get("tail_on", 0) or 0),
                                                "max_tail_on": int(forward_replay.get("max_tail_on", 0) or 0),
                                                "hit_frames": forward_hit_frames,
                                                "tail_hit_frames": forward_tail_hit_frames,
                                                "pass": bool(forward_replay.get("pass", False)),
                                                "rows": list(forward_replay.get("rows", []) or []),
                                                "mode": str(forward_replay.get("mode") or "y_in_x"),
                                                "seed_id": int(yid),
                                                "confirm_frame": None if confirm_frame is None else int(confirm_frame),
                                            }
                                            status = "passed" if bool(forward.get("pass", False)) else "failed_forward"
                                        else:
                                            status = f"failed_forward:{str(forward_replay.get('reason') or 'unknown')}"

                                        if str(status) == "passed":
                                            break

                                    y_within_5 = bool(y_first_frame is not None)
                            else:
                                status = f"failed_reverse:{str(replay.get('reason') or 'unknown')}"

                            session_row = {
                                "pair": [int(pair[0]), int(pair[1])],
                                "start_frame_idx": int(start_frame_idx),
                                "seed_frame_idx": int(seed_union_frame_idx),
                                "seed_box": [int(v) for v in seed_box_replay],
                                "seed_box_raw": [int(v) for v in seed_box],
                                "seed_source": str(
                                    replay.get("seed_source")
                                    or ("union_mask" if seed_mask_replay is not None else "seed_box")
                                ),
                                "seed_mask_area": int(replay.get("seed_mask_area", 0) or 0),
                                "seed_mask_bbox": [
                                    int(v)
                                    for v in list(replay.get("seed_mask_bbox", []) or [])[:4]
                                ],
                                "proxy_id": int(proxy_id_row),
                                "reverse": {
                                    "y_first_frame": None if y_first_frame is None else int(y_first_frame),
                                    "y_within_5": bool(y_within_5),
                                    "forward_seed_frame": None if forward_seed_frame is None else int(forward_seed_frame),
                                    "xy_ab_match_pass": bool(xy_match.get("pass", False)),
                                    "xy_ab_match_score": float(xy_match.get("score", 0.0) or 0.0),
                                    "xy_ab_match_th": float(xy_match.get("threshold", self.attach_reverse_xy_match_th) or self.attach_reverse_xy_match_th),
                                    "xy_ab_match_frame": int(xy_match.get("frame_idx")) if xy_match.get("frame_idx") is not None else None,
                                    "xy_ab_match_y_id": int(xy_match.get("y_id")) if xy_match.get("y_id") is not None else None,
                                    "xy_ab_match_assignment": str(xy_match.get("assignment") or ""),
                                    "xy_ab_match_reason": str(xy_match.get("reason") or ""),
                                    "xy_ab_match_a_in_x": float(xy_match.get("a_in_x", 0.0) or 0.0),
                                    "xy_ab_match_b_in_x": float(xy_match.get("b_in_x", 0.0) or 0.0),
                                    "xy_ab_match_a_in_y": float(xy_match.get("a_in_y", 0.0) or 0.0),
                                    "xy_ab_match_b_in_y": float(xy_match.get("b_in_y", 0.0) or 0.0),
                                    "reverse_min_frame_idx": int(min(frame_indices_start_asc)),
                                    "reverse_max_frame_idx": int(max(frame_indices_start_asc)),
                                    "seed_frame_idx": int(seed_union_frame_idx),
                                    "forward_end_frame_idx": int(forward_end_frame_idx),
                                    "rows": [dict(r) for r in list(replay.get("rows", []) or [])],
                                },
                                "forward": {
                                    "window": int(forward.get("window", self.attach_forward_window)),
                                    "count_on": int(forward.get("count_on", 0) or 0),
                                    "hit_frames": [int(x) for x in list(forward.get("hit_frames", []) or [])],
                                    "pass": bool(forward.get("pass", False)),
                                    "mode": str(forward.get("mode", "y_in_x") or "y_in_x"),
                                    "seed_id": int(forward.get("seed_id")) if forward.get("seed_id") is not None else None,
                                    "seed_frame": None if forward_seed_frame is None else int(forward_seed_frame),
                                },
                                "status": str(status),
                                "cache_hit": False,
                                "created_frame_idx": int(self.frame_idx),
                                "last_seen_frame": int(self.frame_idx),
                                "_start_proxy": {
                                    "proxy_id": int(proxy_id_row),
                                    "seed_box": [int(v) for v in seed_box_replay],
                                    "seed_box_raw": [int(v) for v in seed_box],
                                    "seed_source": str(
                                        replay.get("seed_source")
                                        or ("union_mask" if seed_mask_replay is not None else "seed_box")
                                    ),
                                    "seed_mask_area": int(replay.get("seed_mask_area", 0) or 0),
                                    "seed_mask_bbox": [
                                        int(v)
                                        for v in list(replay.get("seed_mask_bbox", []) or [])[:4]
                                    ],
                                    "masks_by_frame": replay.get("proxy_masks_by_frame", {}) or {},
                                    "pair": [int(pair[0]), int(pair[1])],
                                    "start_frame_idx": int(start_frame_idx),
                                    "seed_frame_idx": int(seed_union_frame_idx),
                                },
                                "_start_replay": replay,
                            }
                            self._save_attach_replay_debug_visuals(
                                pair=(int(pair[0]), int(pair[1])),
                                start_frame_idx=int(start_frame_idx),
                                seed_box=[int(v) for v in seed_box_replay],
                                replay=replay,
                                reverse_eval={
                                    "y_first_frame": None if y_first_frame is None else int(y_first_frame),
                                    "y_within_5": bool(y_within_5),
                                    "xy_ab_match_pass": bool(xy_match.get("pass", False)),
                                    "xy_ab_match_reason": str(xy_match.get("reason") or ""),
                                    "xy_ab_match_score": float(xy_match.get("score", 0.0) or 0.0),
                                    "xy_ab_match_assignment": str(xy_match.get("assignment") or ""),
                                },
                                forward_replay=forward_replay,
                                session_status=str(status),
                                pair_qualified=bool(qualified),
                                frames_by_index=frames_by_index_all,
                                target_boxes_by_frame=target_boxes_by_frame_all,
                                tracked_masks_by_frame=tracked_masks_by_frame_all,
                            )
                            replay_exec_time_ms += float(max(0.0, (time.perf_counter() - miss_eval_t0) * 1000.0))
                            self._attach_replay_cache[session_key] = session_row

                        session_row["last_seen_frame"] = int(self.frame_idx)
                        self._attach_sessions[session_key] = session_row

                if qualified:
                    final_key = self._attach_session_cache_key(pair, int(start_frame_idx))
                    srow_final = self._attach_sessions.get(final_key)
                    if isinstance(srow_final, dict):
                        _append_session_payload(srow_final, pair, int(start_frame_idx))

            # Keep only active-qualified lock rows; ended tails release the lock immediately.
            for pair in list(self._attach_pair_locked_start.keys()):
                if pair not in qualified_pairs_live:
                    self._attach_pair_locked_start.pop(pair, None)

            start_seed_frame_idx_val = min(start_seed_frames_all) if start_seed_frames_all else None
            start_seed_signal_boxes_val: List[List[int]] = []
            seen_seed_boxes: Set[Tuple[int, int, int, int]] = set()
            for b in start_seed_boxes_all:
                key_b = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                if key_b in seen_seed_boxes:
                    continue
                seen_seed_boxes.add(key_b)
                start_seed_signal_boxes_val.append([int(v) for v in b])

            attach_proxy_backfill["start_seed_frame_idx"] = (
                int(start_seed_frame_idx_val) if start_seed_frame_idx_val is not None else None
            )
            attach_proxy_backfill["start_seed_signal_boxes"] = start_seed_signal_boxes_val
            attach_proxy_backfill["proxies_start_seed"] = start_proxies_payload
            attach_proxy_backfill["start_seed_replay"] = replay_rows
            attach_proxy_backfill["attach_sessions"] = attach_sessions_payload
            # Merge-session gate diagnostics:
            # for each pair tail row, explain why it can/cannot enter merge candidate stage.
            session_best_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
            proxy_count_by_key: Dict[Tuple[int, int, int], int] = {}

            def _diag_key_from_pair_start(pair_raw: Any, start_raw: Any) -> Optional[Tuple[int, int, int]]:
                if not isinstance(pair_raw, (list, tuple)) or len(pair_raw) < 2:
                    return None
                try:
                    a = int(pair_raw[0])
                    b = int(pair_raw[1])
                    s = int(start_raw)
                except Exception:
                    return None
                if int(a) == int(b) or int(s) < 0:
                    return None
                return (int(min(a, b)), int(max(a, b)), int(s))

            for srow in list(attach_sessions_payload or []):
                key_s = _diag_key_from_pair_start(
                    srow.get("pair"),
                    srow.get("start_frame_idx", -1),
                )
                if key_s is None:
                    continue
                rev_s = srow.get("reverse") if isinstance(srow.get("reverse"), dict) else {}
                fwd_s = srow.get("forward") if isinstance(srow.get("forward"), dict) else {}
                rank_s = (
                    int(fwd_s.get("count_on", 0) or 0),
                    float(rev_s.get("xy_ab_match_score", 0.0) or 0.0),
                )
                prev_s = session_best_by_key.get(key_s)
                if prev_s is None:
                    srow_keep = dict(srow)
                    srow_keep["_rank"] = rank_s
                    session_best_by_key[key_s] = srow_keep
                else:
                    prev_rank = prev_s.get("_rank", (0, 0.0))
                    if rank_s > prev_rank:
                        srow_keep = dict(srow)
                        srow_keep["_rank"] = rank_s
                        session_best_by_key[key_s] = srow_keep

            for prow in list(start_proxies_payload or []):
                key_p = _diag_key_from_pair_start(
                    prow.get("pair"),
                    prow.get("start_frame_idx", -1),
                )
                if key_p is None:
                    continue
                proxy_count_by_key[key_p] = int(proxy_count_by_key.get(key_p, 0) + 1)

            merge_session_gate_rows: List[Dict[str, Any]] = []
            merge_session_reason_counts: Dict[str, int] = {}
            for prow in list(pair_tail_rows or []):
                pair_raw = list(prow.get("pair", []) or [])
                if len(pair_raw) < 2:
                    continue
                pair_norm = [int(min(int(pair_raw[0]), int(pair_raw[1]))), int(max(int(pair_raw[0]), int(pair_raw[1])))]
                try:
                    start_fi = int(prow.get("start_frame_idx", -1))
                except Exception:
                    start_fi = -1
                tail_count_v = int(prow.get("tail_count", 0) or 0)
                required_v = int(prow.get("required", pair_req) or pair_req)
                qualified_v = bool(prow.get("qualified", False))
                blocked_reason_v = str(prow.get("blocked_reason") or "")
                key_m = _diag_key_from_pair_start(pair_norm, start_fi)
                session_row_v = session_best_by_key.get(key_m) if key_m is not None else None
                reverse_v = (
                    session_row_v.get("reverse")
                    if isinstance(session_row_v, dict) and isinstance(session_row_v.get("reverse"), dict)
                    else {}
                )
                forward_v = (
                    session_row_v.get("forward")
                    if isinstance(session_row_v, dict) and isinstance(session_row_v.get("forward"), dict)
                    else {}
                )
                session_status_v = (
                    str(session_row_v.get("status") or "")
                    if isinstance(session_row_v, dict)
                    else ""
                )
                session_status_pass_v = bool(session_status_v.startswith("passed"))
                reverse_pass_v = bool(reverse_v.get("xy_ab_match_pass", False))
                forward_pass_v = bool(forward_v.get("pass", False))

                gate_pass_v = False
                gate_reason_v = ""
                if not bool(qualified_v):
                    gate_reason_v = (
                        f"tail_blocked:{blocked_reason_v}"
                        if blocked_reason_v
                        else "tail_not_qualified"
                    )
                elif not isinstance(session_row_v, dict):
                    gate_reason_v = "no_attach_session_for_pair_start"
                elif not bool(session_status_pass_v):
                    gate_reason_v = (
                        f"session_status:{session_status_v}"
                        if session_status_v
                        else "session_status:empty"
                    )
                elif not bool(reverse_pass_v):
                    gate_reason_v = "reverse_xy_ab_match_failed"
                elif not bool(forward_pass_v):
                    gate_reason_v = "forward_inclusion_failed"
                else:
                    gate_pass_v = True
                    gate_reason_v = "passed"

                merge_session_gate_rows.append({
                    "pair": [int(pair_norm[0]), int(pair_norm[1])],
                    "start_frame_idx": int(start_fi),
                    "tail_count": int(tail_count_v),
                    "required": int(required_v),
                    "qualified": bool(qualified_v),
                    "blocked_reason": str(blocked_reason_v),
                    "hit_frames": [int(x) for x in list(prow.get("hit_frames", []) or [])],
                    "mode_history": [str(x) for x in list(prow.get("mode_history", []) or [])],
                    "det_mask_tail_count": int(prow.get("det_mask_tail_count", 0) or 0),
                    "fallback_tail_count": int(prow.get("fallback_tail_count", 0) or 0),
                    "det_mask_tail_ratio": float(prow.get("det_mask_tail_ratio", 0.0) or 0.0),
                    "start_proxy_count": int(proxy_count_by_key.get(key_m, 0) if key_m is not None else 0),
                    "session_exists": bool(isinstance(session_row_v, dict)),
                    "session_status": str(session_status_v),
                    "session_cache_hit": bool(session_row_v.get("cache_hit", False)) if isinstance(session_row_v, dict) else False,
                    "session_reverse_xy_ab_match_pass": bool(reverse_pass_v),
                    "session_reverse_xy_ab_match_reason": str(reverse_v.get("xy_ab_match_reason") or ""),
                    "session_forward_pass": bool(forward_pass_v),
                    "session_forward_count_on": int(forward_v.get("count_on", 0) or 0),
                    "session_forward_window": int(forward_v.get("window", 0) or 0),
                    "merge_session_gate_pass": bool(gate_pass_v),
                    "merge_session_gate_reason": str(gate_reason_v),
                })
                merge_session_reason_counts[str(gate_reason_v)] = int(
                    merge_session_reason_counts.get(str(gate_reason_v), 0) + 1
                )

            merge_session_gate_rows.sort(
                key=lambda r: (
                    int(r.get("pair", [0, 0])[0]),
                    int(r.get("pair", [0, 0])[1]),
                    int(r.get("start_frame_idx", -1)),
                )
            )
            attach_proxy_backfill["merge_session_gate"] = {
                "required_frames": int(pair_req),
                "rows": merge_session_gate_rows,
                "rows_count": int(len(merge_session_gate_rows)),
                "qualified_rows_count": int(sum(1 for r in merge_session_gate_rows if bool(r.get("qualified", False)))),
                "gate_pass_rows_count": int(sum(1 for r in merge_session_gate_rows if bool(r.get("merge_session_gate_pass", False)))),
                "reason_counts": {str(k): int(v) for k, v in merge_session_reason_counts.items()},
            }
            attach_proxy_backfill["replay_stats"] = {
                "candidate_pairs": int(replay_candidate_pairs),
                "qualified_pairs": int(replay_qualified_pairs),
                "cache_hits": int(replay_cache_hits),
                "cache_misses": int(replay_cache_misses),
                "replay_exec_count": int(replay_exec_count),
                "replay_exec_time_ms": float(replay_exec_time_ms),
            }

        # 7.7 Structural merge logic
        # Gate source: attach seed-X replay sessions (must pass XY pair + forward checks).
        # Merge mask source: seed-X proxy mask from replay (no detector-mask fallback).
        if bool(self.enable_structural_ops):
            pair_req_merge = int(max(1, int(self.attach_proxy_signal_persist_frames)))
            attach_sessions_rows = [
                r for r in list(attach_proxy_backfill.get("attach_sessions", []) or [])
                if isinstance(r, dict)
            ]
            start_proxy_rows = [
                r for r in list(attach_proxy_backfill.get("proxies_start_seed", []) or [])
                if isinstance(r, dict)
            ]

            def _norm_pair_key(raw: Any) -> Optional[Tuple[int, int]]:
                if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                    return None
                try:
                    a = int(raw[0])
                    b = int(raw[1])
                except Exception:
                    return None
                if int(a) == int(b):
                    return None
                return (int(min(a, b)), int(max(a, b)))

            def _session_passes_merge_gate(srow: Dict[str, Any]) -> bool:
                status = str(srow.get("status") or "")
                reverse_row = srow.get("reverse") if isinstance(srow.get("reverse"), dict) else {}
                forward_row = srow.get("forward") if isinstance(srow.get("forward"), dict) else {}
                return bool(
                    status.startswith("passed")
                    and bool(reverse_row.get("xy_ab_match_pass", False))
                    and bool(forward_row.get("pass", False))
                )

            def _pick_proxy_mask(masks_by_frame_raw: Any, current_fi: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
                if not isinstance(masks_by_frame_raw, dict):
                    return None, None
                rows: List[Tuple[int, np.ndarray]] = []
                for fi_raw, mm_raw in masks_by_frame_raw.items():
                    try:
                        fi_i = int(fi_raw)
                    except Exception:
                        continue
                    mm = _to_bool_mask(mm_raw)
                    if mm is None or np.count_nonzero(mm) <= 0:
                        continue
                    rows.append((int(fi_i), mm.astype(bool)))
                if not rows:
                    return None, None
                rows.sort(key=lambda x: int(x[0]))
                for fi_i, mm in rows:
                    if int(fi_i) == int(current_fi):
                        return mm, int(fi_i)
                past = [(fi_i, mm) for fi_i, mm in rows if int(fi_i) <= int(current_fi)]
                if past:
                    fi_i, mm = past[-1]
                    return mm, int(fi_i)
                fi_i, mm = min(rows, key=lambda x: abs(int(x[0]) - int(current_fi)))
                return mm, int(fi_i)

            def _resolve_alias(alias_map: Dict[int, int], raw_id: int) -> int:
                cur = int(raw_id)
                seen: Set[int] = set()
                while int(cur) in alias_map and int(cur) not in seen:
                    seen.add(int(cur))
                    nxt = int(alias_map.get(int(cur), cur))
                    if int(nxt) == int(cur):
                        break
                    cur = int(nxt)
                return int(cur)

            def _birth_frame(oid: int) -> int:
                return int(self._id_birth_frame.get(int(oid), int(self.frame_idx)))

            sessions_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
            for srow in attach_sessions_rows:
                pair_s = _norm_pair_key(srow.get("pair"))
                if pair_s is None:
                    continue
                try:
                    sfi = int(srow.get("start_frame_idx", -1))
                except Exception:
                    sfi = -1
                if int(sfi) < 0:
                    continue
                if not _session_passes_merge_gate(srow):
                    continue
                rev_row = srow.get("reverse") if isinstance(srow.get("reverse"), dict) else {}
                fwd_row = srow.get("forward") if isinstance(srow.get("forward"), dict) else {}
                sess_rank = (
                    int(fwd_row.get("count_on", 0) or 0),
                    float(rev_row.get("xy_ab_match_score", 0.0) or 0.0),
                )
                key_s = (int(pair_s[0]), int(pair_s[1]), int(sfi))
                prev = sessions_by_key.get(key_s)
                if prev is None:
                    sessions_by_key[key_s] = dict(srow)
                    sessions_by_key[key_s]["_rank"] = sess_rank
                else:
                    prev_rank = prev.get("_rank", (0, 0.0))
                    if sess_rank > prev_rank:
                        sessions_by_key[key_s] = dict(srow)
                        sessions_by_key[key_s]["_rank"] = sess_rank

            proxies_by_key: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
            for prow in start_proxy_rows:
                pair_p = _norm_pair_key(prow.get("pair"))
                if pair_p is None:
                    continue
                try:
                    sfi = int(prow.get("start_frame_idx", -1))
                except Exception:
                    sfi = -1
                if int(sfi) < 0:
                    continue
                key_p = (int(pair_p[0]), int(pair_p[1]), int(sfi))
                proxies_by_key.setdefault(key_p, []).append(dict(prow))

            pair_tail_rows = [
                r for r in list(attach_proxy_backfill.get("pair_signal_tail", []) or [])
                if isinstance(r, dict) and bool(r.get("qualified", False))
            ]
            merge_alias: Dict[int, int] = {}
            blocked_merge_keys: Set[Tuple[int, int, int, int, int]] = set()
            merge_iter_limit = int(max(1, len([oid for oid in tracked_masks.keys() if int(oid) >= 100])))
            merge_iter = 0
            while int(merge_iter) < int(merge_iter_limit):
                merge_iter += 1
                merge_candidates_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
                for row in pair_tail_rows:
                    pair_raw = list(row.get("pair", []) or [])
                    if len(pair_raw) < 2:
                        continue
                    src_a_id = int(min(int(pair_raw[0]), int(pair_raw[1])))
                    src_b_id = int(max(int(pair_raw[0]), int(pair_raw[1])))
                    if int(src_a_id) == int(src_b_id):
                        continue
                    if int(src_a_id) in split_parent_ids_frame or int(src_b_id) in split_parent_ids_frame:
                        continue
                    tail_count = int(row.get("tail_count", 0) or 0)
                    if int(tail_count) < int(pair_req_merge):
                        continue
                    try:
                        start_frame_idx = int(row.get("start_frame_idx", -1))
                    except Exception:
                        start_frame_idx = -1
                    if int(start_frame_idx) < 0:
                        continue
                    source_key = (int(src_a_id), int(src_b_id), int(start_frame_idx))
                    session_row = sessions_by_key.get(source_key)
                    if not isinstance(session_row, dict):
                        continue
                    proxy_rows_for_key = list(proxies_by_key.get(source_key, []) or [])
                    if not proxy_rows_for_key:
                        continue

                    a_id = int(_resolve_alias(merge_alias, int(src_a_id)))
                    b_id = int(_resolve_alias(merge_alias, int(src_b_id)))
                    if int(a_id) == int(b_id):
                        continue
                    struct_cooldown = int(max(0, int(getattr(self, "merge_after_attach_cooldown_frames", 0))))
                    if int(struct_cooldown) > 0:
                        last_evt_a = int(self._struct_event_last_frame.get(int(a_id), -10**9))
                        last_evt_b = int(self._struct_event_last_frame.get(int(b_id), -10**9))
                        last_evt = int(max(int(last_evt_a), int(last_evt_b)))
                        if int(last_evt) > -10**8:
                            evt_age = int(self.frame_idx) - int(last_evt)
                            if int(evt_age) < int(struct_cooldown):
                                continue
                    cand_key = (
                        int(src_a_id),
                        int(src_b_id),
                        int(start_frame_idx),
                        int(min(a_id, b_id)),
                        int(max(a_id, b_id)),
                    )
                    if cand_key in blocked_merge_keys:
                        continue
                    a_mask = _to_bool_mask(tracked_masks.get(int(a_id)))
                    b_mask = _to_bool_mask(tracked_masks.get(int(b_id)))
                    if a_mask is None or b_mask is None or a_mask.shape != b_mask.shape:
                        continue
                    if np.count_nonzero(a_mask) <= 0 or np.count_nonzero(b_mask) <= 0:
                        continue

                    union_mask = np.logical_or(a_mask, b_mask).astype(bool)
                    union_components = int(self._connected_components_count(union_mask))
                    if int(union_components) > 1:
                        continue

                    session_seed_box = session_row.get("seed_box")
                    session_seed_box_norm: Optional[List[int]] = None
                    if isinstance(session_seed_box, (list, tuple)) and len(session_seed_box) == 4:
                        try:
                            session_seed_box_norm = [int(v) for v in list(session_seed_box)[:4]]
                        except Exception:
                            session_seed_box_norm = None
                    # Reject merge candidate when attach-session seed box is too large.
                    if bool(self.merge_seed_box_size_gate) and session_seed_box_norm is not None:
                        try:
                            sx1, sy1, sx2, sy2 = [int(v) for v in list(session_seed_box_norm)[:4]]
                            sh, sw = union_mask.shape[:2]
                            seed_area = float(max(0, sx2 - sx1) * max(0, sy2 - sy1))
                            frame_area = float(max(1, int(sh) * int(sw)))
                            seed_area_ratio = float(seed_area / frame_area)
                            if seed_area_ratio > float(self.merge_seed_box_max_area_ratio):
                                continue
                        except Exception:
                            pass

                    best_seed: Optional[Dict[str, Any]] = None
                    reverse_row_dbg = (
                        session_row.get("reverse")
                        if isinstance(session_row.get("reverse"), dict)
                        else {}
                    )
                    reverse_xy_score = float(
                        np.clip(
                            float(reverse_row_dbg.get("xy_ab_match_score", 0.0) or 0.0),
                            0.0,
                            1.0,
                        )
                    )
                    for prow in proxy_rows_for_key:
                        masks_by_frame = prow.get("masks_by_frame")
                        mm_seed, mm_seed_frame = _pick_proxy_mask(masks_by_frame, int(self.frame_idx))
                        if mm_seed is None or mm_seed.shape != union_mask.shape:
                            continue
                        comp_cnt = int(self._connected_components_count(mm_seed))
                        if int(comp_cnt) > 1:
                            continue
                        a_in = float(self._mask_in_mask_ratio(a_mask, mm_seed))
                        b_in = float(self._mask_in_mask_ratio(b_mask, mm_seed))
                        union_cover = float(self._mask_in_mask_ratio(union_mask, mm_seed))
                        # Do not apply an additional current-frame X-only inclusion gate here.
                        # Merge gating is decided by replay evidence + pair persistence.
                        score_seed = float(
                            np.clip(
                                0.7 * float(reverse_xy_score)
                                + 0.3 * float(min(a_in, b_in, union_cover)),
                                0.0,
                                1.0,
                            )
                        )
                        prow_seed_box = prow.get("seed_box")
                        seed_box_match = False
                        if (
                            session_seed_box_norm is not None
                            and isinstance(prow_seed_box, (list, tuple))
                            and len(prow_seed_box) == 4
                        ):
                            try:
                                seed_box_match = [int(v) for v in list(prow_seed_box)[:4]] == session_seed_box_norm
                            except Exception:
                                seed_box_match = False
                        cand_seed = {
                            "mask": mm_seed.astype(bool).copy(),
                            "mask_frame_idx": None if mm_seed_frame is None else int(mm_seed_frame),
                            "a_in": float(a_in),
                            "b_in": float(b_in),
                            "union_cover": float(union_cover),
                            "components": int(comp_cnt),
                            "score": float(score_seed),
                            "seed_box_match": bool(seed_box_match),
                            "source": "seed_x_replay",
                        }
                        if best_seed is None:
                            best_seed = cand_seed
                        else:
                            rank_new = (
                                1 if bool(cand_seed.get("seed_box_match", False)) else 0,
                                float(cand_seed.get("score", 0.0)),
                            )
                            rank_old = (
                                1 if bool(best_seed.get("seed_box_match", False)) else 0,
                                float(best_seed.get("score", 0.0)),
                            )
                            if rank_new > rank_old:
                                best_seed = cand_seed
                    if best_seed is None:
                        continue

                    # Use current A∪B mask for merge apply; replay X is evidence, not final geometry.
                    merge_mask = union_mask.astype(bool).copy()
                    merge_score = float(best_seed.get("score", 0.0) or 0.0)
                    persist_score = float(min(1.0, float(tail_count) / float(max(1, pair_req_merge))))
                    total_score = float(np.clip(0.6 * merge_score + 0.4 * persist_score, 0.0, 1.0))
                    birth_a = int(_birth_frame(int(a_id)))
                    birth_b = int(_birth_frame(int(b_id)))
                    reverse_session = session_row.get("reverse") if isinstance(session_row.get("reverse"), dict) else {}
                    forward_session = session_row.get("forward") if isinstance(session_row.get("forward"), dict) else {}
                    merge_confirmed_tail = {
                        "event_type": "merge",
                        "confirm_mode": "pair_tail_replay",
                        "source_pair": [int(src_a_id), int(src_b_id)],
                        "resolved_pair": [int(min(a_id, b_id)), int(max(a_id, b_id))],
                        "start_frame_idx": int(start_frame_idx),
                        "required": int(row.get("required", pair_req_merge) or pair_req_merge),
                        "tail_count": int(tail_count),
                        "hit_frames": [int(x) for x in list(row.get("hit_frames", []) or [])],
                        "mode_history": [str(x) for x in list(row.get("mode_history", []) or [])],
                        "start_mode": str(row.get("start_mode") or "unknown"),
                        "det_mask_tail_count": int(row.get("det_mask_tail_count", 0) or 0),
                        "fallback_tail_count": int(row.get("fallback_tail_count", 0) or 0),
                        "det_mask_tail_ratio": float(row.get("det_mask_tail_ratio", 0.0) or 0.0),
                        "fallback_in_tail": bool(row.get("fallback_in_tail", False)),
                        "session_status": str(session_row.get("status") or ""),
                        "reverse_xy_ab_match_pass": bool(reverse_session.get("xy_ab_match_pass", False)),
                        "reverse_xy_ab_match_score": float(reverse_session.get("xy_ab_match_score", 0.0) or 0.0),
                        "forward_confirm_frame": (
                            int(forward_session.get("confirm_frame"))
                            if forward_session.get("confirm_frame") is not None
                            else None
                        ),
                        "forward_max_tail_on": int(forward_session.get("max_tail_on", forward_session.get("tail_on", 0)) or 0),
                        "forward_hit_frames": [int(x) for x in list(forward_session.get("tail_hit_frames", []) or [])],
                    }
                    cand_row = {
                        "a_id": int(a_id),
                        "b_id": int(b_id),
                        "src_pair": [int(src_a_id), int(src_b_id)],
                        "start_frame_idx": int(start_frame_idx),
                        "tail_count": int(tail_count),
                        "score": float(total_score),
                        "merge_mask": merge_mask,
                        "rank_max_birth": int(max(birth_a, birth_b)),
                        "rank_sum_birth": int(birth_a + birth_b),
                        "cand_key": cand_key,
                        "merge_dbg": {
                            "source": str(best_seed.get("source", "seed_x_replay")),
                            "mask_frame_idx": best_seed.get("mask_frame_idx"),
                            "a_in": float(best_seed.get("a_in", 0.0)),
                            "b_in": float(best_seed.get("b_in", 0.0)),
                            "union_cover": float(best_seed.get("union_cover", 0.0)),
                            "components": int(best_seed.get("components", 0)),
                            "union_components": int(union_components),
                            "seed_box_match": bool(best_seed.get("seed_box_match", False)),
                            "session_status": str(session_row.get("status") or ""),
                        },
                        "confirmed_tail": merge_confirmed_tail,
                        "reason": "merge_seed_union_replay",
                    }
                    mapped_pair = (int(min(a_id, b_id)), int(max(a_id, b_id)))
                    prev_cand = merge_candidates_by_pair.get(mapped_pair)
                    if prev_cand is None:
                        merge_candidates_by_pair[mapped_pair] = cand_row
                    else:
                        prev_rank = (
                            float(prev_cand.get("score", 0.0)),
                            int(prev_cand.get("tail_count", 0)),
                        )
                        cur_rank = (float(cand_row.get("score", 0.0)), int(cand_row.get("tail_count", 0)))
                        if cur_rank > prev_rank:
                            merge_candidates_by_pair[mapped_pair] = cand_row

                merge_candidates = list(merge_candidates_by_pair.values())
                if not merge_candidates:
                    break
                # Priority: older IDs first, then stronger merge score.
                merge_candidates.sort(
                    key=lambda r: (
                        int(r.get("rank_max_birth", int(self.frame_idx))),
                        int(r.get("rank_sum_birth", int(self.frame_idx) * 2)),
                        -float(r.get("score", 0.0)),
                        -int(r.get("tail_count", 0)),
                    )
                )

                merged_this_round = False
                for cand in merge_candidates:
                    a_id = int(cand.get("a_id"))
                    b_id = int(cand.get("b_id"))
                    if int(a_id) not in tracked_masks or int(b_id) not in tracked_masks:
                        blocked_merge_keys.add(tuple(cand.get("cand_key")))
                        continue
                    merge_ev = self._apply_merge_transaction(
                        a_id=int(a_id),
                        b_id=int(b_id),
                        merged_mask=cand.get("merge_mask"),
                        reason=str(cand.get("reason") or "merge_seed_union_replay"),
                        score=float(cand.get("score", 0.0) or 0.0),
                        frame_idx=int(self.frame_idx),
                        tracked_masks=tracked_masks,
                        tracked_boxes=tracked_boxes,
                        object_masks=object_masks,
                        object_boxes=object_boxes,
                        new_objects=new_objects,
                        confirmed_tail=(dict(cand.get("confirmed_tail")) if isinstance(cand.get("confirmed_tail"), dict) else None),
                    )
                    if merge_ev is None:
                        blocked_merge_keys.add(tuple(cand.get("cand_key")))
                        continue
                    child_ids = [int(x) for x in list(merge_ev.get("children", []) or [])]
                    if child_ids:
                        new_id = int(child_ids[0])
                        merge_alias[int(a_id)] = int(new_id)
                        merge_alias[int(b_id)] = int(new_id)
                    print(
                        f"[Hotrack] F{self.frame_idx} OBJ_MERGE_STRUCT: "
                        f"obj-{a_id-100}+obj-{b_id-100} -> "
                        f"obj-{child_ids[0]-100 if child_ids else -1} "
                        f"score={float(cand.get('score', 0.0)):.2f} "
                        f"src={str((cand.get('merge_dbg') or {}).get('source', 'unknown'))} "
                        f"src_pair={str(cand.get('src_pair', []))}"
                    )
                    merged_this_round = True
                    break

                if not bool(merged_this_round):
                    break

        self._prev_gray = gray
        
        # 8. Increment frame index
        self.frame_idx += 1

        # Refresh per-ID last-seen timestamp from finalized active tracks.
        frame_done_idx = int(self.frame_idx - 1)
        for oid in self.all_ids:
            oid_int = int(oid)
            if oid_int not in self._id_birth_frame:
                self._id_birth_frame[oid_int] = int(frame_done_idx)
        for oid, m in tracked_masks.items():
            oid_int = int(oid)
            if oid_int < 100:
                continue
            if m is None or np.count_nonzero(m) == 0:
                continue
            self._id_last_seen_frame[oid_int] = frame_done_idx

        self._prev_tracked_masks = {int(k): v.copy() for k, v in tracked_masks.items() if v is not None}
        
        # 9. Prepare serializable masks (RLE encoding for JSON)
        serializable_masks = {}
        for obj_id, mask in tracked_masks.items():
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            ys, xs = np.where(mask)
            rle = self._mask_to_rle(mask)
            serializable_masks[int(obj_id)] = {
                "encoding": "rle",
                "counts": rle,
                "shape": [int(mask.shape[0]), int(mask.shape[1])],
                "area": int(np.sum(mask)),
                "bbox": [int(x) for x in tracked_boxes.get(obj_id, [0, 0, 0, 0])]
            }
        
        tracked_masks_u8 = {int(k): (v.astype(np.uint8) if v is not None else v) for k, v in tracked_masks.items()}
        new_object_ids: List[int] = []
        seen_new_ids_final: Set[int] = set()
        for row in list(new_objects or []):
            if not isinstance(row, dict):
                continue
            oid_val = row.get("obj_id")
            if oid_val is None:
                continue
            try:
                oid = int(oid_val)
            except Exception:
                continue
            if oid in seen_new_ids_final:
                continue
            if int(oid) < 100:
                continue
            if int(oid) not in self.all_ids:
                continue
            mm = tracked_masks.get(int(oid))
            if mm is None or np.count_nonzero(mm) <= 0:
                continue
            seen_new_ids_final.add(int(oid))
            new_object_ids.append(int(oid))
        dino_similarity = self._dino_similarity_for_new_objects(image_bgr, new_object_ids, tracked_masks)
        dino_status = self._dino_status()

        self._prev_hand_boxes = current_hand_boxes
        self._prev_object_boxes = current_object_boxes

        cand_total = int(len(candidate_log_rows))
        cand_accepted = int(sum(1 for r in candidate_log_rows if str(r.get("status")) == "accepted"))
        cand_rejected = int(sum(1 for r in candidate_log_rows if str(r.get("status")) == "rejected"))
        sam2_call_total = 0.0
        for key, value in sorted(dict(self._sam2_timing_frame or {}).items()):
            timing[key] = float(value)
            sam2_call_total += float(value)
        timing["sam2_call_total"] = float(sam2_call_total)

        # 10. Return result
        result = {
            "frame_idx": int(self.frame_idx - 1),
            "frame_tag": frame_tag if frame_tag else f"{int(self.frame_idx - 1):06d}",
            "detections": detections,
            "detections_raw": detections_raw,
            "new_hands": new_hands,
            "new_objects": new_object_ids,
            "tracked_masks": tracked_masks_u8,  # For visualization
            "tracked_boxes": tracked_boxes,
            "serializable_masks": serializable_masks,  # For JSON export
            "overlap_info": overlap_info,  # For graph analysis
            "dino_similarity": dino_similarity,
            "dino_status": dino_status,
            "A_bg": A_bg,
            "bg_motion_ok": bool(bg_motion_ok),
            "bg_inlier_ratio": float(bg_inlier_ratio),
            "bg_num_matches": int(bg_num_matches),
            "frame_gray": gray,
            "geom": geom,
            "motion": motion,
            "backfill_masks": backfill_masks,
            "backfill_frame_indices": backfill_frame_indices,
            "attach_proxy_backfill": attach_proxy_backfill,
            "all_ids": self.all_ids.copy(),
            "hand_ids": [i for i in self.all_ids if i < 100],
            "object_ids": [i for i in self.all_ids if i >= 100],
            "events": self.events,
            "struct_events": [dict(x) for x in list(self._struct_events_frame or [])],
            "id_transitions": [dict(x) for x in list(self._id_transitions_frame or [])],
            "id_remap": {},
            "timing": timing,
            "sam2_inputs": {
                "object_boxes": [list(map(int, b)) for b in (target_boxes or [])],
                "points": [entry.get("points", []) for entry in (sam2_input_debug or [])],
                "debug": sam2_input_debug,
                "hand_points": [entry.get("points", []) for entry in (hand_sam2_input_debug or [])],
                "hand_debug": hand_sam2_input_debug,
                "dt_point_count": int(self.dt_point_count),
                "dt_point_min_dist": int(self.dt_point_min_dist),
                "enable_prompt_refinement": bool(self.enable_prompt_refinement),
            },
            "raw_object_candidates": raw_object_candidates,
            "candidate_scorer": {
                "enabled": bool(self.enable_candidate_scorer),
                "model_path": str(self.candidate_model_path),
                "threshold": float(self.candidate_score_threshold),
                "backfill_top_p": int(self.candidate_backfill_top_p),
                "candidate_count": int(cand_total),
                "accepted_count": int(cand_accepted),
                "rejected_count": int(cand_rejected),
            },
            "split_confirm_summary": {
                "count": int(len(split_confirm_emitted_rows)),
                "rows": [dict(r) for r in list(split_confirm_emitted_rows or [])],
            },
            "rejections": reject_events,
        }

        attach_proxy_meta = result.get("attach_proxy_backfill", {}) or {}
        attach_proxy_signal_meta = (attach_proxy_meta.get("signal", {}) or {}) if isinstance(attach_proxy_meta, dict) else {}
        attach_proxy_replay_stats = (attach_proxy_meta.get("replay_stats", {}) or {}) if isinstance(attach_proxy_meta, dict) else {}
        attach_proxy_merge_gate_meta = (attach_proxy_meta.get("merge_session_gate", {}) or {}) if isinstance(attach_proxy_meta, dict) else {}
        attach_proxy_merge_gate_rows = list(attach_proxy_merge_gate_meta.get("rows", []) or []) if isinstance(attach_proxy_merge_gate_meta, dict) else []
        attach_proxy_merge_gate_reason_counts = dict(attach_proxy_merge_gate_meta.get("reason_counts", {}) or {}) if isinstance(attach_proxy_merge_gate_meta, dict) else {}
        confusion_summary = []
        missing_confusion_embed_ids = []
        for oid, info in sorted((self.confusion_objects_history or {}).items(), key=lambda x: int(x[0])):
            oid_int = int(oid)
            emb_ready = bool(info.get("embed") is not None)
            if not emb_ready:
                missing_confusion_embed_ids.append(oid_int)
            confusion_summary.append({
                "obj_id": oid_int,
                "frame": int(info.get("frame", -1)),
                "reason": str(info.get("reason", "")),
                "embed_ready": emb_ready,
            })

        track_payload = {
            "frame_idx": result["frame_idx"],
            "frame_tag": result["frame_tag"],
            "hand_ids": result["hand_ids"],
            "object_ids": result["object_ids"],
            "new_hands": result["new_hands"],
            "new_objects": result["new_objects"],
            "events": result["events"],
            "struct_events": result.get("struct_events", []) or [],
            "id_transitions": result.get("id_transitions", []) or [],
            "tracked_boxes": result["tracked_boxes"],
            "serializable_masks": result["serializable_masks"],
            "overlap_info": result.get("overlap_info", {}) or {},
            "dino_similarity": result.get("dino_similarity", []) or [],
            "dino_status": result.get("dino_status", {}) or {},
            "confusion_summary": {
                "count": int(len(self.confusion_objects_history)),
                "missing_embed_ids": [int(x) for x in missing_confusion_embed_ids],
                "entries": confusion_summary,
            },
            "timing": result.get("timing", {}) or {},
            "geom": result.get("geom", {}) or {},
            "motion": result.get("motion", {}) or {},
            "bg_motion_ok": bool(result.get("bg_motion_ok", False)),
            "A_bg": result.get("A_bg"),
            "backfill_frame_indices": result.get("backfill_frame_indices", []) or [],
            "attach_proxy_backfill_summary": {
                "frame_count": int(len((attach_proxy_meta.get("frame_indices", []) if isinstance(attach_proxy_meta, dict) else []) or [])),
                "proxy_count": int(len((attach_proxy_meta.get("proxies", []) if isinstance(attach_proxy_meta, dict) else []) or [])),
                "start_seed_proxy_count": int(len((attach_proxy_meta.get("proxies_start_seed", []) if isinstance(attach_proxy_meta, dict) else []) or [])),
                "start_seed_replay_count": int(len((attach_proxy_meta.get("start_seed_replay", []) if isinstance(attach_proxy_meta, dict) else []) or [])),
                "attach_sessions_count": int(len((attach_proxy_meta.get("attach_sessions", []) if isinstance(attach_proxy_meta, dict) else []) or [])),
                "start_seed_replay_ok_count": int(sum(
                    1
                    for r in list((attach_proxy_meta.get("start_seed_replay", []) if isinstance(attach_proxy_meta, dict) else []) or [])
                    if isinstance(r, dict) and bool(r.get("ok", False))
                )),
                "signal_on": bool(attach_proxy_signal_meta.get("signal", False)),
                "signal_pairs_count": int(len(attach_proxy_signal_meta.get("signal_pairs", []) or [])),
                "replay_candidate_pairs": int(attach_proxy_replay_stats.get("candidate_pairs", 0) or 0),
                "replay_qualified_pairs": int(attach_proxy_replay_stats.get("qualified_pairs", 0) or 0),
                "replay_cache_hits": int(attach_proxy_replay_stats.get("cache_hits", 0) or 0),
                "replay_cache_misses": int(attach_proxy_replay_stats.get("cache_misses", 0) or 0),
                "replay_exec_count": int(attach_proxy_replay_stats.get("replay_exec_count", 0) or 0),
                "replay_exec_time_ms": float(attach_proxy_replay_stats.get("replay_exec_time_ms", 0.0) or 0.0),
                "merge_session_gate_rows_count": int(attach_proxy_merge_gate_meta.get("rows_count", len(attach_proxy_merge_gate_rows)) if isinstance(attach_proxy_merge_gate_meta, dict) else len(attach_proxy_merge_gate_rows)),
                "merge_session_gate_qualified_rows_count": int(attach_proxy_merge_gate_meta.get("qualified_rows_count", 0) if isinstance(attach_proxy_merge_gate_meta, dict) else 0),
                "merge_session_gate_pass_rows_count": int(attach_proxy_merge_gate_meta.get("gate_pass_rows_count", 0) if isinstance(attach_proxy_merge_gate_meta, dict) else 0),
                "merge_session_gate_reason_counts": {str(k): int(v) for k, v in attach_proxy_merge_gate_reason_counts.items()},
                "merge_session_gate_rows": attach_proxy_merge_gate_rows,
            },
            "id_remap": result.get("id_remap", {}) or {},
            "sam2_inputs": result.get("sam2_inputs", {}) or {},
            "candidate_scorer": result.get("candidate_scorer", {}) or {},
            "split_confirm_summary": result.get("split_confirm_summary", {}) or {},
            "rejections": result.get("rejections", []) or [],
            "detections_raw": result.get("detections_raw", {}) or {},
            "detections_filtered": result.get("detections", {}) or {},
        }
        a_bg = track_payload.get("A_bg")
        if isinstance(a_bg, np.ndarray):
            track_payload["A_bg"] = a_bg.tolist()
        if self.save_tracking_json:
            os.makedirs(self.tracking_json_dir, exist_ok=True)
            track_path = os.path.join(self.tracking_json_dir, f"{result['frame_tag']}_track.json")
            with open(track_path, "w", encoding="utf-8") as f:
                json.dump(track_payload, f, ensure_ascii=False, indent=2)

        return result

    def _run_yolo_detector_only(
        self,
        image_bgr: np.ndarray,
        *,
        target_contact_code: str = "P",
        apply_target_filter: bool = False,
    ) -> Dict[str, Any]:
        detections = detect_image_bgr(
            self.ho_model,
            image_bgr,
            use_cuda=self.use_cuda,
            thresh_hand=self.ho_thresh_hand,
            thresh_obj=self.ho_thresh_obj,
        )
        if apply_target_filter:
            detections = self._suppress_non_target_matches_for_same_object(
                detections,
                target_contact_code=str(target_contact_code),
            )
        return detections
