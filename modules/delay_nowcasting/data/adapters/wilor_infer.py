"""HOT3D preview 영상에 실제 WiLoR 를 돌려 배포 조건의 history 를 만든다 (B 계획 §11 Phase 5).

GT history 성능은 motion prediction 상한일 뿐이다(§10.3 A). 배포 경로는 프레임마다
독립적으로 WiLoR 를 돌리므로 시간 연속성이 강제되지 않고 프레임별 흔들림이 남는다.
그 조건에서도 개선 방향이 유지되는지 보려면 실제 WiLoR 출력이 필요하다.

실측으로 확정한 HOT3D 영상 파이프라인 (2026-08-17):

  1. preview mp4 frame k  <->  cache frame_idx k        (lag 0, 상호상관 0.934)
     mp4 는 cache 보다 프레임이 1 개 적다(마지막 프레임 없음).
  2. mp4 는 annotation 좌표계에서 **CW 90 도 돌아가 있다**. 되돌리려면 CCW 90 도 회전.
     검증: 회전별 detector box 중심 vs GT box2d 중심 오차
           ccw90 = 25.3 px, none = 465.6, cw90 = 652.4, 180 = 463.0 px
  3. 회전 후 FISHEYE624 로 rectify 해야 WiLoR(원근 학습)에 넣을 수 있다.
  4. detector class 0 = LEFT, 1 = RIGHT (GT box2d 의 hand_index 와 같다).

root lifting: 서버는 depth 센서에서 실제 wrist depth 를 받아 lifting 하지만
(`main_handtrack.py: lift_pose_cam3d`) HOT3D cache 에는 depth 가 없다. 그래서
GT wrist depth 를 써서 lifting 한다. 이렇게 하면 **root depth 오차는 배제되고 WiLoR 의
2D/articulation 오차만** history 에 남는다. 즉 이 조건도 여전히 낙관적이며, 실제 배포는
여기에 root depth 오차가 더해진다. 결과를 보고할 때 반드시 함께 밝힌다.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..canonical_schema import NUM_JOINTS, WRIST

# 실측으로 확정한 상수. 바꾸려면 위 docstring 의 검증을 다시 돌려야 한다.
MP4_TO_ANNOTATION_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE
DETECTOR_CLASS_TO_HANDEDNESS = {0: "LEFT", 1: "RIGHT"}


@dataclass
class HandDetection:
    handedness: str
    joints_camera: np.ndarray     # (21, 3) absolute, meter, rectified camera frame
    joints_2d: np.ndarray         # (21, 2) rectified pixel
    bbox: np.ndarray              # (4,)
    root_relative: np.ndarray     # (21, 3) WiLoR 원본 출력


def load_tracker(repo_root: Path):
    """WiLoR ONNX tracker 를 만든다. modules/ 를 import 경로에 넣어야 한다."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    modules = str(repo_root / "modules")
    if modules not in sys.path:
        sys.path.insert(0, modules)
    from handtracker_onnx import WilorHandTrackerONNX

    return WilorHandTrackerONNX()


def tracker_metadata(tracker, repo_root: Path) -> dict:
    """B 계획 §11 Phase 5 가 요구하는 필수 metadata."""
    import hashlib

    import onnxruntime as ort

    onnx_path = repo_root / "pretrained" / "handtracker_onnx" / "wilor_final_standard.onnx"
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:16] if onnx_path.exists() else "missing"
    return {
        "wilor_onnx_sha256_16": digest,
        "onnxruntime_version": ort.__version__,
        "execution_providers": list(tracker.session.get_providers()),
        "image_size": int(tracker.cfg.MODEL.IMAGE_SIZE),
        "bbox_shape": list(tracker.cfg.MODEL.BBOX_SHAPE),
        "joint_convention": "mano_to_canonical (WiLoR 출력 순서 그대로)",
        "root_lifting": "GT wrist depth (HOT3D cache 에 depth 없음). root depth 오차는 제외됨",
        "mp4_rotation": "ROTATE_90_COUNTERCLOCKWISE",
    }


def infer_frame(tracker, rectified: np.ndarray, pinhole, gt_wrist_camera: dict[str, np.ndarray],
                max_hands: int = 2) -> list[HandDetection]:
    """rectified 이미지 한 장에서 검출된 손마다 절대 3D pose 를 만든다.

    gt_wrist_camera: handedness -> (3,) GT wrist 위치(rectified camera frame).
    해당 손의 GT 가 없으면 그 손은 건너뛴다(임의 depth 를 지어내지 않는다).
    """
    import torch

    from handtracker_onnx.wilor_onnx_utils import cam_crop_to_full

    boxes, classes = tracker.detect(rectified)
    if boxes is None:
        return []

    results: list[HandDetection] = []
    seen: set[str] = set()
    for box, cls in zip(boxes, classes):
        handedness = DETECTOR_CLASS_TO_HANDEDNESS.get(int(cls))
        if handedness is None or handedness in seen or len(seen) >= max_hands:
            continue
        wrist_gt = gt_wrist_camera.get(handedness)
        if wrist_gt is None:
            continue
        seen.add(handedness)

        is_right = int(cls)
        # WILOR 는 오른손 전용이라 왼손은 패치를 반전해 넣어야 한다
        patch, center, bbox_size = tracker._preprocess_hand_patch(
            rectified, box, do_flip=(is_right == 0))
        outputs = tracker.session.run(None, {tracker.input_name: patch})
        pred_cam = outputs[0].astype(np.float32)
        joints = outputs[1].astype(np.float32)[0].copy()          # (21, 3) root-relative

        pred_cam[0, 1] = pred_cam[0, 1] * float(2 * is_right - 1)
        if is_right == 0:
            joints[:, 0] = -joints[:, 0]

        scaled_focal = float(tracker.cfg.EXTRA.FOCAL_LENGTH / tracker.cfg.MODEL.IMAGE_SIZE
                             * max(rectified.shape[1], rectified.shape[0]))
        cam_t = cam_crop_to_full(
            torch.from_numpy(pred_cam), torch.from_numpy(np.array([center], np.float32)),
            torch.from_numpy(np.array([bbox_size], np.float32)),
            torch.from_numpy(np.array([[rectified.shape[1], rectified.shape[0]]], np.float32)),
            torch.tensor(scaled_focal, dtype=torch.float32)).cpu().numpy()[0]

        joints_2d = tracker._project_full_img(
            joints, cam_t, scaled_focal,
            np.array([rectified.shape[1], rectified.shape[0]], np.float32))

        # 서버와 같은 방식: wrist 의 깊이는 센서(여기서는 GT)에서 받고, 나머지는
        # root-relative z 로 올린다. 2D 는 WiLoR 의 것을 그대로 쓴다.
        depth = float(wrist_gt[2])
        z = depth + (joints[:, 2] - joints[WRIST, 2])
        camera = np.empty((NUM_JOINTS, 3), dtype=np.float64)
        camera[:, 2] = z
        camera[:, 0] = (joints_2d[:, 0] - pinhole.cx) / pinhole.fx * z
        camera[:, 1] = (joints_2d[:, 1] - pinhole.cy) / pinhole.fy * z

        results.append(HandDetection(handedness, camera, joints_2d.astype(np.float64),
                                     np.asarray(box, np.float64), joints.astype(np.float64)))
    return results
