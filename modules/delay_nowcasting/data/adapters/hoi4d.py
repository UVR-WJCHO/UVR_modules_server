"""HOI4D adapter — 시퀀스 탐색, MANO 정해, intrinsics, timestamp.

HOI4D 는 앞선 두 데이터셋과 세 가지가 다르다.

  1. **3D 관절이 주석에 없다.** pickle 은 MANO 계수(poseCoeff 48, beta 10, trans 3)와
     kps2D 만 담는다. 21 관절은 MANO forward 로 직접 만든다. 검증은 재투영으로 했고
     아래 규약에서 중앙값 0.83 px 였다 (임의로 바꾸면 그 값이 깨진다).
       - poseCoeff[:3] = global_orient, poseCoeff[3:48] = hand_pose (PCA 아님)
       - flat_hand_mean=True   (False 로 두면 재투영이 40 px 로 벌어진다)
       - 결과가 곧 카메라 좌표다. 별도 변환이 없다.
       - MANO 원순서를 canonical 21 순서로 재배열해야 한다 (JOINT_MAP)
  2. **RGB 가 mp4 다.** 시퀀스당 파일 하나이고 프레임을 순차 디코딩해야 한다.
  3. **15 fps 고정이고 world 좌표계가 없다.** timestamp 는 index/15 로 만들고,
     camera_to_world 는 단위행렬로 둔다 — 이 데이터셋에는 world frame 이 없다.
     2.5D 표현은 카메라 좌표만 쓰므로 학습·평가에 영향이 없다.

MANO_LEFT.pkl 이 없어 **오른손만** 쓴다.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..canonical_schema import NUM_JOINTS

SOURCE_DATASET = "HOI4D"
FPS = 15.0
NS_PER_FRAME = int(round(1e9 / FPS))
IMAGE_SIZE = (1920, 1080)
HANDEDNESS = "RIGHT"

# MANO 원순서: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
# canonical  : 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
# None 은 손끝이며 vertex 에서 가져온다.
JOINT_MAP: tuple[int | None, ...] = (
    0, 13, 14, 15, None, 1, 2, 3, None, 4, 5, 6, None, 10, 11, 12, None, 7, 8, 9, None)
TIP_VERTEX: dict[int, int] = {4: 745, 8: 317, 12: 445, 16: 556, 20: 673}


@dataclass(frozen=True)
class SequenceInfo:
    sequence_id: str          # "<cam>/H*/C*/N*/S*/s*/T*"
    camera: str
    pose_dir: Path
    video: Path
    frames: np.ndarray        # 주석이 있는 frame index (오름차순)


def discover_sequences(root: Path, cameras: list[str]) -> list[SequenceInfo]:
    """handpose 와 mp4 가 **둘 다** 있는 시퀀스만 돌려준다."""
    out: list[SequenceInfo] = []
    for camera in sorted(cameras):
        pose_root = root / "handpose" / "refinehandpose_right" / camera
        if not pose_root.is_dir():
            continue
        for pose_dir in sorted(p for p in pose_root.glob("H*/C*/N*/S*/s*/T*") if p.is_dir()):
            frames = sorted(int(f.stem) for f in pose_dir.glob("*.pickle") if f.stem.isdigit())
            if not frames:
                continue
            relative = pose_dir.relative_to(pose_root)
            video = root / "HOI4D_release" / camera / relative / "align_rgb" / "image.mp4"
            if not video.is_file():
                continue
            out.append(SequenceInfo(f"{camera}/{relative}", camera, pose_dir, video,
                                    np.asarray(frames, dtype=np.int64)))
    return out


def read_intrinsics(root: Path, camera: str) -> np.ndarray:
    """(3, 3). HOI4D 는 카메라별로 하나이고 시퀀스마다 바뀌지 않는다."""
    return np.load(root / "camera_params" / camera / "intrin.npy").astype(np.float64)


def frame_timestamps_ns(frames: np.ndarray) -> np.ndarray:
    """15 fps 고정이라 index 로 만든다. 실제 캡처 시각은 배포되지 않는다."""
    return np.asarray(frames, dtype=np.int64) * NS_PER_FRAME


class ManoJointSolver:
    """MANO 계수 -> canonical 21 관절 (카메라 좌표, meter).

    smplx.create 는 model_path 뒤에 model_type 을 덧붙이므로 MANO 를 직접 만든다.
    batch_size 는 호출마다 달라지므로 필요할 때 다시 만든다.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._layer, self._batch = None, 0

    def _get(self, batch: int):
        import smplx

        if self._layer is None or self._batch != batch:
            self._layer = smplx.MANO(model_path=str(self.model_dir), is_rhand=True,
                                     use_pca=False, flat_hand_mean=True, batch_size=batch)
            self._batch = batch
        return self._layer

    def __call__(self, pose: np.ndarray, betas: np.ndarray, transl: np.ndarray) -> np.ndarray:
        """(B, 48), (B, 10), (B, 3) -> (B, 21, 3)."""
        import torch

        layer = self._get(len(pose))
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
        with torch.no_grad():
            out = layer(global_orient=t(pose[:, :3]), hand_pose=t(pose[:, 3:48]),
                        betas=t(betas), transl=t(transl))
        joints = out.joints.numpy()
        vertices = out.vertices.numpy()
        return np.stack([vertices[:, TIP_VERTEX[i]] if m is None else joints[:, m]
                         for i, m in enumerate(JOINT_MAP)], axis=1)


def read_joint_track(info: SequenceInfo, solver: ManoJointSolver, intrinsics: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(joints_camera (T,21,3), valid (T,), in_frame (T,)). T = 주석이 있는 프레임 수."""
    # 규격에 맞지 않는 pickle 이 드물게 섞여 있다. 하나 때문에 시퀀스 전체를 버리지 않고
    # 그 프레임만 invalid 로 두되, 배열 모양은 유지해 행 index 대응을 지킨다.
    n = len(info.frames)
    pose = np.zeros((n, 48)); betas = np.zeros((n, 10)); transl = np.zeros((n, 3))
    readable = np.zeros(n, dtype=bool)
    for i, frame in enumerate(info.frames):
        try:
            with open(info.pose_dir / f"{frame}.pickle", "rb") as handle:
                record = pickle.load(handle)
            p = np.asarray(record["poseCoeff"], np.float64).ravel()
            b = np.asarray(record["beta"], np.float64).ravel()
            t = np.asarray(record["trans"], np.float64).ravel()
        except Exception:
            continue
        if p.shape != (48,) or b.shape != (10,) or t.shape != (3,):
            continue
        pose[i], betas[i], transl[i], readable[i] = p, b, t, True

    joints = solver(pose, betas, transl)
    valid = (readable & np.isfinite(joints).all(axis=(1, 2))
             & (joints[:, :, 2] > 0.05).all(axis=1))

    # 화면 밖 프레임은 이미지 기반 방법이 애초에 볼 수 없으므로 제외한다 (DexYCB 와 같은 규칙).
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = fx * joints[:, :, 0] / joints[:, :, 2] + cx
    v = fy * joints[:, :, 1] / joints[:, :, 2] + cy
    width, height = IMAGE_SIZE
    in_frame = ((u >= 0) & (u < width) & (v >= 0) & (v < height)).all(axis=1)
    return joints, valid, in_frame


def read_frames(video: Path, wanted: np.ndarray):
    """mp4 를 순차 디코딩하며 (frame_index, BGR) 을 내보낸다.

    임의 접근(CAP_PROP_POS_FRAMES)은 B-frame 이 섞인 파일에서 어긋나므로 쓰지 않는다.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    wanted_set, index = set(int(w) for w in wanted), 0
    try:
        while wanted_set:
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted_set:
                wanted_set.discard(index)
                yield index, image
            index += 1
    finally:
        capture.release()
