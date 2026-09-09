"""DexYCB -> canonical 21-joint pose.

HOT3D 와 달리 손이 볼 것이 거의 없다. `labels_*.npz` 의 `joint_3d` 가 이미
**canonical 21-joint MANO 순서**(= WiLoR 출력 순서)의 camera-frame 좌표라 joint mapping 이
필요 없다.

실측으로 확인한 규약 (2026-08-18):
  - `calibration/extrinsics_<id>/extrinsics.yml` 의 3x4 는 **T_world_camera** 다.
    8 개 카메라의 joint_3d 를 world 로 옮기면 0.000 mm 로 일치한다.
  - 무효 프레임은 joint_3d 가 전부 -1 이다.
  - 시퀀스당 손은 하나(`meta.yml: mano_sides`), 길이는 약 74 frame(약 2.5 초).
  - per-frame timestamp 가 없다. 8 카메라가 동기 촬영된 30 fps 이므로
    frame index x 33.333 ms 로 만든다. HOT3D 와 달리 간격 지터가 없다.

GT 궤적은 카메라와 무관하게 하나뿐이므로, 카메라는 조건 C(WiLoR 입력)에서만 갈린다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..canonical_schema import NUM_JOINTS, WRIST

SOURCE_DATASET = "DexYCB"
FRAME_INTERVAL_NS = 33_333_333          # 30 fps 동기 촬영
IMAGE_SIZE = (640, 480)
INVALID_VALUE = -1.0


@dataclass
class SequenceInfo:
    subject: str
    sequence: str
    handedness: str
    serials: list[str]
    num_frames: int
    extrinsics_id: str
    mano_calib: str

    @property
    def sequence_id(self) -> str:
        return f"{self.subject}/{self.sequence}"


def read_meta(sequence_dir: Path) -> SequenceInfo:
    meta = yaml.safe_load((sequence_dir / "meta.yml").read_text())
    sides = meta["mano_sides"]
    if len(sides) != 1:
        raise ValueError(f"{sequence_dir}: 손이 {len(sides)} 개다. 시퀀스당 1 개를 가정한다")
    return SequenceInfo(
        subject=sequence_dir.parent.name, sequence=sequence_dir.name,
        handedness=sides[0].upper(), serials=[str(s) for s in meta["serials"]],
        num_frames=int(meta["num_frames"]), extrinsics_id=str(meta["extrinsics"]),
        mano_calib=str(meta["mano_calib"][0]),
    )


def read_extrinsics(calibration_root: Path, extrinsics_id: str) -> dict[str, np.ndarray]:
    """serial -> (4, 4) T_world_camera."""
    path = calibration_root / f"extrinsics_{extrinsics_id}" / "extrinsics.yml"
    raw = yaml.load(path.read_text(), Loader=yaml.UnsafeLoader)["extrinsics"]
    out = {}
    for serial, values in raw.items():
        if serial == "apriltag":
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        out[str(serial)] = matrix
    return out


def read_intrinsics(calibration_root: Path, serial: str,
                    resolution: str = "640x480") -> dict:
    # intrinsics 파일에도 !!python/tuple (color->depth extrinsic) 이 들어 있어
    # safe_load 로는 읽히지 않는다.
    values = yaml.load(
        (calibration_root / "intrinsics" / f"{serial}_{resolution}.yml").read_text(),
        Loader=yaml.UnsafeLoader)
    color = values["color"]
    return {"fx": float(color["fx"]), "fy": float(color["fy"]),
            "cx": float(color["ppx"]), "cy": float(color["ppy"])}


def read_joint_track(sequence_dir: Path, serial: str, num_frames: int,
                     image_size: tuple[int, int] = IMAGE_SIZE, margin: int = 0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F, 21, 3) camera-frame joints, (F,) valid mask, (F,) 화면 내 여부.

    DexYCB 의 `joint_3d` 는 손이 **화면 밖에 있어도 유효값**이다(3D 라벨이라 시야와
    무관). 그래서 유효성만 보고 카메라를 고르면 손이 안 보이는 뷰가 뽑히고, 그 위에서
    WiLoR 를 돌리면 검출률이 0 이 된다. 라벨의 `joint_2d` 로 화면 내 여부를 따로 본다.
    """
    width, height = image_size
    joints = np.full((num_frames, NUM_JOINTS, 3), np.nan)
    valid = np.zeros(num_frames, dtype=bool)
    in_frame = np.zeros(num_frames, dtype=bool)
    for frame in range(num_frames):
        path = sequence_dir / serial / f"labels_{frame:06d}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        value = data["joint_3d"][0].astype(np.float64)
        if np.all(value == INVALID_VALUE):
            continue
        joints[frame] = value
        valid[frame] = True
        uv = data["joint_2d"][0].astype(np.float64)
        in_frame[frame] = bool(
            np.isfinite(uv).all()
            and (uv[:, 0] >= margin).all() and (uv[:, 0] < width - margin).all()
            and (uv[:, 1] >= margin).all() and (uv[:, 1] < height - margin).all())
    return joints, valid, in_frame


def choose_camera(sequence_dir: Path, info: SequenceInfo) -> str:
    """이미지 기반 실험에 쓸 카메라 하나를 결정론적으로 고른다.

    기준은 **21 joint 가 모두 화면 안에 들어오는 프레임 비율**이다. 동률이면 손이
    가까운 쪽. 처음에는 유효 프레임 수와 거리만 봤는데, 3D 라벨은 화면 밖에서도
    유효해서 손이 안 보이는 카메라가 뽑혔다(val 100 개 중 24 개가 검출률 0).
    """
    best = None
    for serial in info.serials:
        joints, valid, in_frame = read_joint_track(sequence_dir, serial, info.num_frames)
        usable = valid & in_frame
        if not usable.any():
            continue
        depth = float(np.nanmedian(joints[usable][:, WRIST, 2]))
        key = (-int(usable.sum()), depth, serial)
        if best is None or key < best[0]:
            best = (key, serial)
    if best is None:
        raise ValueError(f"{sequence_dir}: 21 joint 가 모두 보이는 카메라가 없다")
    return best[1]


def frame_timestamps_ns(num_frames: int, offset_ns: int = 0) -> np.ndarray:
    return offset_ns + np.arange(num_frames, dtype=np.int64) * FRAME_INTERVAL_NS
