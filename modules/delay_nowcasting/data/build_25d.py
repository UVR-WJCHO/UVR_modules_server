"""frame 표를 2.5D 표현으로 바꿔 쓴다.

WiLoR 가 원래 내놓는 형태 그대로다 — 관절마다 (정규화 image 좌표, 손목 기준 상대 depth).
손목 절대 depth 는 어디에도 쓰지 않는다. 그 채널이 forecast 입력의 유일한 오염원이었다
(HL2 실측: 프레임간 손목 depth 변화 중앙 1.0 mm, 최대 458 mm, 3D 튐과 상관 +0.93).
실제 거리는 배포에서 기기가 최신 depth 로 복원한다.

  u_n, v_n : x/z, y/z  (무차원)
  rel_z    : z - z_wrist  (meter). 손목은 정의상 0

머리 회전은 지우지 못한다. 광선을 회전시키려면 그 점의 절대 거리가 필요한데 그것이
바로 버린 값이다. 그래서 이 표현은 카메라 좌표계에 매여 있고, 머리가 돌면 정지한 손도
움직이는 것으로 보인다. 그 크기는 따로 측정해야 한다.

  python -m modules.delay_nowcasting.data.build_25d --dataset mixed_v1
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from ..config import REPO_ROOT
from .canonical_schema import NUM_JOINTS, WRIST


def to_25d(joints_camera: np.ndarray) -> np.ndarray:
    """(N, 21, 3) 카메라 좌표 -> (N, 21, 3) = (u_n, v_n, rel_z).

    무효 프레임(NaN)은 NaN 그대로 통과시킨다.
    """
    z = joints_camera[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ray = joints_camera[:, :, :2] / z[:, :, None]
    rel = z - z[:, [WRIST]]
    return np.concatenate([ray, rel[:, :, None]], axis=-1)


def from_25d(pose_25d: np.ndarray, wrist_depth_m: np.ndarray) -> np.ndarray:
    """(N, 21, 3) 2.5D + (N,) 손목 depth -> (N, 21, 3) 카메라 좌표. 기기가 하는 일이다."""
    z = wrist_depth_m[:, None] + pose_25d[:, :, 2]
    return np.concatenate([pose_25d[:, :, :2] * z[:, :, None], z[:, :, None]], axis=-1)


def convert(source, out_path) -> None:
    frames = pl.read_parquet(source)
    joints_camera = (frames["joints_camera"].to_numpy()
                     .reshape(-1, NUM_JOINTS, 3).astype(np.float64))
    converted = to_25d(joints_camera)
    frames = frames.with_columns(
        pl.Series("joints_world", converted.reshape(len(frames), -1).astype(np.float32)))
    frames.write_parquet(out_path)
    finite = np.isfinite(converted).all((1, 2))
    print(f"{out_path.name}: {len(frames):,} 행, 유한 {finite.mean()*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="frame 표를 2.5D 표현으로 변환")
    ap.add_argument("--dataset", default="mixed_v1")
    ap.add_argument("--files", nargs="+", default=[
        "wilor_frames_train", "wilor_frames_val",
        "canonical_frames_train", "canonical_frames_val"])
    args = ap.parse_args()
    root = REPO_ROOT / "research_data" / args.dataset
    for name in args.files:
        source = root / f"{name}.parquet"
        if not source.exists():
            print(f"{name}: 없음 — 건너뛴다")
            continue
        convert(source, root / f"{name.replace('_frames', '25d_frames')}.parquet")


if __name__ == "__main__":
    main()
