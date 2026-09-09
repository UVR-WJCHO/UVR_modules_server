"""Nowcaster 예측을 Depth-ICP 로 정제한 뒤 최종 자세를 비교한다.

이 연구의 산출물은 nowcaster 의 예측 자체가 아니라, 예측을 seed 로 삼아 **목표 시점의
depth 로 ICP 를 돌린 최종 자세** 다. 그래서 비교도 ICP 이후에 해야 한다. ICP 는 global
rigid 만 푼다(`staged=True`) — 손가락 관절은 seed 가 준 그대로 두므로, nowcaster 의
articulation 예측이 그대로 최종 결과에 남는다.

`research/Display-Time Hand-Pose Correction/src` 의 기존 구현을 그대로 쓴다. 그쪽
`phase_eval1000.py` 는 stale 자세를 seed 로 쓰는데(= Hold + ICP), 여기서는 seed 를
각 nowcaster 의 예측으로 바꿔 끼운다.

  skel = Skeleton.calibrate_from_gt(seed_joints)   # rest = seed 자세, theta=0 이 seed 재현
  obs  = build_samples(seq, serial, anchor_frame, target_frame, skel, params)
  (R, t, th), ok = DepthSolver(...).solve_guarded(skel.Rg0, skel.tg0, 0)
  final = skel.fk(R, t, th)[0]

한계(결과에 함께 적을 것): `build_samples` 는 anchor frame 의 **GT** joint 로 손 영역
hull 을 만든다. 실제 배포에서는 WiLoR anchor 를 써야 하므로 이 조건은 낙관적이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ICP_SRC = (Path(__file__).resolve().parents[2]
           / "Display-Time Hand-Pose Correction" / "src")


def _import_icp():
    if str(ICP_SRC) not in sys.path:
        sys.path.insert(0, str(ICP_SRC))
    from depth_icp import DepthParams, DepthSolver, build_samples  # noqa: E402
    from dexycb_loader import DexYCBSequence  # noqa: E402
    from skeleton import DEX_IDX, Skeleton  # noqa: E402

    return DepthParams, DepthSolver, build_samples, DexYCBSequence, Skeleton, DEX_IDX


def _rigid_from_to(source: np.ndarray, destination: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """source (N, 3) -> destination (N, 3) 최적 rigid 변환 (R, t). Kabsch."""
    source_mean, destination_mean = source.mean(0), destination.mean(0)
    covariance = (source - source_mean).T @ (destination - destination_mean)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, destination_mean - rotation @ source_mean


def default_params(**overrides):
    DepthParams = _import_icp()[0]
    params = DepthParams(staged=True, **overrides)   # global rigid 만
    return params


class IcpRefiner:
    """(sequence, camera) 하나에 대해 여러 seed 를 ICP 로 정제한다."""

    def __init__(self, calibration_root: Path, sequence_dir: Path, params=None):
        (DepthParams, self.DepthSolver, self.build_samples,
         DexYCBSequence, self.Skeleton, self.DEX_IDX) = _import_icp()
        self.seq = DexYCBSequence(str(calibration_root), str(sequence_dir))
        self.params = params if params is not None else DepthParams(staged=True)

    def refine(self, serial: str, anchor_frame: int, target_frame: int,
               seed_joints_camera: np.ndarray,
               anchor_joints_camera: np.ndarray) -> tuple[np.ndarray, bool]:
        """예측 자세를 seed 로 ICP 정제. 반환 (21, 3) 최종 자세와 발산 가드 통과 여부.

        **skeleton 의 rest 는 반드시 anchor 자세여야 한다.** `DepthSolver` 는 표면점을
        `rel = R_rest^T (P - p_rest[attach])` 로 묶는데, P 는 anchor frame depth 에서
        뽑은 점이다. rest 를 예측 자세로 두면 anchor 의 표면 형상이 다른 자세의 골격에
        묶여 손 모델이 일그러진다(예측이 좋을수록 더 나빠진다).

        `staged=True` 는 global rigid 만 풀므로, 예측이 기여할 수 있는 것도 rigid 성분
        뿐이다. 그래서 anchor -> 예측 의 최적 rigid 변환을 구해 초기값으로 넣는다.
        """
        skel = self.Skeleton.calibrate_from_gt(anchor_joints_camera)
        obs = self.build_samples(self.seq, serial, anchor_frame, target_frame,
                                 skel, self.params)
        rotation0, translation0 = _rigid_from_to(anchor_joints_camera, seed_joints_camera)
        solver = self.DepthSolver(skel, obs, self.params)
        (rotation, translation, theta), ok = solver.solve_guarded(
            rotation0 @ skel.Rg0, rotation0 @ skel.tg0 + translation0, np.zeros(skel.m))
        joints, _, _ = skel.fk(rotation, translation, theta)
        return joints[self.DEX_IDX], bool(ok)
