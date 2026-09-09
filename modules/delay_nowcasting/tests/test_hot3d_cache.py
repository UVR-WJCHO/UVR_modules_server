"""빌드된 HOT3D canonical cache 에 대한 검증 (B 계획 §11 Phase 1, §12).

cache 가 없으면 통째로 skip 한다. 합성 데이터로는 확인할 수 없는 것들만 여기 둔다.
"""
import numpy as np
import polars as pl
import pytest

from modules.delay_nowcasting.config import REPO_ROOT, load_config
from modules.delay_nowcasting.data.build_windows import load_canonical_frames
from modules.delay_nowcasting.data.canonical_schema import (
    NUM_JOINTS,
    chirality,
    transform_points,
    invert_transform,
    validate_canonical_frames,
)

CONFIG = load_config(REPO_ROOT / "modules/delay_nowcasting/configs/data/hot3d_v1.yaml")
CACHE = REPO_ROOT / CONFIG["output_root"] / CONFIG.name / "canonical_frames_val.parquet"
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="canonical cache 미생성")


@pytest.fixture(scope="module")
def frames():
    return load_canonical_frames(CACHE)


def test_schema_units_and_handedness(frames):
    validate_canonical_frames(frames)


def test_invalid_rows_are_never_zero_filled(frames):
    """결측 frame 을 zero pose 로 채우면 정지한 손처럼 보인다 (B 계획 §4.1).

    가시성 필터 도입 후 `valid=False` 는 두 가지를 뜻한다 — MANO pose 가 없거나
    (joints 가 NaN), 21 joint 가 rectified 화면 밖이거나(joints 는 유한하지만 이미지
    기반 방법이 볼 수 없다). 어느 쪽도 0 으로 채우지 않는다.
    """
    invalid = frames.filter(~pl.col("valid"))
    assert invalid.height > 0
    joints = invalid["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3)
    assert invalid["visibility"].to_numpy().max() == 0

    missing_pose = np.isnan(joints).all(axis=(1, 2))
    out_of_frame = np.isfinite(joints).all(axis=(1, 2))
    assert bool((missing_pose | out_of_frame).all()), "NaN 과 유한값이 섞인 행이 있다"
    if out_of_frame.any():
        # 화면 밖 프레임도 실제 좌표여야 한다. 0 으로 채우면 원점에 손이 있는 셈이 된다
        assert np.abs(joints[out_of_frame]).max() > 1e-6


def test_left_and_right_are_mirror_images(frames):
    """world 좌표 chirality 로 좌/우를 검사한다. palm frame 은 chirality 를 지우므로 쓸 수 없다."""
    valid = frames.filter(pl.col("valid"))
    joints = valid["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    sign = chirality(joints)
    hand = valid["handedness"].to_numpy()
    assert (sign[hand == "LEFT"] < 0).all()
    assert (sign[hand == "RIGHT"] > 0).all()


def test_camera_world_round_trip(frames):
    sub = frames.filter(pl.col("valid")).head(5000)
    world = sub["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    camera = sub["joints_camera"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    T = sub["camera_to_world"].to_numpy().reshape(-1, 4, 4).astype(np.float64)
    assert np.abs(transform_points(T, camera) - world).max() < 1e-6
    assert np.abs(transform_points(invert_transform(T), world) - camera).max() < 1e-6


def test_timestamps_increase_within_each_track(frames):
    for key, group in frames.partition_by(["sequence_id", "handedness"], as_dict=True).items():
        ts = group["timestamp_ns"].to_numpy()
        assert (np.diff(ts) > 0).all(), key


def test_bone_lengths_are_rigid_except_at_fingertips(frames):
    """MANO 의 joint-to-joint bone 은 pose 와 무관하게 고정이다.

    단, tip 5 개는 joint regressor 가 아니라 **mesh vertex** 에서 나오므로 굽힘에 따라
    길이가 변한다. WiLoR 출력도 같은 성질이라 배포와 일관되지만, §7.5 의 bone-length
    loss 는 tip bone 을 그대로 쓰면 안 된다.
    """
    from modules.delay_nowcasting.data.canonical_schema import BONES, FINGERTIPS, bone_lengths

    sub = frames.filter(pl.col("valid")).head(20000)
    joints = sub["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    lengths = bone_lengths(joints)
    spread = lengths.std(axis=0) / lengths.mean(axis=0)

    tip_bones = [i for i, (_, child) in enumerate(BONES) if child in FINGERTIPS]
    rigid_bones = [i for i in range(len(BONES)) if i not in tip_bones]
    assert len(tip_bones) == 5
    assert spread[rigid_bones].max() < 1e-4
    assert spread[tip_bones].max() < 0.10
