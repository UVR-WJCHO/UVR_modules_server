"""B 계획 §11 Phase 1 / §12 Baseline tests."""
import numpy as np
import pytest

from modules.delay_nowcasting.data.canonical_schema import NUM_JOINTS
from modules.delay_nowcasting.methods import baselines

N = 8
B = 5
rng = np.random.default_rng(0)


def _times(irregular=False):
    """(B, N) anchor=0 기준 상대 시각 ms."""
    if not irregular:
        base = np.arange(-N + 1, 1) * 33.333
        return np.tile(base, (B, 1))
    dt = rng.uniform(28.0, 40.0, size=(B, N - 1))
    t = np.concatenate([np.zeros((B, 1)), np.cumsum(dt, axis=1)], axis=1)
    return t - t[:, -1:]


def _linear_motion(times, velocity, start):
    """등속 운동: J(t) = start + v * t."""
    return start[:, None] + velocity[:, None] * (times[..., None, None] / 1000.0)


def _accel_motion(times, velocity, accel, start):
    t = times[..., None, None] / 1000.0
    return start[:, None] + velocity[:, None] * t + 0.5 * accel[:, None] * t ** 2


@pytest.fixture
def static_case():
    times = _times()
    start = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.05
    joints = np.repeat(start[:, None], N, axis=1)
    return joints, times, np.full(B, 100.0), start


def test_static_sequence_hold_and_cv_agree(static_case):
    joints, times, horizon, start = static_case
    for name, fn in baselines.BASELINES.items():
        pred = fn(joints, times, horizon)
        assert np.abs(pred - start).max() < 1e-9, name


@pytest.mark.parametrize("irregular", [False, True])
def test_constant_velocity_is_exact_on_constant_velocity_motion(irregular):
    """timestamp-aware CV 는 등속 운동에서 오차가 numerical zero 여야 한다."""
    times = _times(irregular)
    velocity = rng.normal(size=(B, NUM_JOINTS, 3))
    start = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.05
    joints = _linear_motion(times, velocity, start)
    horizon = rng.uniform(0, 150, size=B)
    target = start + velocity * (horizon[:, None, None] / 1000.0)

    for name in ("cv_2frame", "cv_robust", "const_accel"):
        pred = baselines.BASELINES[name](joints, times, horizon)
        assert np.abs(pred - target).max() < 1e-9, f"{name} irregular={irregular}"


def test_constant_acceleration_is_exact_on_accelerating_motion():
    times = _times()
    velocity = rng.normal(size=(B, NUM_JOINTS, 3))
    accel = rng.normal(size=(B, NUM_JOINTS, 3)) * 5.0
    start = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.05
    joints = _accel_motion(times, velocity, accel, start)
    horizon = rng.uniform(0, 150, size=B)
    h = horizon[:, None, None] / 1000.0
    target = start + velocity * h + 0.5 * accel * h ** 2

    pred = baselines.BASELINES["const_accel"](joints, times, horizon)
    assert np.abs(pred - target).max() < 1e-8
    # 등가속 운동에서 CV 는 틀려야 한다(테스트가 자명하게 통과하지 않는지 확인)
    cv = baselines.BASELINES["cv_robust"](joints, times, horizon)
    assert np.abs(cv - target).max() > 1e-4


def test_horizon_zero_is_the_anchor():
    """horizon=0 에서 Hold 오차는 numerical zero (B 계획 §11 Phase 1)."""
    times = _times()
    joints = rng.normal(size=(B, N, NUM_JOINTS, 3)) * 0.05
    horizon = np.zeros(B)
    for name in ("hold", "cv_2frame", "cv_robust", "const_accel"):
        pred = baselines.BASELINES[name](joints, times, horizon)
        assert np.abs(pred - joints[:, -1]).max() < 1e-9, name


def test_clipping_bounds_velocity_without_nan():
    times = _times()
    velocity = rng.normal(size=(B, NUM_JOINTS, 3)) * 50.0     # 비현실적으로 빠름
    start = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.05
    joints = _linear_motion(times, velocity, start)
    horizon = np.full(B, 100.0)

    raw = baselines.constant_velocity(joints, times, horizon, n_fit=4)
    clipped = baselines.constant_velocity(joints, times, horizon, n_fit=4, clip_speed_mps=2.0)
    assert np.isfinite(clipped).all()
    displacement = np.linalg.norm(clipped - joints[:, -1], axis=-1)
    assert displacement.max() <= 2.0 * 0.1 + 1e-9
    assert np.linalg.norm(raw - joints[:, -1], axis=-1).max() > displacement.max()


def test_kalman_tracks_constant_velocity():
    """KF 는 등속 운동을 정확히는 못 맞춰도 Hold 보다는 훨씬 나아야 한다."""
    times = _times()
    velocity = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.5
    start = rng.normal(size=(B, NUM_JOINTS, 3)) * 0.05
    joints = _linear_motion(times, velocity, start)
    horizon = np.full(B, 100.0)
    target = start + velocity * 0.1

    kf_err = np.linalg.norm(baselines.BASELINES["kalman_cv"](joints, times, horizon) - target,
                            axis=-1).mean()
    hold_err = np.linalg.norm(baselines.BASELINES["hold"](joints, times, horizon) - target,
                              axis=-1).mean()
    assert kf_err < 0.1 * hold_err


def test_no_method_reads_the_future():
    """history 마지막 이후는 존재하지 않지만, anchor 이전 frame 변조가 결과를 바꾸는지로
    causal 경로가 실제로 history 를 쓰는지 확인한다."""
    times = _times()
    joints = rng.normal(size=(B, N, NUM_JOINTS, 3)) * 0.05
    horizon = np.full(B, 100.0)
    perturbed = joints.copy()
    perturbed[:, 0] += 1.0                                    # 가장 오래된 frame 만 변경

    assert np.abs(baselines.BASELINES["hold"](joints, times, horizon)
                  - baselines.BASELINES["hold"](perturbed, times, horizon)).max() == 0.0
    assert np.abs(baselines.BASELINES["cv_2frame"](joints, times, horizon)
                  - baselines.BASELINES["cv_2frame"](perturbed, times, horizon)).max() == 0.0
    # n_fit=4 는 최근 4 frame 만 보므로 가장 오래된 frame 변조에 영향받지 않아야 한다
    assert np.abs(baselines.BASELINES["cv_robust"](joints, times, horizon)
                  - baselines.BASELINES["cv_robust"](perturbed, times, horizon)).max() == 0.0
