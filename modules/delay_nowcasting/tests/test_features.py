"""torch feature/base 가 numpy baseline 과 같은 값을 내는지, 모델이 계약을 지키는지."""
import numpy as np
import pytest
import torch

from modules.delay_nowcasting.data.canonical_schema import NUM_JOINTS
from modules.delay_nowcasting.methods import baselines, features
from modules.delay_nowcasting.methods.residual_mlp import ResidualMLP
from modules.delay_nowcasting.training.losses import RIGID_BONES, nowcast_loss

B, N = 7, 8
rng = np.random.default_rng(0)


@pytest.fixture
def case():
    dt = rng.uniform(28.0, 40.0, size=(B, N - 1))
    t = np.concatenate([np.zeros((B, 1)), np.cumsum(dt, axis=1)], axis=1)
    times = t - t[:, -1:]
    joints = np.cumsum(rng.normal(size=(B, N, NUM_JOINTS, 3)) * 0.01, axis=1) + 0.1
    horizon = rng.uniform(0, 150, size=B)
    return joints, times, horizon


def _torch(*arrays):
    return [torch.as_tensor(a, dtype=torch.float64) for a in arrays]


def test_torch_cv_base_matches_numpy(case):
    joints, times, horizon = case
    for n_fit in (2, 4):
        expected = baselines.constant_velocity(joints, times, horizon, n_fit=n_fit)
        actual = features.constant_velocity_base(*_torch(joints, times, horizon), n_fit=n_fit)
        assert np.abs(actual.numpy() - expected).max() < 1e-10, n_fit


def test_torch_polynomial_fit_matches_numpy(case):
    joints, times, horizon = case
    for degree in (1, 2):
        expected = baselines._fit_polynomial(joints, times, degree=degree, n_fit=4)
        actual = features.fit_polynomial(*_torch(joints, times), degree=degree, n_fit=4)
        assert np.abs(actual.numpy() - expected).max() < 1e-9, degree


def test_horizon_encoding_shape_and_endpoints():
    clamp = features.HORIZON_CLAMP_MS
    horizon = torch.tensor([0.0, clamp / 2, clamp, clamp * 2])
    encoding = features.horizon_encoding(horizon)
    assert encoding.shape == (4, features.HORIZON_ENCODING_DIM)
    # clamp 를 넘는 값은 clamp 와 같은 encoding 이 된다 (§7.4)
    assert torch.allclose(encoding[2], encoding[3])
    assert torch.allclose(encoding[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]))


def test_horizon_clamp_covers_the_configured_grid():
    """encoding 정규화 기준이 grid 상한보다 작으면 그 위 horizon 이 뭉개진다."""
    import yaml
    from modules.delay_nowcasting.config import REPO_ROOT

    for name in ("hot3d_v1", "dexycb_v1", "mixed_v1"):
        cfg = yaml.safe_load(
            (REPO_ROOT / f"modules/delay_nowcasting/configs/data/{name}.yaml").read_text())
        largest = max(cfg["evaluation"]["horizons_ms"])
        assert largest <= features.HORIZON_CLAMP_MS, f"{name}: {largest} > clamp"


def test_feature_dim_matches_builder(case):
    joints, times, horizon = case
    built = features.build_features(*_torch(joints, times, horizon))
    assert built.shape == (B, features.feature_dim(N, NUM_JOINTS))


def test_features_are_translation_invariant(case):
    """world 원점을 옮겨도 feature 는 변하지 않아야 한다."""
    joints, times, horizon = case
    shifted = joints + np.array([3.0, -2.0, 5.0])
    a = features.build_features(*_torch(joints, times, horizon))
    b = features.build_features(*_torch(shifted, times, horizon))
    assert torch.allclose(a, b, atol=1e-9)


def test_untrained_model_equals_constant_velocity_base(case):
    """출력층을 0 으로 초기화했으므로 학습 전 모델은 정확히 CV base 다."""
    joints, times, horizon = case
    model = ResidualMLP(history_length=N).double().eval()
    with torch.no_grad():
        prediction = model(*_torch(joints, times, horizon))
    expected = baselines.constant_velocity(joints, times, horizon, n_fit=2)
    assert np.abs(prediction.numpy() - expected).max() < 1e-10


def test_model_parameter_budget():
    """B 계획 §7.1: parameter < 1M."""
    assert ResidualMLP(history_length=8).num_parameters < 1_000_000


def test_bone_loss_excludes_fingertips():
    """MANO tip 은 mesh vertex 라 길이가 변한다 (Phase 1). 고정 제약을 걸면 안 된다."""
    from modules.delay_nowcasting.data.canonical_schema import BONES, FINGERTIPS

    assert len(RIGID_BONES) == len(BONES) - 5
    assert all(child not in FINGERTIPS for _, child in RIGID_BONES)


def test_loss_is_zero_for_perfect_prediction(case):
    joints, times, horizon = case
    target = torch.as_tensor(joints[:, -1], dtype=torch.float32)
    anchor = target.clone()
    total, terms = nowcast_loss(target.clone(), target, anchor,
                                torch.as_tensor(horizon, dtype=torch.float32))
    assert float(total) == pytest.approx(0.0, abs=1e-9)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in terms.values())


def test_loss_grows_with_error(case):
    joints, times, horizon = case
    target = torch.as_tensor(joints[:, -1], dtype=torch.float32)
    anchor = torch.as_tensor(joints[:, -2], dtype=torch.float32)
    h = torch.as_tensor(horizon, dtype=torch.float32)
    small, _ = nowcast_loss(target + 0.001, target, anchor, h)
    large, _ = nowcast_loss(target + 0.010, target, anchor, h)
    assert float(large) > float(small) > 0.0
