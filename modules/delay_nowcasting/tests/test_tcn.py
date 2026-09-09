"""TCN 의 causality 와 대조군 구성 확인 (B 계획 §7.2, §12 Model tests)."""
import numpy as np
import pytest
import torch

from modules.delay_nowcasting.data.canonical_schema import NUM_JOINTS, WRIST
from modules.delay_nowcasting.methods import baselines
from modules.delay_nowcasting.methods.decomposition import compose, decompose
from modules.delay_nowcasting.methods.tcn_lite import TCNLite

B, N = 6, 8
rng = np.random.default_rng(0)


@pytest.fixture
def case():
    dt = rng.uniform(28.0, 40.0, size=(B, N - 1))
    t = np.concatenate([np.zeros((B, 1)), np.cumsum(dt, axis=1)], axis=1)
    times = torch.as_tensor(t - t[:, -1:], dtype=torch.float64)
    joints = torch.as_tensor(
        np.cumsum(rng.normal(size=(B, N, NUM_JOINTS, 3)) * 0.01, axis=1) + 0.1,
        dtype=torch.float64)
    horizon = torch.as_tensor(rng.uniform(0, 150, size=B), dtype=torch.float64)
    return joints, times, horizon


def _model(**kwargs):
    return TCNLite(history_length=N, **kwargs).double().eval()


def test_untrained_model_equals_constant_velocity_base(case):
    """출력층을 0 으로 초기화했으므로 학습 전에는 정확히 CV base 여야 한다."""
    joints, times, horizon = case
    with torch.no_grad():
        prediction = _model()(joints, times, horizon)
    expected = baselines.constant_velocity(joints.numpy(), times.numpy(), horizon.numpy(),
                                           n_fit=2)
    assert np.abs(prediction.numpy() - expected).max() < 1e-9


def test_decomposed_model_also_starts_at_the_base(case):
    joints, times, horizon = case
    with torch.no_grad():
        prediction = _model(decomposed=True)(joints, times, horizon)
    expected = baselines.constant_velocity(joints.numpy(), times.numpy(), horizon.numpy(),
                                           n_fit=2)
    assert np.abs(prediction.numpy() - expected).max() < 1e-9


def test_no_future_leakage_through_the_causal_stack(case):
    """anchor 이후 입력은 존재하지 않지만, 내부 conv 가 오른쪽 padding 을 쓰면
    history 뒤쪽 프레임이 앞쪽 출력에 새어 들어간다. 마지막 프레임만 바꿔서
    그 전 시점 state 가 그대로인지 본다."""
    joints, times, horizon = case
    model = _model()
    model.train(False)

    features = model.per_frame_features(joints, times, None)
    normalized = (features - model.feature_mean) / model.feature_std
    with torch.no_grad():
        x = model.frame_encoder(normalized).transpose(1, 2)
        for block in model.blocks:
            x = block(x)
        before = x.clone()

        poisoned = normalized.clone()
        poisoned[:, -1] += 100.0                       # 마지막(anchor) 프레임만 오염
        y = model.frame_encoder(poisoned).transpose(1, 2)
        for block in model.blocks:
            y = block(y)

    # anchor 이전 시점의 state 는 전혀 변하지 않아야 한다
    assert torch.allclose(before[:, :, :-1], y[:, :, :-1], atol=1e-10)
    # anchor state 는 당연히 변해야 한다(테스트가 자명하게 통과하지 않는지 확인)
    assert not torch.allclose(before[:, :, -1], y[:, :, -1])


def test_horizon_encoding_ablation_changes_dependence(case):
    joints, times, horizon = case
    conditioned = _model(use_horizon_encoding=True)
    fixed = _model(use_horizon_encoding=False)
    # 학습 전에는 둘 다 base 라 horizon 에 따라 base 만 달라진다. 구조 차이만 확인한다.
    assert any("horizon_encoder" in name for name, _ in conditioned.named_parameters())
    assert not any("horizon_encoder" in name for name, _ in fixed.named_parameters())


def test_cv_base_ablation_falls_back_to_hold(case):
    joints, times, horizon = case
    with torch.no_grad():
        prediction = _model(use_cv_base=False)(joints, times, horizon)
    assert torch.allclose(prediction, joints[:, -1], atol=1e-12)


def test_parameter_counts_are_comparable_across_variants():
    """B 계획 §6.5: 이득이 모델 용량 차이에서 오지 않도록 맞춘다."""
    direct = TCNLite(history_length=N).num_parameters
    fixed = TCNLite(history_length=N, use_horizon_encoding=False).num_parameters
    decomposed = TCNLite(history_length=N, decomposed=True).num_parameters
    assert abs(direct - decomposed) / direct < 0.05
    # horizon encoder 만큼은 줄어드는 게 정상이지만 10% 를 넘으면 안 된다
    assert 0 < (direct - fixed) / direct < 0.10


def test_decompose_compose_round_trip(case):
    joints, _, _ = case
    wrist, local = decompose(joints[:, -1])
    restored = compose(wrist, local)
    assert torch.allclose(restored, joints[:, -1], atol=1e-12)
    assert torch.allclose(local[:, WRIST], torch.zeros(3, dtype=local.dtype), atol=1e-12)


def test_handedness_changes_the_output_after_perturbing_weights(case):
    joints, times, horizon = case
    model = _model()
    torch.nn.init.normal_(model.head.weight, std=0.01)
    left = torch.zeros(B, dtype=torch.long)
    right = torch.ones(B, dtype=torch.long)
    with torch.no_grad():
        assert not torch.allclose(model(joints, times, horizon, left),
                                  model(joints, times, horizon, right))
