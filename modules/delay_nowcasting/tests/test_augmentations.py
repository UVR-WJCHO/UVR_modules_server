"""augmentation 이 계약대로 동작하는지 (B 계획 §4.5, §10.2 ablation 9)."""
import numpy as np
import pytest
import torch

from modules.delay_nowcasting.data.augmentations import NoiseConfig, corrupt_history
from modules.delay_nowcasting.data.canonical_schema import FINGERTIPS, NUM_JOINTS

B, N = 64, 8


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    history = torch.randn(B, N, NUM_JOINTS, 3, generator=g) * 0.05
    return history, torch.ones(B, N, NUM_JOINTS)


def test_everything_off_is_identity():
    history, visibility = _case()
    out, vis = corrupt_history(history, visibility, NoiseConfig())
    assert torch.equal(out, history)
    assert torch.equal(vis, visibility)


def test_each_component_can_be_disabled_independently():
    history, visibility = _case()
    full = NoiseConfig(jitter_m=0.004, drift_m=0.01, bias_m=0.01,
                       drop_rate=0.2, burst_drop_rate=0.05)
    for name in ("jitter", "drift", "bias", "drop", "burst"):
        without = NoiseConfig(**{**full.__dict__,
                                 "enabled": tuple(x for x in full.enabled if x != name)})
        out, vis = corrupt_history(history, visibility, without,
                                   torch.Generator().manual_seed(0))
        if name in ("drop", "burst"):
            continue
        assert not torch.equal(out, history), name


def test_jitter_scale_matches_request():
    history, visibility = _case()
    sigma = 0.004
    out, _ = corrupt_history(history, visibility,
                             NoiseConfig(jitter_m=sigma, enabled=("jitter",)),
                             torch.Generator().manual_seed(0))
    per_axis = (out - history).std()
    assert abs(float(per_axis) - sigma) / sigma < 0.05


def test_fingertip_jitter_is_larger_when_requested():
    history, visibility = _case()
    out, _ = corrupt_history(history, visibility,
                             NoiseConfig(jitter_m=0.004, fingertip_jitter_scale=3.0,
                                         enabled=("jitter",)),
                             torch.Generator().manual_seed(0))
    delta = (out - history).norm(dim=-1)
    tips = delta[:, :, list(FINGERTIPS)].mean()
    others = delta[:, :, [j for j in range(NUM_JOINTS) if j not in FINGERTIPS]].mean()
    assert float(tips) > 2.0 * float(others)


def test_drift_is_temporally_correlated_unlike_jitter():
    history, visibility = _case()
    g = torch.Generator().manual_seed(0)
    drifted, _ = corrupt_history(history, visibility,
                                 NoiseConfig(drift_m=0.01, enabled=("drift",)), g)
    jittered, _ = corrupt_history(history, visibility,
                                  NoiseConfig(jitter_m=0.01, enabled=("jitter",)), g)

    def lag1(x):
        e = (x - history).reshape(B, N, -1)
        a, b = e[:, :-1].flatten(), e[:, 1:].flatten()
        return float(np.corrcoef(a.numpy(), b.numpy())[0, 1])

    assert lag1(drifted) > 0.5
    assert abs(lag1(jittered)) < 0.1


def test_dropped_frames_hold_the_previous_pose_not_zero():
    history, visibility = _case()
    out, vis = corrupt_history(history, visibility,
                               NoiseConfig(drop_rate=0.5, enabled=("drop",)),
                               torch.Generator().manual_seed(0))
    missing = vis[:, :, 0] == 0
    assert missing.any()
    assert not torch.isclose(out[missing], torch.zeros(3)).all()
    # 누락된 프레임은 직전 프레임과 같아야 한다
    b, t = torch.nonzero(missing, as_tuple=True)
    later = t > 0
    assert torch.allclose(out[b[later], t[later]], out[b[later], t[later] - 1])


def test_anchor_is_never_dropped():
    """anchor 가 없으면 forecast 자체가 불가능하다."""
    history, visibility = _case()
    _, vis = corrupt_history(history, visibility,
                             NoiseConfig(drop_rate=0.9, burst_drop_rate=0.5,
                                         enabled=("drop", "burst")),
                             torch.Generator().manual_seed(0))
    assert bool((vis[:, -1] == 1).all())


def test_from_measurement_maps_statistics_to_parameters():
    stats = {
        "per_joint_jitter_mm": [3.0] * NUM_JOINTS,
        "jitter_mm_median": 3.5, "drift_mm_median": 9.0, "bias_mm_median": 13.0,
        "drop_rate": 0.18, "burst_drop_median": 2.0, "burst_drop_p95": 6.0,
    }
    for tip in FINGERTIPS:
        stats["per_joint_jitter_mm"][tip] = 6.0
    config = NoiseConfig.from_measurement(stats)
    assert config.jitter_m == pytest.approx(0.0035)
    assert config.bias_m == pytest.approx(0.013)
    assert config.drop_rate == pytest.approx(0.18)
    assert config.fingertip_jitter_scale == pytest.approx(2.0)
    assert config.burst_drop_max == 6
