"""학습된 모델이 target 을 볼 수 없음을 데이터 경로 끝단에서 확인한다.

앞선 window 테스트는 index 수준의 causality 를 본다. 여기서는 실제 frame 표 -> gather ->
모델 forward 전체 경로에서, target frame 을 바꿔도 예측이 한 비트도 안 변하는지 본다.
"""
import numpy as np
import polars as pl
import pytest
import torch

from modules.delay_nowcasting.config import REPO_ROOT, load_config
from modules.delay_nowcasting.data.canonical_schema import NUM_JOINTS
from modules.delay_nowcasting.training.dataset import load_split

CONFIG = load_config(REPO_ROOT / "modules/delay_nowcasting/configs/data/hot3d_v1.yaml")
CACHE = REPO_ROOT / CONFIG["output_root"] / CONFIG.name
CHECKPOINT = CACHE / "residual_mlp" / "seed0" / "best_model.pt"

pytestmark = pytest.mark.skipif(
    not (CACHE / "windows_val.parquet").exists(), reason="window cache 미생성")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data():
    return load_split(CACHE, "val", DEVICE)


def _model():
    """현재 horizon clamp 로 학습된 checkpoint 만 쓴다.

    clamp 를 바꾸면 encoding 스케일이 달라져 예전 checkpoint 는 의미가 없다.
    checkpoint 에 기록된 config 로 판별한다.
    """
    from modules.delay_nowcasting.methods import features
    from modules.delay_nowcasting.training import checkpoint as ckpt

    if not CHECKPOINT.exists():
        pytest.skip("checkpoint 미생성")
    model, payload = ckpt.load(CHECKPOINT, DEVICE)
    trained_clamp = payload.get("horizon_clamp_ms")
    if trained_clamp is not None and trained_clamp != features.HORIZON_CLAMP_MS:
        pytest.skip(f"checkpoint 가 clamp {trained_clamp} 로 학습됨 "
                    f"(현재 {features.HORIZON_CLAMP_MS}). 재학습 필요")
    if trained_clamp is None:
        pytest.skip("clamp 기록이 없는 옛 checkpoint")
    return model


def test_history_rows_are_strictly_before_target_row(data):
    ts = data.timestamp_ns
    history_ts = ts[data.history_rows]
    target_ts = ts[data.target_row]
    horizon = data.horizon_ms
    future = horizon > 0
    assert bool((history_ts.amax(dim=1)[future] < target_ts[future]).all())


def test_prediction_ignores_the_target_frame(data):
    """target frame 의 좌표를 오염시켜도 예측은 그대로여야 한다."""
    model = _model()
    index = torch.arange(0, 20000, device=DEVICE)
    batch = data.batch(index)
    with torch.no_grad():
        before = model(batch["history"], batch["history_time_ms"], batch["horizon_ms"])

    poisoned = data.joints.clone()
    target_rows = data.target_row[index]
    # history 로도 쓰이는 row 는 제외하고, 순수하게 target 으로만 쓰인 row 를 오염시킨다
    history_rows = torch.unique(data.history_rows[index])
    mask = ~torch.isin(target_rows, history_rows)
    assert bool(mask.any())
    poisoned[target_rows[mask]] += 1.0

    data.joints, original = poisoned, data.joints
    try:
        after_batch = data.batch(index)
        with torch.no_grad():
            after = model(after_batch["history"], after_batch["history_time_ms"],
                          after_batch["horizon_ms"])
    finally:
        data.joints = original

    assert torch.equal(before, after)
    # 그러나 오차는 변해야 한다(테스트가 자명하게 통과하지 않는지 확인)
    changed = (after_batch["target"][mask] - batch["target"][mask]).abs().max()
    assert float(changed) > 0.5


def test_val_subjects_are_disjoint_from_train():
    train = pl.read_parquet(CACHE / "windows_train.parquet", columns=["subject_id"])
    val = pl.read_parquet(CACHE / "windows_val.parquet", columns=["subject_id"])
    assert set(train["subject_id"].unique()) & set(val["subject_id"].unique()) == set()


def test_model_at_horizon_zero_stays_near_the_anchor(data):
    """horizon 0 은 예측이 아니라 항등이어야 한다. 크게 벗어나면 조건화가 깨진 것이다."""
    model = _model()
    zero = np.flatnonzero(data.requested_horizon_ms == 0.0)[:20000]
    index = torch.as_tensor(zero, device=DEVICE)
    batch = data.batch(index)
    with torch.no_grad():
        prediction = model(batch["history"], batch["history_time_ms"], batch["horizon_ms"])
    error_mm = torch.linalg.norm(prediction - batch["anchor"], dim=-1).mean() * 1000
    assert float(error_mm) < 1.0
