"""B 계획 §12 Window/causality tests."""
import numpy as np
import polars as pl
import pytest

from modules.delay_nowcasting.config import ResolvedConfig, config_hash
from modules.delay_nowcasting.data.build_windows import build_windows, gather
from modules.delay_nowcasting.data.canonical_schema import NUM_JOINTS, canonical_frame_schema

DT_NS = 33_333_333


def make_config(history_length=8, max_gap_ms=100.0, horizons=(0, 33, 66, 100, 133)):
    data = {
        "name": "synthetic",
        "window": {
            "history_length": history_length,
            "max_history_gap_ms": max_gap_ms,
            "horizon_tolerance_ms": 8.0,
            "native_horizons_ms": list(horizons),
        },
    }
    return ResolvedConfig("synthetic", None, data, config_hash(data))


def make_frames(tracks=(("SEQ_A", "LEFT"), ("SEQ_A", "RIGHT"), ("SEQ_B", "RIGHT")),
                n=40, invalid_at=(), start_ns=1_000_000_000):
    parts = []
    for seq, hand in tracks:
        ts = start_ns + np.arange(n, dtype=np.int64) * DT_NS
        valid = np.ones(n, dtype=bool)
        valid[list(invalid_at)] = False
        # track 마다 다른 상수 속도로 움직이게 해 track 혼입을 검출할 수 있게 한다
        speed = 1.0 if hand == "LEFT" else -2.0
        offset = 0.0 if seq == "SEQ_A" else 10.0
        base = offset + speed * (ts - start_ns) / 1e9
        joints = np.repeat(base[:, None], NUM_JOINTS * 3, axis=1).astype(np.float32)
        joints[~valid] = np.nan
        parts.append(pl.DataFrame({
            "sequence_id": np.full(n, seq), "subject_id": np.full(n, "P0001"),
            "frame_idx": np.arange(n), "timestamp_ns": ts,
            "handedness": np.full(n, hand), "track_id": np.full(n, f"{seq}:{hand}"),
            "valid": valid, "joints_world": joints, "joints_camera": joints,
            "visibility": np.where(valid[:, None], 1, 0).repeat(NUM_JOINTS, 1).astype(np.uint8),
            "camera_to_world": np.zeros((n, 16), np.float32),
            "source_dataset": np.full(n, "SYNTH"),
            "source_frame_key": [f"{seq}:{t}" for t in ts],
        }, schema=canonical_frame_schema()))
    return pl.concat(parts).sort(["sequence_id", "handedness", "timestamp_ns"])


def test_history_is_strictly_before_target():
    frames = make_frames()
    windows = build_windows(make_config(), frames)
    ts = frames["timestamp_ns"].to_numpy()
    history_ts = ts[windows["history_rows"].to_numpy()]
    target_ts = ts[windows["target_row"].to_numpy()]
    anchor_ts = ts[windows["anchor_row"].to_numpy()]
    horizon = windows["requested_horizon_ms"].to_numpy()

    assert (history_ts.max(axis=1) == anchor_ts).all()          # anchor 가 history 의 끝
    future = horizon > 0
    assert (history_ts[future].max(axis=1) < target_ts[future]).all()
    assert (target_ts[~future] == anchor_ts[~future]).all()      # horizon 0 은 anchor 자신


def test_windows_never_cross_sequence_or_hand_boundary():
    frames = make_frames()
    windows = build_windows(make_config(), frames)
    seq = frames["sequence_id"].to_numpy()
    hand = frames["handedness"].to_numpy()
    rows = np.concatenate([windows["history_rows"].to_numpy(),
                           windows["target_row"].to_numpy()[:, None]], axis=1)
    assert (seq[rows] == windows["sequence_id"].to_numpy()[:, None]).all()
    assert (hand[rows] == windows["handedness"].to_numpy()[:, None]).all()


def test_changing_a_future_frame_does_not_change_history():
    """target 이후를 바꿔도 history tensor 는 그대로여야 한다 (B 계획 §11 Phase 1)."""
    frames = make_frames()
    all_windows = build_windows(make_config(), frames)

    ts = frames["timestamp_ns"].to_numpy()
    # 앞쪽 절반의 window 만 보고, 그들의 target 보다 뒤인 frame 을 전부 오염시킨다
    cutoff = np.quantile(ts[all_windows["target_row"].to_numpy()], 0.5)
    windows = all_windows.filter(pl.Series(ts[all_windows["target_row"].to_numpy()] <= cutoff))
    before = gather(frames, windows)

    joints = frames["joints_world"].to_numpy().copy()
    future_rows = ts > cutoff
    assert future_rows.any() and windows.height > 0
    joints[future_rows] += 100.0
    perturbed = frames.with_columns(
        pl.Series("joints_world", joints, dtype=frames.schema["joints_world"]))

    after = gather(perturbed, windows)
    assert np.array_equal(before["history_joints"], after["history_joints"])
    assert np.array_equal(before["history_time_ms"], after["history_time_ms"])


def test_invalid_frames_are_skipped_and_gap_limit_applies():
    """무효 frame 은 history 에 들어가지 않고, 그 때문에 생긴 gap 은 한도로 걸러진다."""
    frames = make_frames(invalid_at=range(10, 14))          # 4 frame 결측 = 약 133 ms gap
    windows = build_windows(make_config(max_gap_ms=100.0), frames)
    rows = windows["history_rows"].to_numpy()
    valid = frames["valid"].to_numpy()
    assert valid[rows].all()
    assert valid[windows["target_row"].to_numpy()].all()
    assert (windows["max_history_gap_ms"].to_numpy() <= 100.0 + 1e-3).all()

    loose = build_windows(make_config(max_gap_ms=200.0), frames)
    assert loose.height > windows.height          # 한도를 풀면 sample 이 늘어야 한다


@pytest.mark.parametrize("history_length", [4, 8, 16])
def test_history_length_shapes(history_length):
    frames = make_frames(n=60)
    windows = build_windows(make_config(history_length=history_length), frames)
    batch = gather(frames, windows)
    assert batch["history_joints"].shape[1:] == (history_length, NUM_JOINTS, 3)
    assert batch["history_time_ms"].shape[1] == history_length
    assert (batch["history_time_ms"][:, -1] == 0).all()
    assert (np.diff(batch["history_time_ms"], axis=1) > 0).all()


def test_actual_horizon_matches_real_timestamp_difference():
    frames = make_frames()
    windows = build_windows(make_config(), frames)
    ts = frames["timestamp_ns"].to_numpy()
    expected = (ts[windows["target_row"].to_numpy()]
                - ts[windows["anchor_row"].to_numpy()]) / 1e6
    assert np.abs(expected - windows["horizon_ms"].to_numpy()).max() < 1e-3
    # 요청 horizon 과의 차이는 허용 오차 안이어야 한다
    assert np.abs(expected - windows["requested_horizon_ms"].to_numpy()).max() <= 8.0


def test_sample_keys_are_unique_and_reproducible():
    frames = make_frames()
    a = build_windows(make_config(), frames)
    b = build_windows(make_config(), frames)
    assert a["sample_key"].to_list() == b["sample_key"].to_list()
    assert a["sample_key"].n_unique() == a.height
