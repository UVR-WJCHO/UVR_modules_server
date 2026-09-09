"""Causal history/target window index.

  python -m modules.delay_nowcasting.data.build_windows \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml --split val

window 을 통째로 복사하지 않고 canonical frame 표의 **row index** 만 남긴다. 같은 frame 이
여러 sample 의 history 에 재사용되므로 index 쪽이 몇십 배 작고, 모든 방법이 정확히 같은
window 를 보게 된다(B 계획 §5: 모든 방법은 동일한 window cache 와 evaluator 를 쓴다).

B 계획 §4.4 의 규칙:
  1. sample 은 한 sequence, 한 hand, 한 track 안에서만 만든다
  2. history 는 target 이후 frame 을 절대 참조하지 않는다
  3. velocity 는 frame index 가 아니라 timestamp 차이로 계산한다
  4. history gap 이 임계값을 넘으면 sample 을 제외한다
  5. target 은 실제 미래 frame 을 쓴다 (interpolation 하지 않는다)
  8. horizon_ms 는 요청값이 아니라 실제로 고른 future frame 의 시간차다
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, ResolvedConfig, derive_seed, load_config, sample_key
from .canonical_schema import NUM_JOINTS

# canonical frame 표의 정렬 계약. row index 가 의미를 가지려면 읽는 쪽이 모두 이 순서를 쓴다.
FRAME_SORT_KEYS = ("sequence_id", "handedness", "timestamp_ns")
NS_PER_MS = 1_000_000.0


def load_canonical_frames(path: str | Path) -> pl.DataFrame:
    """canonical frame 표를 정해진 순서로 읽는다. row index 는 이 순서를 기준으로 한다."""
    return pl.read_parquet(path).sort(list(FRAME_SORT_KEYS))


def window_schema(history_length: int) -> dict:
    return {
        "sample_key": pl.Utf8,
        "sequence_id": pl.Utf8,
        "subject_id": pl.Utf8,
        "handedness": pl.Utf8,
        "track_id": pl.Utf8,
        "anchor_row": pl.Int32,
        "history_rows": pl.Array(pl.Int32, history_length),
        "target_row": pl.Int32,
        "target_row2": pl.Int32,          # 보간 상대 frame. 없으면 target_row 와 같다
        "target_weight": pl.Float32,      # 0 이면 보간 없음 (네이티브로 맞는 경우)
        "requested_horizon_ms": pl.Float32,
        "horizon_ms": pl.Float32,           # 실제 future frame 과의 시간차 (§4.4-8)
        "max_history_gap_ms": pl.Float32,
        "history_span_ms": pl.Float32,
    }


def _history_offsets(n_anchors: int, history_length: int, max_stride: int,
                     rng: np.random.Generator, target_fps: float | None = None,
                     source_fps: float = 30.0, min_fps: float | None = None) -> np.ndarray:
    """anchor 기준 history offset. max_stride>1 이면 프레임률을 섞는다 (§4.4 보강).

    데이터셋은 전부 30 fps 라 stride 를 키우면 그만큼 낮은 프레임률이 된다. 다만 정수
    stride 만 쓰면 30/15/10 fps 세 점만 학습된다. 배포 프레임률은 그 사이 어디든 될 수
    있으므로 **목표 간격을 연속 구간에서 뽑고**, 그 평균이 되도록 인접한 두 정수 stride 를
    섞는다. 목표 1.5 면 gap 이 1 과 2 를 오가 실효 20 fps 가 된다. 보간은 하지 않는다.

    window 의 30% 는 간격을 균일하게 둔다. 벤치마크 평가가 균일 30 fps 이므로 그 조건이
    학습 분포에서 빠지면 안 된다.
    """
    if target_fps is not None:
        # 평가용: 프레임률을 고정한다. 누적 위치를 반올림해 평균 간격이 정확히 맞도록
        # 배치하므로 20 fps 처럼 정수 stride 가 아닌 값도 만들 수 있다.
        stride = source_fps / float(target_fps)
        k = np.arange(history_length - 1, -1, -1)
        return np.tile(-np.rint(k * stride).astype(np.int64), (n_anchors, 1))

    if max_stride <= 1:
        return np.tile(np.arange(-history_length + 1, 1), (n_anchors, 1))

    # 실제 링크는 매 프레임 프레임률이 바뀌지 않는다. 한 세션 동안 대체로 일정한
    # 프레임률을 유지하다가 간헐적으로만 늦어진다. rng 는 track 별로 시드되므로
    # 기준 프레임률을 여기서 **한 번만** 뽑으면 track 전체가 그 값을 공유한다.
    n_gaps = history_length - 1
    base_fps = rng.uniform(min_fps or source_fps / max_stride, source_fps)
    stride = source_fps / base_fps
    k = np.arange(history_length - 1, -1, -1)
    base_gaps = np.diff(-np.rint(k * stride).astype(np.int64))    # (n_gaps,) 양수

    gaps = np.tile(base_gaps, (n_anchors, 1))
    # 간헐적 지연: 낮은 확률로 한두 프레임이 늦게 온다
    late = rng.random((n_anchors, n_gaps)) < 0.08
    gaps = gaps + late * rng.integers(1, 3, size=(n_anchors, n_gaps))
    gaps = np.clip(gaps, 1, max_stride + 1)

    offsets = np.zeros((n_anchors, history_length), dtype=np.int64)
    offsets[:, :-1] = -np.cumsum(gaps[:, ::-1], axis=1)[:, ::-1]
    return offsets


def _track_windows(rows: np.ndarray, timestamps: np.ndarray, horizons_ms: np.ndarray,
                   history_length: int, max_gap_ms: float, horizon_tolerance_ms: float,
                   max_stride: int = 1, rng: np.random.Generator | None = None,
                   target_fps: float | None = None, min_fps: float | None = None,
                   interpolate: bool = False):
    """한 track 의 유효 frame 만 받아 (anchor, history, target) 조합을 만든다.

    rows/timestamps 는 유효 frame 만, timestamp 오름차순이어야 한다.
    """
    n = len(rows)
    if n < history_length:
        return None

    # history: anchor 기준 과거 N 개 유효 frame. target 이후를 보지 않는 것이 자명하다.
    span_max = ((history_length - 1) * max_stride if target_fps is None
                else int(np.ceil((history_length - 1) * 30.0 / target_fps)))
    anchors = np.arange(span_max, n)
    if not len(anchors):
        return None
    offsets = _history_offsets(len(anchors), history_length, max_stride,
                               rng or np.random.default_rng(0), target_fps, 30.0, min_fps)
    hist_idx = anchors[:, None] + offsets                        # (A, N)
    hist_ts = timestamps[hist_idx]
    gaps = np.diff(hist_ts, axis=1) / NS_PER_MS
    max_gap = gaps.max(axis=1)
    span = (hist_ts[:, -1] - hist_ts[:, 0]) / NS_PER_MS
    keep = (max_gap <= max_gap_ms) & (hist_idx[:, 0] >= 0)
    anchors, hist_idx, max_gap, span = anchors[keep], hist_idx[keep], max_gap[keep], span[keep]
    if not len(anchors):
        return None

    anchor_ts = timestamps[anchors]
    out = []
    for horizon in horizons_ms:
        wanted = anchor_ts + np.int64(round(horizon * NS_PER_MS))
        # 요청 시각에 가장 가까운 frame. horizon>0 이면 anchor 이후여야 한다.
        right = np.searchsorted(timestamps, wanted, side="left")
        left = np.clip(right - 1, 0, n - 1)
        right = np.clip(right, 0, n - 1)
        pick_right = np.abs(timestamps[right] - wanted) <= np.abs(timestamps[left] - wanted)
        target = np.where(pick_right, right, left)
        if horizon > 0:
            # track 끝에서는 미래 frame 이 없다. clip 뒤 tolerance 검사에서 걸러진다.
            target = np.clip(np.maximum(target, anchors + 1), 0, n - 1)

        actual_ms = (timestamps[target] - anchor_ts) / NS_PER_MS
        ok = np.abs(actual_ms - horizon) <= horizon_tolerance_ms

        # 요청 시각에 맞는 frame 이 없으면 앞뒤 두 frame 을 선형 보간한다. **target 에만**
        # 적용한다 — history 를 보간하면 배포 추정기의 잡음 구조가 매끈해져, 이 논문이
        # 보인 "실제 추정기 출력으로 학습해야 한다" 는 조건이 깨진다.
        # 15 fps 데이터셋(HOI4D)에서 33 ms 배수 격자를 쓰기 위한 것이다.
        second, weight = target.copy(), np.zeros(len(target), dtype=np.float64)
        if horizon > 0 and interpolate:
            lo = np.clip(np.searchsorted(timestamps, wanted, side="right") - 1, 0, n - 1)
            hi = np.clip(lo + 1, 0, n - 1)
            span_ns = (timestamps[hi] - timestamps[lo]).astype(np.float64)
            gap_ok = (hi > lo) & (span_ns > 0) & (span_ns / NS_PER_MS <= max_gap_ms)
            inside = (timestamps[lo] <= wanted) & (wanted <= timestamps[hi])
            # lo == anchor 도 허용한다. anchor 와 다음 frame 사이를 메우는 경우이고,
            # anchor 도 실제 GT frame 이라 보간 끝점으로 쓸 수 있다 (33 ms @ 15 fps).
            fill = (~ok) & gap_ok & inside & (lo >= anchors)
            if fill.any():
                w = (wanted[fill] - timestamps[lo[fill]]).astype(np.float64) / span_ns[fill]
                target[fill], second[fill], weight[fill] = lo[fill], hi[fill], w
                actual_ms[fill] = horizon
                ok |= fill
        if horizon == 0:
            ok &= target == anchors
        if not ok.any():
            continue
        out.append({
            "anchor_row": rows[anchors[ok]],
            "history_rows": rows[hist_idx[ok]],
            "target_row": rows[target[ok]],
            "target_row2": rows[second[ok]],
            "target_weight": weight[ok].astype(np.float32),
            "requested_horizon_ms": np.full(ok.sum(), horizon, dtype=np.float32),
            "horizon_ms": actual_ms[ok].astype(np.float32),
            "max_history_gap_ms": max_gap[ok].astype(np.float32),
            "history_span_ms": span[ok].astype(np.float32),
        })
    return out


def build_windows(cfg: ResolvedConfig, frames: pl.DataFrame,
                  horizons_ms: list[float] | None = None,
                  target_fps: float | None = None) -> pl.DataFrame:
    window_cfg = cfg["window"]
    history_length = int(window_cfg["history_length"])
    max_gap_ms = float(window_cfg["max_history_gap_ms"])
    tolerance = float(window_cfg.get("horizon_tolerance_ms", 8.0))
    max_stride = int(window_cfg.get("history_max_stride", 1))
    min_fps = window_cfg.get("history_min_fps")
    # native_horizons_ms 는 실제 frame 이 존재하는 시각, interpolated_horizons_ms 는 두 frame
    # 사이를 선형으로 메워 만드는 시각이다. 15 fps 데이터셋에서 33 ms 배수 격자를 쓰기 위한
    # 것이며, 보간은 target 에만 적용된다(history 는 실제 추정기 출력 그대로 둔다).
    extra = list(window_cfg.get("interpolated_horizons_ms") or [])
    horizons = np.asarray(
        horizons_ms if horizons_ms is not None
        else sorted(set(list(window_cfg["native_horizons_ms"]) + extra)),
        dtype=np.float64)
    interpolate = bool(extra) or horizons_ms is not None

    frames = frames.with_row_index("row")
    valid = frames.filter(pl.col("valid"))

    parts = []
    for (sequence_id, handedness), group in valid.partition_by(
            ["sequence_id", "handedness"], as_dict=True).items():
        rows = group["row"].to_numpy().astype(np.int32)
        timestamps = group["timestamp_ns"].to_numpy()
        chunks = _track_windows(rows, timestamps, horizons, history_length,
                                max_gap_ms, tolerance, max_stride,
                                np.random.default_rng(
                                    derive_seed(0, sequence_id, handedness)),
                                target_fps, min_fps, interpolate)
        if not chunks:
            continue
        subject_id = group["subject_id"][0]
        track_id = group["track_id"][0]
        for chunk in chunks:
            n = len(chunk["anchor_row"])
            anchor_frames = frames["frame_idx"].to_numpy()[chunk["anchor_row"]]
            parts.append(pl.DataFrame({
                "sample_key": [
                    sample_key(sequence_id, handedness, int(f), float(h))
                    for f, h in zip(anchor_frames, chunk["requested_horizon_ms"])
                ],
                "sequence_id": np.full(n, sequence_id),
                "subject_id": np.full(n, subject_id),
                "handedness": np.full(n, handedness),
                "track_id": np.full(n, track_id),
                **chunk,
            }, schema=window_schema(history_length)))

    if not parts:
        return pl.DataFrame(schema=window_schema(history_length))
    return pl.concat(parts).sort(["sequence_id", "handedness", "anchor_row",
                                  "requested_horizon_ms"])


def gather(frames: pl.DataFrame, windows: pl.DataFrame,
           target_frames: pl.DataFrame | None = None) -> dict[str, np.ndarray]:
    """window index -> baseline/model 이 바로 쓰는 tensor 묶음.

    `target_frames` 를 주면 history 와 target 을 서로 다른 표에서 읽는다. 조건 C
    (history = WiLoR, target = GT) 를 이 인자로 표현한다. 두 표는 행 순서가 같아야 한다.
    """
    joints = frames["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3)
    visibility = frames["visibility"].to_numpy().reshape(-1, NUM_JOINTS)
    timestamps = frames["timestamp_ns"].to_numpy()
    target_joints = (joints if target_frames is None
                     else target_frames["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3))

    history_rows = windows["history_rows"].to_numpy()
    target_rows = windows["target_row"].to_numpy()
    history_ts = timestamps[history_rows]
    return {
        "history_joints": joints[history_rows],                       # (B, N, 21, 3)
        "history_visibility": visibility[history_rows],               # (B, N, 21)
        "history_dt_ms": np.diff(history_ts, axis=1, prepend=history_ts[:, :1]) / NS_PER_MS,
        "history_time_ms": (history_ts - history_ts[:, -1:]) / NS_PER_MS,   # anchor=0, 과거는 음수
        "target_joints": target_joints[target_rows],                  # (B, 21, 3)
        "horizon_ms": windows["horizon_ms"].to_numpy().astype(np.float64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="causal window index 생성")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=None)
    ap.add_argument("--frames-file", default=None,
                    help="기본은 canonical_frames_<split>.parquet. WiLoR history 조건에서는 "
                         "wilor_frames_<split>.parquet 를 준다")
    ap.add_argument("--tag", default="windows", help="출력 파일 이름 prefix")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="평가용. history 프레임률을 이 값으로 고정한다 "
                         "(생략하면 config 의 증강 설정을 쓴다)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    frames_file = args.frames_file or f"canonical_frames_{args.split}.parquet"
    frames = load_canonical_frames(out_dir / frames_file)
    windows = build_windows(cfg, frames, args.horizons_ms, args.target_fps)

    path = out_dir / f"{args.tag}_{args.split}.parquet"
    windows.write_parquet(path)
    print(f"config  : {cfg.path} (hash {cfg.hash})")
    print(f"frames  : {frames.height} rows ({frames['valid'].sum()} valid)")
    print(f"windows : {windows.height} samples -> {path}")
    if windows.height:
        print(windows.group_by("requested_horizon_ms").agg(
            pl.len().alias("n"),
            pl.col("horizon_ms").mean().round(2).alias("actual_mean_ms"),
            pl.col("horizon_ms").std().round(2).alias("actual_std_ms"),
        ).sort("requested_horizon_ms"))


if __name__ == "__main__":
    main()
