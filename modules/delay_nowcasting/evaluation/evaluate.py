"""Baseline / 모델 평가와 결과 표 생성.

  python -m modules.delay_nowcasting.evaluation.evaluate \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml --split val

B 계획 §9.5: frame 을 독립 표본으로 보지 않는다. sequence 단위로 먼저 집계하고,
같은 sequence 에서 방법끼리 paired 로 비교한다. 이 스크립트는 sequence 단위 long-format
표까지 만들고, 요약과 통계는 summarize.py / bootstrap.py 가 맡는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.build_windows import gather, load_canonical_frames
from ..methods.baselines import BASELINES
from .metrics import articulation_speed_mps, sample_metrics, tercile_labels, wrist_speed_mps
from .result_schema import metric_result_schema

CHUNK = 100_000
METRIC_NAMES = ("abs_mpjpe_mm", "root_rel_mpjpe_mm", "wrist_err_mm", "fingertip_mpjpe_mm")


def torch_method(checkpoint_path: Path, device: str = "cuda"):
    """학습된 checkpoint 를 baseline 과 같은 numpy 인터페이스로 감싼다.

    모든 방법이 같은 window 와 같은 evaluator 를 통과해야 비교가 성립한다 (§5).
    """
    import torch

    from ..training import checkpoint as ckpt

    model, payload = ckpt.load(checkpoint_path, device)

    def predict(history, history_time_ms, horizon_ms, handedness=None, visibility=None):
        def to(x, dtype=torch.float32):
            return None if x is None else torch.as_tensor(x, dtype=dtype, device=device)

        with torch.no_grad():
            out = model(to(history), to(history_time_ms), to(horizon_ms),
                        to(handedness, torch.long), to(visibility))
        return out.double().cpu().numpy()

    predict.needs_context = True
    return predict, payload


def _predict_all(frames: pl.DataFrame, windows: pl.DataFrame,
                 methods: dict[str, callable], target_frames: pl.DataFrame | None = None,
                 noise=None, noise_seed: int = 1234
                 ) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """chunk 단위로 모든 방법을 돌려 sample 별 metric 과 motion 통계를 모은다."""
    n = windows.height
    values = {name: {m: np.empty(n, dtype=np.float64) for m in METRIC_NAMES}
              for name in methods}
    motion = {"wrist_speed_mps": np.empty(n), "articulation_speed_mps": np.empty(n)}

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        batch = gather(frames, windows[start:stop], target_frames)
        if noise is not None:
            import torch

            from ..data.augmentations import corrupt_history

            generator = torch.Generator().manual_seed(noise_seed + start)
            corrupted, vis = corrupt_history(
                torch.as_tensor(batch["history_joints"], dtype=torch.float64),
                torch.as_tensor(batch["history_visibility"], dtype=torch.float64),
                noise, generator)
            batch["history_joints"] = corrupted.numpy()
            batch["history_visibility"] = vis.numpy()
        history = batch["history_joints"].astype(np.float64)
        times = batch["history_time_ms"]
        horizon = batch["horizon_ms"]
        target = batch["target_joints"].astype(np.float64)

        motion["wrist_speed_mps"][start:stop] = wrist_speed_mps(history, times)
        motion["articulation_speed_mps"][start:stop] = articulation_speed_mps(history, times)
        handedness = (windows["handedness"][start:stop].to_numpy() == "RIGHT").astype("int64")
        visibility = batch["history_visibility"].astype(np.float32)
        for name, fn in methods.items():
            prediction = (fn(history, times, horizon, handedness, visibility)
                          if getattr(fn, "needs_context", False)
                          else fn(history, times, horizon))
            metrics = sample_metrics(prediction, target)
            for metric, value in metrics.items():
                values[name][metric][start:stop] = value
    return values, motion


def _long_table(windows: pl.DataFrame, values: dict, motion: dict, split: str,
                config_hash: str, model_versions: dict[str, str] | None = None,
                seeds: dict[str, int] | None = None,
                history_source: str = "gt") -> pl.DataFrame:
    """sample 별 metric -> (method, sequence, hand, horizon, group) 단위 long 표."""
    keys = windows.select(["sequence_id", "subject_id", "handedness",
                           "requested_horizon_ms"])
    groups = {
        "all": np.full(windows.height, "all"),
        "wrist_speed_tercile": tercile_labels(motion["wrist_speed_mps"]),
        "articulation_speed_tercile": tercile_labels(motion["articulation_speed_mps"]),
    }
    hold_error = values["hold"]["abs_mpjpe_mm"]

    parts = []
    for method, metrics in values.items():
        per_sample = keys.with_columns(
            [pl.Series(m, metrics[m]) for m in METRIC_NAMES]
            + [pl.Series("worse_than_hold", (metrics["abs_mpjpe_mm"] > hold_error).astype(float))]
            + [pl.Series(f"__group_{key}", value) for key, value in groups.items()]
        )
        for group_key in groups:
            agg = per_sample.group_by(
                ["sequence_id", "subject_id", "handedness", "requested_horizon_ms",
                 f"__group_{group_key}"]
            ).agg(
                [pl.col(m).mean().alias(m) for m in METRIC_NAMES]
                + [pl.col("abs_mpjpe_mm").quantile(0.95).alias("p95_abs_mpjpe_mm"),
                   pl.col("worse_than_hold").mean().alias("worse_than_hold_ratio"),
                   pl.len().alias("n_frames")]
            ).rename({f"__group_{group_key}": "group_value"})

            long = agg.unpivot(
                index=["sequence_id", "subject_id", "handedness", "requested_horizon_ms",
                       "group_value", "n_frames"],
                on=list(METRIC_NAMES) + ["p95_abs_mpjpe_mm", "worse_than_hold_ratio"],
                variable_name="metric", value_name="value",
            ).with_columns(
                method=pl.lit(method),
                model_version=pl.lit((model_versions or {}).get(method, f"baseline/{method}")),
                config_hash=pl.lit(config_hash),
                seed=pl.lit((seeds or {}).get(method), dtype=pl.Int64),
                split=pl.lit(split),
                history_source=pl.lit(history_source),
                interpolated_target=pl.lit(False),
                group_key=pl.lit(group_key),
            ).rename({"requested_horizon_ms": "horizon_ms"})
            parts.append(long.select(list(metric_result_schema())).cast(metric_result_schema()))
    return pl.concat(parts)


def summarize(table: pl.DataFrame, metric: str = "abs_mpjpe_mm",
              group_key: str = "all", group_value: str = "all") -> pl.DataFrame:
    """sequence 평균을 다시 평균낸 method x horizon 표 (§9.5: sequence 가 표본 단위)."""
    return (table
            .filter((pl.col("metric") == metric) & (pl.col("group_key") == group_key)
                    & (pl.col("group_value") == group_value))
            .group_by(["method", "horizon_ms"])
            .agg(pl.col("value").mean().round(2).alias(metric),
                 pl.col("sequence_id").n_unique().alias("n_seq"))
            .pivot(on="horizon_ms", index="method", values=metric)
            .sort("method"))


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline 평가")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--methods", nargs="*", default=list(BASELINES))
    ap.add_argument("--checkpoint", nargs="*", default=[],
                    help="학습된 checkpoint 경로. method 이름은 <config name>/<디렉터리명>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="baseline", help="결과 파일 이름 prefix")
    ap.add_argument("--history-frames", default=None,
                    help="history 를 읽을 frame 표. 기본은 canonical_frames_<split>.parquet")
    ap.add_argument("--target-frames", default=None,
                    help="target 을 읽을 frame 표. 기본은 history 와 같다. 조건 C 에서는 "
                         "history=WiLoR, target=GT 로 나눈다")
    ap.add_argument("--windows-file", default=None)
    ap.add_argument("--history-source", default="gt", choices=["gt", "corrupted_gt", "wilor"])
    ap.add_argument("--noise-stats", default=None,
                    help="이 JSON 의 측정값으로 history 를 열화시킨다 (조건 B)")
    ap.add_argument("--noise-override", nargs="*", default=[],
                    help="key=value 로 noise parameter 를 덮어쓴다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    history_file = args.history_frames or f"canonical_frames_{args.split}.parquet"
    frames = load_canonical_frames(out_dir / history_file)
    target_frames = (load_canonical_frames(out_dir / args.target_frames)
                     if args.target_frames else frames)
    windows = pl.read_parquet(out_dir / (args.windows_file or f"windows_{args.split}.parquet"))
    methods = {name: BASELINES[name] for name in args.methods}
    if "hold" not in methods:
        methods = {"hold": BASELINES["hold"], **methods}

    model_versions, seeds = {}, {}
    for raw in args.checkpoint:
        path = Path(raw)
        name = f"{path.parent.parent.name}/{path.parent.name}"
        predict, payload = torch_method(path, args.device)
        methods[name] = predict
        model_versions[name] = f"{name}@{payload['config_hash']}"
        seeds[name] = payload["seed"]

    print(f"config  : {cfg.path} (hash {cfg.hash})")
    print(f"split   : {args.split}  windows={windows.height}  methods={list(methods)}")
    noise = None
    if args.noise_stats:
        import json

        from ..data.augmentations import NoiseConfig

        overrides = {}
        for item in args.noise_override:
            key, _, value = item.partition("=")
            overrides[key] = (tuple(value.split(",")) if key == "enabled"
                              else float(value))
        noise = NoiseConfig.from_measurement(
            json.loads(Path(args.noise_stats).read_text()), **overrides)
        print(f"noise   : {noise}")

    values, motion = _predict_all(frames, windows, methods,
                                  target_frames if args.target_frames else None, noise)
    table = _long_table(windows, values, motion, args.split, cfg.hash, model_versions, seeds,
                        args.history_source)

    path = out_dir / f"{args.tag}_metrics_{args.split}.parquet"
    table.write_parquet(path)
    print(f"metrics : {table.height} rows -> {path}\n")

    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print("absolute MPJPE (mm), sequence 평균, 전체")
        print(summarize(table))
        for tercile in ("low", "high"):
            print(f"\nabsolute MPJPE (mm), wrist speed {tercile} tercile")
            print(summarize(table, group_key="wrist_speed_tercile", group_value=tercile))


if __name__ == "__main__":
    main()
