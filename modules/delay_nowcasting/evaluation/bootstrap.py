"""Sequence bootstrap 을 이용한 paired 비교 (B 계획 §9.5).

frame 을 독립 표본으로 보면 CI 가 터무니없이 좁아진다. 한 sequence 안의 frame 은
강하게 상관되어 있으므로, **sequence 를 표본 단위**로 리샘플링한다. 두 방법은 같은
sequence 에서 짝지어 비교한다.
"""
from __future__ import annotations

import numpy as np
import polars as pl

DEFAULT_ITERATIONS = 10_000


def paired_units(table: pl.DataFrame, method_a: str, method_b: str, metric: str,
                 horizon_ms: float, group_key: str = "all",
                 group_value: str = "all") -> pl.DataFrame:
    """(sequence, hand) 단위로 두 방법을 나란히 놓는다."""
    keys = ["sequence_id", "subject_id", "handedness"]
    selected = table.filter(
        (pl.col("metric") == metric) & (pl.col("horizon_ms") == horizon_ms)
        & (pl.col("group_key") == group_key) & (pl.col("group_value") == group_value)
        & pl.col("method").is_in([method_a, method_b])
    )
    wide = selected.pivot(on="method", index=keys + ["n_frames"], values="value")
    if method_a not in wide.columns or method_b not in wide.columns:
        raise ValueError(f"{method_a} 또는 {method_b} 의 결과가 표에 없다")
    return wide.drop_nulls([method_a, method_b])


def paired_bootstrap(table: pl.DataFrame, method: str, reference: str, metric: str,
                     horizon_ms: float, group_key: str = "all", group_value: str = "all",
                     iterations: int = DEFAULT_ITERATIONS, seed: int = 0) -> dict:
    """method 가 reference 대비 얼마나 개선했는지와 그 95% CI.

    개선률은 (reference - method) / reference 로, 양수면 method 가 낫다.
    리샘플링 단위는 sequence 이며, 같은 sequence 의 두 값이 함께 뽑힌다(paired).
    """
    units = paired_units(table, method, reference, metric, horizon_ms, group_key, group_value)
    a = units[method].to_numpy()
    b = units[reference].to_numpy()
    n = len(a)
    if n < 2:
        raise ValueError(f"표본이 {n}개뿐이라 bootstrap 할 수 없다")

    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(iterations, n))
    improvement = (b[index].mean(axis=1) - a[index].mean(axis=1)) / b[index].mean(axis=1)
    low, high = np.quantile(improvement, [0.025, 0.975])

    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "horizon_ms": horizon_ms,
        "group_key": group_key,
        "group_value": group_value,
        "n_units": n,
        "method_mean": float(a.mean()),
        "reference_mean": float(b.mean()),
        "improvement": float((b.mean() - a.mean()) / b.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "improves_in_ci": bool(low > 0),
        "win_rate": float((a < b).mean()),
    }


def best_reference(table: pl.DataFrame, candidates: list[str], metric: str,
                   horizon_ms: float) -> str:
    """주어진 horizon 에서 sequence 평균이 가장 낮은 방법을 고른다.

    Phase 1 에서 확인했듯 어느 CV 가 최강인지는 history 조건에 따라 달라지므로,
    "best non-learned baseline" 은 매번 다시 고른다 (§6.2).
    """
    scores = (table
              .filter((pl.col("metric") == metric) & (pl.col("horizon_ms") == horizon_ms)
                      & (pl.col("group_key") == "all") & pl.col("method").is_in(candidates))
              .group_by("method").agg(pl.col("value").mean().alias("score"))
              .sort("score"))
    return scores["method"][0]
