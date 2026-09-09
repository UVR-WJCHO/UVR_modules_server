"""결과 표 요약과 gate 판정 (B 계획 §9.5, §11).

  python -m modules.delay_nowcasting.evaluation.summarize \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml \
      --split val --tag phase2 --method residual_mlp/seed0 --gate 2

Gate 판정은 반드시 sequence 단위 paired 비교로 한다. 비교 대상인 "best non-learned
baseline" 은 horizon 마다 다시 고른다 (Phase 1: 어느 CV 가 최강인지는 조건에 따라 다르다).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ..config import REPO_ROOT, load_config
from ..methods.baselines import BASELINES
from .bootstrap import best_reference, paired_bootstrap

GATE_THRESHOLDS = {2: 0.02, 3: 0.02}       # §11 Phase 2 Gate 2 는 평균 2% 개선
NON_LEARNED = list(BASELINES)


def gate_report(table: pl.DataFrame, method: str, horizons: list[float],
                metric: str = "abs_mpjpe_mm", iterations: int = 10_000) -> list[dict]:
    rows = []
    for horizon in horizons:
        reference = best_reference(table, NON_LEARNED, metric, horizon)
        rows.append(paired_bootstrap(table, method, reference, metric, horizon,
                                     iterations=iterations))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="결과 요약과 gate 판정")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--method", required=True, help="판정 대상 method 이름")
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=[66.0, 100.0])
    ap.add_argument("--gate", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    table = pl.read_parquet(out_dir / f"{args.tag}_metrics_{args.split}.parquet")

    rows = gate_report(table, args.method, args.horizons_ms)
    print(f"split {args.split}, metric absolute MPJPE, sequence-hand 단위 paired bootstrap\n")
    header = (f"{'horizon':>8} {'best baseline':>16} {'baseline':>10} {'model':>10} "
              f"{'개선':>8} {'95% CI':>20} {'win rate':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{row['ci_low'] * 100:+.1f}%, {row['ci_high'] * 100:+.1f}%]"
        print(f"{row['horizon_ms']:>7.0f}ms {row['reference']:>16} "
              f"{row['reference_mean']:>9.2f}mm {row['method_mean']:>9.2f}mm "
              f"{row['improvement'] * 100:>7.1f}% {ci:>20} {row['win_rate'] * 100:>8.1f}%")

    verdict = None
    if args.gate is not None:
        threshold = GATE_THRESHOLDS[args.gate]
        mean_improvement = sum(r["improvement"] for r in rows) / len(rows)
        all_positive_ci = all(r["improves_in_ci"] for r in rows)
        passed = mean_improvement >= threshold
        verdict = {"gate": args.gate, "threshold": threshold,
                   "mean_improvement": mean_improvement,
                   "all_ci_positive": all_positive_ci, "passed": passed}
        print(f"\nGate {args.gate}: 평균 개선 {mean_improvement * 100:.1f}% "
              f"(기준 {threshold * 100:.0f}%), 모든 horizon 의 CI 가 개선 방향: "
              f"{all_positive_ci} -> {'통과' if passed else '미통과'}")

    payload = {"split": args.split, "method": args.method, "rows": rows, "verdict": verdict}
    path = out_dir / f"{args.tag}_gate_{args.split}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nreport : {path}")


if __name__ == "__main__":
    main()
