"""정확도·안정성·추종성을 함께 재서 시간적 일관성 항의 거래를 본다.

selection 지표는 정확도만 보므로, 예측이 굳어 지연이 재발해도 드러나지 않는다.
세 축을 함께 재야 판단할 수 있다.

  정확도   GT 대비 MPJPE
  안정성   연속한 두 anchor 예측의 프레임간 변화량. GT 자신의 변화량이 하한이다
  추종성   예측 변화량 중 GT 변화 방향으로 간 성분의 비율. 1 에 가까울수록 잘 따라간다.
           예측이 굳으면 이 값이 0 으로 간다

  python -m modules.delay_nowcasting.evaluation.temporal_tradeoff \
      --models mixed_fps_clip mixed_tmp_s02 mixed_tmp_t02 --fps 15 --horizon 199
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..config import REPO_ROOT
from ..data.build_windows import gather, load_canonical_frames
from ..training import checkpoint as ckpt


def measure(model, b0, b1, hd0, hd1) -> dict:
    def run(b, hd):
        with torch.no_grad():
            return model(
                torch.as_tensor(b["history_joints"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(b["history_time_ms"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(b["horizon_ms"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(hd, device="cuda"),
                torch.ones(b["history_joints"].shape[:3], device="cuda")
            ).double().cpu().numpy()

    p0, p1 = run(b0, hd0), run(b1, hd1)
    g0, g1 = b0["target_joints"], b1["target_joints"]
    dp, dg = p1 - p0, g1 - g0
    # 추종성: 예측 변화를 GT 변화 방향에 투영한 비율
    num = np.einsum("bjk,bjk->bj", dp, dg)
    den = np.einsum("bjk,bjk->bj", dg, dg)
    keep = den > 1e-10
    return {
        "accuracy": float((np.linalg.norm(p0 - g0, axis=-1).mean()
                           + np.linalg.norm(p1 - g1, axis=-1).mean()) / 2 * 1000),
        "jitter": float(np.linalg.norm(dp, axis=-1).mean() * 1000),
        "follow": float((num[keep] / den[keep]).mean()),
        "gt_motion": float(np.linalg.norm(dg, axis=-1).mean() * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="시간적 일관성 거래 평가")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="dexycb_v1")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--horizon", type=float, default=199.0)
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()

    out = REPO_ROOT / "research_data"
    O = out / args.dataset
    hist = load_canonical_frames(O / "wilor_frames_val.parquet")
    tgt = load_canonical_frames(O / "canonical_frames_val.parquet")
    w = pl.read_parquet(O / f"windows_fps{args.fps}_val.parquet").filter(
        pl.col("requested_horizon_ms") == args.horizon)
    anchor = w["anchor_row"].to_numpy()
    pair = np.flatnonzero(np.r_[False, anchor[1:] == anchor[:-1] + 1])[:args.limit]

    def pack(idx):
        sub = w[idx.tolist()]
        return (gather(hist, sub, tgt),
                (sub["handedness"].to_numpy() == "RIGHT").astype(np.int64))

    b0, hd0 = pack(pair - 1)
    b1, hd1 = pack(pair)

    print(f"{args.dataset} val @ {args.fps} fps, horizon {int(args.horizon)} ms, "
          f"{len(pair):,} 쌍")
    print(f"{'모델':>22} {'정확도(mm)':>11} {'떨림(mm)':>10} {'추종성':>8}")
    for name in args.models:
        seeds = sorted((out / "mixed_v1" / name).glob("seed*/best_model.pt"))
        if not seeds:
            print(f"{name:>22}   checkpoint 없음")
            continue
        rows = [measure(ckpt.load(p, "cuda")[0].eval(), b0, b1, hd0, hd1) for p in seeds]
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{name:>22} {m['accuracy']:>11.2f} {m['jitter']:>10.2f} {m['follow']:>8.2f}")
    print(f"{'GT (하한)':>22} {0.0:>11.2f} {m['gt_motion']:>10.2f} {1.00:>8.2f}")


if __name__ == "__main__":
    main()
