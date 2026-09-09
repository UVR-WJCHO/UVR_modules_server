"""모델 x EMA 조합을 연속 시퀀스에서 비교한다.

학습의 시간적 일관성 항과 서버 후처리 EMA 는 같은 문제를 다른 지점에서 다룬다.
둘을 겹쳐 쓰는 것이 이득인지, 하나만 쓰는 것이 나은지 이 표로 판단한다.

  python -m modules.delay_nowcasting.evaluation.smoothing_sweep \
      --models mixed_fps_clip mixed_tmp_s10 --alphas 1.0 0.8 0.6 0.4
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import torch

from ..config import REPO_ROOT
from ..data.build_windows import gather, load_canonical_frames
from ..training import checkpoint as ckpt


def runs_of(w: pl.DataFrame, min_len: int, limit: int):
    tid = w["track_id"].to_numpy()
    ar = w["anchor_row"].to_numpy()
    brk = np.r_[0, np.flatnonzero((tid[1:] != tid[:-1]) | (ar[1:] != ar[:-1] + 1)) + 1, len(ar)]
    return [(a, b) for a, b in zip(brk[:-1], brk[1:]) if b - a >= min_len][:limit]


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return series
    out = series.copy()
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="모델 x EMA 조합 비교")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4])
    ap.add_argument("--dataset", default="dexycb_v1")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--horizon", type=float, default=199.0)
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    out = REPO_ROOT / "research_data"
    O = out / args.dataset
    hist = load_canonical_frames(O / "wilor_frames_val.parquet")
    tgt = load_canonical_frames(O / "canonical_frames_val.parquet")
    w = (pl.read_parquet(O / f"windows_fps{args.fps}_val.parquet")
         .filter(pl.col("requested_horizon_ms") == args.horizon)
         .sort(["track_id", "anchor_row"]))
    segments = runs_of(w, 25, args.runs)
    print(f"{args.dataset} @ {args.fps} fps, horizon {int(args.horizon)} ms, "
          f"연속 구간 {len(segments)} 개 / {sum(b - a for a, b in segments):,} 프레임\n")

    gt_motion = []
    print(f"{'모델':>18} {'EMA':>5} {'정확도(mm)':>11} {'떨림(mm)':>10} {'추종성':>8}")
    for name in args.models:
        paths = sorted((out / "mixed_v1" / name).glob("seed*/best_model.pt"))
        if not paths:
            print(f"{name:>18}   checkpoint 없음")
            continue
        model = ckpt.load(paths[0], "cuda")[0].eval()
        acc = {a: [] for a in args.alphas}
        jit = {a: [] for a in args.alphas}
        fol = {a: [] for a in args.alphas}
        for lo, hi in segments:
            sub = w[list(range(lo, hi))]
            b = gather(hist, sub, tgt)
            hd = (sub["handedness"].to_numpy() == "RIGHT").astype(np.int64)
            with torch.no_grad():
                p = model(
                    torch.as_tensor(b["history_joints"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(b["history_time_ms"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(b["horizon_ms"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(hd, device="cuda"),
                    torch.ones(b["history_joints"].shape[:3], device="cuda"),
                ).double().cpu().numpy()
            g = b["target_joints"]
            dg = np.diff(g, axis=0)
            gt_motion.append(np.linalg.norm(dg, axis=-1).mean() * 1000)
            den = np.einsum("tjk,tjk->tj", dg, dg)
            ok = den > 1e-10
            for a in args.alphas:
                s = ema(p, a)
                ds = np.diff(s, axis=0)
                acc[a].append(np.linalg.norm(s - g, axis=-1).mean() * 1000)
                jit[a].append(np.linalg.norm(ds, axis=-1).mean() * 1000)
                num = np.einsum("tjk,tjk->tj", ds, dg)
                fol[a].append((num[ok] / den[ok]).mean())
        for a in args.alphas:
            print(f"{name:>18} {a:>5.1f} {np.mean(acc[a]):>11.2f} "
                  f"{np.mean(jit[a]):>10.2f} {np.mean(fol[a]):>8.2f}")
    print(f"{'GT (하한)':>18} {'':>5} {0.0:>11.2f} {np.mean(gt_motion):>10.2f} {1.00:>8.2f}")


if __name__ == "__main__":
    main()
