"""Teaser inset 용 실측 3-pose 추출 — 2.5D 판 (stale anchor / GT / forecast).

export_inset_pose.py 의 2.5D 대응이다. 좌표가 이미 카메라 정규화 image 좌표
(u_n, v_n, rel_z) 라 world->camera 변환이 없고, inset 이 그리는 (u_n, v_n) 이 곧
화면 평면이다. 오차는 논문과 같은 방식으로 GT 손목 depth 로 lift 해 mm 로 잰다.

대표성이 중요하다. 보기 좋은 프레임을 고르면 seed 를 고르는 것과 같은 잘못이 된다.

  python -m modules.delay_nowcasting.evaluation.export_inset_pose_25d \
      --config modules/delay_nowcasting/configs/data/dexycb_v1.yaml \
      --split test --horizon-ms 233 \
      --checkpoint research_data/dexycb_v1/dexycb_25d/seed{0,1,2}/best_model.pt \
      --out research/paper/figures/inset.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.build_25d import from_25d
from ..data.build_windows import gather, load_canonical_frames
from ..data.canonical_schema import NUM_JOINTS, WRIST
from .evaluate import torch_method
from .metrics import MM_PER_M

TIP_MIDDLE = 12


def main() -> None:
    ap = argparse.ArgumentParser(description="teaser inset 용 실측 자세 추출 (2.5D)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizon-ms", type=float, default=233.0)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = REPO_ROOT / cfg["output_root"] / cfg.name
    history = load_canonical_frames(out_dir / f"wilor25d_frames_{args.split}.parquet")
    targets = load_canonical_frames(out_dir / f"canonical25d_frames_{args.split}.parquet")
    windows = pl.read_parquet(out_dir / f"windows_wilor_{args.split}.parquet").filter(
        pl.col("requested_horizon_ms") == args.horizon_ms)
    print(f"windows @ {args.horizon_ms:.0f} ms: {windows.height}")

    batch = gather(history, windows, targets)
    gt = batch["target_joints"].astype(np.float64)          # 2.5D
    stale = batch["history_joints"][:, -1].astype(np.float64)

    # mm 오차는 논문과 같은 방식으로 잰다: GT 손목 depth 로 lift 한 camera frame 거리
    rows = np.asarray(windows["target_row"].to_numpy(), dtype=np.int64)
    cam = load_canonical_frames(out_dir / f"canonical_frames_{args.split}.parquet")
    wrist_depth = (cam["joints_camera"].to_numpy().reshape(-1, NUM_JOINTS, 3)
                   .astype(np.float64)[rows][:, WRIST, 2])

    # seed 를 고르지 않는다: 전체 오차가 3-seed 평균에 가장 가까운 seed 를 쓴다
    errors, preds = [], []
    for path in args.checkpoint:
        predict, _ = torch_method(Path(path), args.device)
        p = np.concatenate([
            predict(batch["history_joints"][i:i + 50_000].astype(np.float64),
                    batch["history_time_ms"][i:i + 50_000],
                    batch["horizon_ms"][i:i + 50_000],
                    (windows["handedness"].to_numpy()[i:i + 50_000] == "RIGHT").astype("int64"),
                    batch["history_visibility"][i:i + 50_000].astype(np.float32))
            for i in range(0, len(gt), 50_000)])
        preds.append(p)
        errors.append(np.linalg.norm(from_25d(p, wrist_depth) - from_25d(gt, wrist_depth),
                                     axis=-1).mean())
    mean_of_seeds = float(np.mean(errors))
    pick = int(np.argmin([abs(e - mean_of_seeds) for e in errors]))
    pred = preds[pick]
    print(f"seed 별 MPJPE(mm): {[round(e * MM_PER_M, 2) for e in errors]}  "
          f"-> 평균에 가장 가까운 {Path(args.checkpoint[pick]).parent.name} 사용")

    lift = lambda a: from_25d(a, wrist_depth)
    cam_gt, cam_stale, cam_pred = lift(gt), lift(stale), lift(pred)
    err_stale = np.linalg.norm(cam_stale - cam_gt, axis=-1).mean(axis=-1)
    err_pred = np.linalg.norm(cam_pred - cam_gt, axis=-1).mean(axis=-1)

    # inset 은 (u_n, v_n) 평면에 그린다. rel_z 를 버리는 것이 예측을 실제보다 좋아
    # 보이게 만들면 안 되므로, 평면 비율이 3D 비율과 맞는 프레임만 고른다.
    flat = lambda a, b: np.linalg.norm((a - b)[..., :2], axis=-1).mean(axis=-1)
    ratio_2d = flat(stale, gt) / np.maximum(flat(pred, gt), 1e-9)
    ratio_3d = err_stale / np.maximum(err_pred, 1e-9)
    span = np.linalg.norm((gt[:, TIP_MIDDLE] - gt[:, WRIST])[:, :2], axis=-1)

    rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
    typical = (np.abs(rank(err_stale) - 0.5) < 0.06) & (np.abs(rank(err_pred) - 0.5) < 0.06)
    faithful = np.abs(np.log(ratio_2d / ratio_3d)) < 0.10
    ok = typical & faithful
    if not ok.any():
        raise SystemExit("조건을 만족하는 프레임이 없다. 임계를 완화하라")
    idx = int(np.flatnonzero(ok)[np.argmax(span[ok])])
    print(f"후보 {int(typical.sum())} (전형) -> {int(ok.sum())} (평면 충실) "
          f"-> 손이 가장 크게 보이는 것 선택")

    poses = {name: value[idx].reshape(NUM_JOINTS, 3)
             for name, value in (("gt", gt), ("stale", stale), ("pred", pred))}
    np.savez(args.out, **poses)
    meta = {
        "dataset": cfg["dataset"].get("source_dataset", cfg.name),
        "split": args.split, "horizon_ms": args.horizon_ms,
        "representation": "2.5D (u_n, v_n, rel_z); inset uses the first two channels",
        "sequence_id": str(windows["sequence_id"][idx]),
        "handedness": str(windows["handedness"][idx]),
        "checkpoint": args.checkpoint[pick],
        "stale_mpjpe_mm": round(float(err_stale[idx]) * MM_PER_M, 2),
        "forecast_mpjpe_mm": round(float(err_pred[idx]) * MM_PER_M, 2),
        "split_median_stale_mm": round(float(np.median(err_stale)) * MM_PER_M, 2),
        "split_median_forecast_mm": round(float(np.median(err_pred)) * MM_PER_M, 2),
        "ratio_3d": round(float(ratio_3d[idx]), 2),
        "ratio_projected": round(float(ratio_2d[idx]), 2),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
