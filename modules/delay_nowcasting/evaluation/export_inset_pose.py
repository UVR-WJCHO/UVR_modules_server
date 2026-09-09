"""Teaser inset 용 실측 3-pose 추출 (stale anchor / GT / forecast).

teaser 의 inset 은 "낡은 자세가 얼마나 어긋나 있고 예측이 그 간격을 얼마나 닫는가" 를
보여준다. 그 크기가 곧 정량적 주장이므로 **실측 좌표만** 쓴다.

대표성이 중요하다. 보기 좋은 프레임을 고르면 seed 를 고르는 것과 같은 종류의 잘못이
된다. 그래서 hold 오차와 모델 오차를 함께 표준화한 뒤 **둘 다 중앙값에 가장 가까운**
표본을 고른다.

  python -m modules.delay_nowcasting.evaluation.export_inset_pose \
      --config modules/delay_nowcasting/configs/data/dexycb_v1.yaml \
      --split test --horizon-ms 133 --out /tmp/inset.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.build_windows import gather, load_canonical_frames
from ..data.canonical_schema import NUM_JOINTS, invert_transform, transform_points
from .evaluate import torch_method
from .metrics import MM_PER_M


def main() -> None:
    ap = argparse.ArgumentParser(description="teaser inset 용 실측 자세 추출")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizon-ms", type=float, default=133.0)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = REPO_ROOT / cfg["output_root"] / cfg.name
    history = load_canonical_frames(out_dir / f"wilor_frames_{args.split}.parquet")
    targets = load_canonical_frames(out_dir / f"canonical_frames_{args.split}.parquet")
    windows = pl.read_parquet(out_dir / f"windows_wilor_{args.split}.parquet").filter(
        pl.col("requested_horizon_ms") == args.horizon_ms)
    print(f"windows @ {args.horizon_ms:.0f} ms: {windows.height}")

    batch = gather(history, windows, targets)
    gt = batch["target_joints"].astype(np.float64)
    stale = batch["history_joints"][:, -1].astype(np.float64)      # hold = anchor

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
        errors.append(np.linalg.norm(p - gt, axis=-1).mean())
    mean_of_seeds = float(np.mean(errors))
    pick = int(np.argmin([abs(e - mean_of_seeds) for e in errors]))
    pred = preds[pick]
    print(f"seed 별 MPJPE(mm): {[round(e * MM_PER_M, 2) for e in errors]}  "
          f"-> 평균에 가장 가까운 {Path(args.checkpoint[pick]).parent.name} 사용")

    err_stale = np.linalg.norm(stale - gt, axis=-1).mean(axis=-1)
    err_pred = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)

    # inset 은 2D 로 그린다. 모델 잔차는 대부분 depth 방향이라, 깊이를 지우는 투영은
    # 예측을 실제보다 좋아 보이게 만든다. 그래서 세 조건을 모두 만족하는 프레임만 고른다.
    #   (1) 두 오차 모두 이 split 의 중앙값 근처   -> 전형적인 프레임
    #   (2) 투영 후 오차 비율이 3D 비율과 일치     -> 그림이 수치를 왜곡하지 않음
    #   (3) 화면 안에서 손이 크게 펼쳐져 보임      -> 스켈레톤을 알아볼 수 있음
    rows = np.asarray(windows["target_row"].to_numpy(), dtype=np.int64)
    T = np.linalg.inv(targets["camera_to_world"].to_numpy()[rows]
                      .reshape(-1, 4, 4).astype(np.float64))
    to_cam = lambda w: np.einsum("nij,nkj->nki", T[:, :3, :3], w) + T[:, None, :3, 3]
    cam_gt, cam_stale, cam_pred = to_cam(gt), to_cam(stale), to_cam(pred)
    flat = lambda a, b: np.linalg.norm((a - b)[..., :2], axis=-1).mean(axis=-1)
    ratio_2d = flat(cam_stale, cam_gt) / np.maximum(flat(cam_pred, cam_gt), 1e-9)
    ratio_3d = err_stale / np.maximum(err_pred, 1e-9)
    span = np.linalg.norm((cam_gt[:, 12] - cam_gt[:, 0])[:, :2], axis=-1)

    rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
    typical = (np.abs(rank(err_stale) - 0.5) < 0.06) & (np.abs(rank(err_pred) - 0.5) < 0.06)
    faithful = np.abs(np.log(ratio_2d / ratio_3d)) < 0.10          # 비율 오차 10% 이내
    ok = typical & faithful
    if not ok.any():
        raise SystemExit("조건을 만족하는 프레임이 없다. 임계를 완화하라")
    idx = int(np.flatnonzero(ok)[np.argmax(span[ok])])
    print(f"후보 {int(typical.sum())} (전형) -> {int(ok.sum())} (투영 충실) "
          f"-> 손이 가장 크게 보이는 것 선택")

    row = int(rows[idx])
    world_to_cam = invert_transform(
        targets["camera_to_world"].to_numpy()[row].reshape(4, 4).astype(np.float64))
    poses = {name: transform_points(world_to_cam, value[idx]).reshape(NUM_JOINTS, 3)
             for name, value in (("gt", gt), ("stale", stale), ("pred", pred))}

    np.savez(args.out, **poses)
    meta = {
        "dataset": cfg["dataset"].get("source_dataset", cfg.name),
        "split": args.split, "horizon_ms": args.horizon_ms,
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
