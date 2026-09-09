"""2.5D checkpoint 를 metric space 에서 평가한다 — 논문 표를 만드는 절차.

  python -m modules.delay_nowcasting.evaluation.evaluate_25d \
      --dataset dexycb_v1 --split val \
      --full dexycb_25d --variant nohorizon:dexycb_25d_nohorizon \
      --variant fixedgain:dexycb_25d_fixedgain

절차를 여기 고정해 둔다. 임시 스크립트로 표를 만들면 나중에 같은 수를 다시 낼 수 없다.

  1. history = 배포 추정기 출력(wilor25d), target = GT(canonical25d). 배포 조건 그대로다.
  2. seed 3 으로 섞어 앞 120,000 window 만 쓴다. 모든 구성이 **같은 window 집합**을 보므로
     비교가 짝지어진다.
  3. 잡음도 증강도 넣지 않는다. 평가는 깨끗한 입력에서 한다.
  4. 예측한 2.5D 를 **GT 손목 depth** 로 lift 해 카메라 좌표 mm 로 잰다. 기기가 공급할
     depth 의 오차를 빼고 예측만 고립시키기 위해서다.
  5. seed 별로 오차를 낸 뒤 **seed 간 평균**을 낸다. 예측을 seed 끼리 평균하는 ensemble 이
     아니다 — 그렇게 하면 실제보다 좋게 나온다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..config import REPO_ROOT
from ..data.build_25d import from_25d
from ..data.canonical_schema import FINGERTIPS, NUM_JOINTS, WRIST
from ..methods.baselines import BASELINES
from ..training import checkpoint as ckpt
from ..training.dataset import load_split

SUBSAMPLE = 120_000
SUBSAMPLE_SEED = 3
DEFAULT_HORIZONS = (33, 100, 166, 233, 300)


def prepare(root: Path, split: str, device: str = "cuda",
            windows: str = "windows_wilor") -> dict:
    """평가용 batch 를 한 번만 만든다. 모든 구성이 이것을 공유한다.

    windows 는 window 표의 tag 다. horizon 0 을 포함한 표는 따로 만들어 두었다 —
    학습이 쓰는 windows_wilor_train 을 건드리지 않기 위해서다.
    """
    data = load_split(root, split, device,
                      history_file=f"wilor25d_frames_{split}.parquet",
                      target_file=f"canonical25d_frames_{split}.parquet",
                      windows_file=f"{windows}_{split}.parquet")
    generator = torch.Generator(device=device).manual_seed(SUBSAMPLE_SEED)
    selected = torch.randperm(len(data), device=device,
                              generator=generator)[:SUBSAMPLE]
    batch = data.batch(selected, None, generator)

    frames = pl.read_parquet(root / f"canonical_frames_{split}.parquet",
                             columns=["joints_camera"])
    camera = frames["joints_camera"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    wrist_depth = camera[data.target_row[selected].cpu().numpy()][:, WRIST, 2]

    target = batch["target"].cpu().numpy().astype(np.float64)
    history = batch["history"].cpu().numpy().astype(np.float64)
    return {
        "batch": batch,
        "wrist_depth": wrist_depth,
        "gt": from_25d(target, wrist_depth),
        "requested": data.requested_horizon_ms[selected.cpu().numpy()],
        "keep": (np.isfinite(target).all((1, 2)) & np.isfinite(history).all((1, 2, 3))
                 & np.isfinite(wrist_depth) & (wrist_depth > 0.05)),
    }


@torch.no_grad()
def evaluate(prepared: dict, run_dir: Path, horizons, device: str = "cuda") -> np.ndarray:
    """(horizon,) mm. seed 별로 재고 seed 간 평균을 낸다."""
    batch, per_seed = prepared["batch"], []
    for path in sorted(run_dir.glob("seed*/best_model.pt")):
        model = ckpt.load(path, device)[0].eval()
        prediction = model(batch["history"], batch["history_time_ms"], batch["horizon_ms"],
                           batch["handedness"], batch["visibility"]).double().cpu().numpy()
        error = np.linalg.norm(from_25d(prediction, prepared["wrist_depth"]) - prepared["gt"],
                               axis=-1).mean(-1) * 1000.0
        per_seed.append([error[prepared["keep"] & (prepared["requested"] == h)].mean()
                         for h in horizons])
    if not per_seed:
        raise SystemExit(f"checkpoint 가 없다: {run_dir}")
    return np.mean(per_seed, axis=0)


def learned_gain(run_dir: Path) -> dict[str, float] | None:
    """관절 묶음별 base_gain 평균. 공통 scalar 로 학습한 구성이면 None."""
    gains = [ckpt.load(p, "cpu")[0].base_gain.detach().numpy().ravel()
             for p in sorted(run_dir.glob("seed*/best_model.pt"))]
    if not gains or gains[0].size != NUM_JOINTS:
        return None
    mean = np.mean(gains, axis=0)
    tips, mcp = list(FINGERTIPS), [1, 5, 9, 13, 17]
    middle = [j for j in range(NUM_JOINTS) if j not in tips + mcp + [WRIST]]
    return {"wrist": float(mean[WRIST]), "mcp": float(mean[mcp].mean()),
            "middle": float(mean[middle].mean()), "tips": float(mean[tips].mean())}


@torch.no_grad()
def evaluate_baseline(prepared: dict, name: str, horizons) -> np.ndarray:
    """비학습 참조를 **같은 window·같은 지표**로 잰다. 모델과 표본이 다르면 비교가 안 된다."""
    batch = prepared["batch"]
    prediction = BASELINES[name](batch["history"].cpu().numpy().astype(np.float64),
                                 batch["history_time_ms"].cpu().numpy().astype(np.float64),
                                 batch["horizon_ms"].cpu().numpy().astype(np.float64))
    error = np.linalg.norm(from_25d(prediction, prepared["wrist_depth"]) - prepared["gt"],
                           axis=-1).mean(-1) * 1000.0
    return np.array([error[prepared["keep"] & (prepared["requested"] == h)].mean()
                     for h in horizons])


def resolve(spec: str, dataset: str) -> Path:
    """'name' 은 같은 데이터셋 안, 'other_v1/name' 은 다른 데이터셋의 run 이다.

    교차 평가(다른 도메인에서 학습한 모델을 이 도메인 데이터로 재기)에 쓴다.
    """
    root = REPO_ROOT / "research_data"
    return root / spec if "/" in spec else root / dataset / spec


def main() -> None:
    ap = argparse.ArgumentParser(description="2.5D checkpoint 를 metric space 에서 평가")
    ap.add_argument("--dataset", required=True, help="outputs 아래 데이터셋 디렉터리")
    ap.add_argument("--split", default="val")
    ap.add_argument("--full", required=True, help="기준이 되는 run 디렉터리 이름")
    ap.add_argument("--variant", action="append", default=[],
                    help="'라벨:run_dir'. 여러 번 줄 수 있다")
    ap.add_argument("--baseline", action="append", default=[],
                    help="'라벨:이름'. methods.baselines 의 키 (hold, cv_robust, kalman_cv ...)")
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=list(DEFAULT_HORIZONS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows", default="windows_wilor",
                    help="window 표 tag. horizon 0 이 필요하면 windows_h0")
    ap.add_argument("--out", default=None, help="JSON 으로도 남긴다")
    args = ap.parse_args()

    root = REPO_ROOT / "research_data" / args.dataset
    prepared = prepare(root, args.split, args.device, args.windows)
    horizons = [float(h) for h in args.horizons_ms]
    print(f"{args.dataset} / {args.split}   "
          f"{int(prepared['keep'].sum()):,} window (부분표본 {SUBSAMPLE:,})")
    header = "".join(f"{int(h):>9}" for h in horizons)
    print(f"{'':>26}{header}")
    results = {}

    for spec in args.baseline:
        label, _, name = spec.partition(":")
        values = evaluate_baseline(prepared, name or label, horizons)
        print(f"{label:>26}" + "".join(f"{v:>9.2f}" for v in values))
        results.setdefault("baselines", {})[label] = values.tolist()

    full = evaluate(prepared, resolve(args.full, args.dataset), horizons, args.device)
    print(f"{'full':>26}" + "".join(f"{v:>9.2f}" for v in full))
    results["full"] = full.tolist()

    for spec in args.variant:
        label, _, name = spec.partition(":")
        values = evaluate(prepared, resolve(name or label, args.dataset),
                          horizons, args.device)
        penalty = (values / full - 1.0) * 100.0
        print(f"{label:>26}" + "".join(f"{v:>9.2f}" for v in values))
        print(f"{'penalty':>26}" + "".join(f"{v:>8.1f}%" for v in penalty))
        results[label] = {"mm": values.tolist(), "penalty_pct": penalty.tolist()}

    gain = learned_gain(resolve(args.full, args.dataset))
    if gain:
        print("\n학습된 base_gain (seed 평균)  "
              + "  ".join(f"{k} {v:.3f}" for k, v in gain.items()))
        results["base_gain"] = gain

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"dataset": args.dataset, "split": args.split, "horizons_ms": horizons,
             "subsample": SUBSAMPLE, "results": results}, indent=2, ensure_ascii=False))
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
