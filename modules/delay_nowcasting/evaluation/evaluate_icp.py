"""Nowcaster 예측 -> Depth-ICP 정제 -> 최종 자세 비교 (DexYCB).

  python -m modules.delay_nowcasting.evaluation.evaluate_icp \
      --config modules/delay_nowcasting/configs/data/dexycb_v1.yaml --split val \
      --checkpoint <ckpt> --max-anchors 200

이 연구의 산출물은 예측 자체가 아니라 ICP 를 거친 최종 자세다. 그래서 각 방법의 예측을
**ICP seed** 로 넣고, 목표 시점 depth 로 정제한 뒤의 오차를 비교한다. ICP 는 global rigid
만 푼다(`staged=True`) — 손가락은 seed 가 준 그대로 남으므로 nowcaster 의 articulation
품질이 최종 결과에 그대로 반영된다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.build_windows import gather, load_canonical_frames
from ..data.canonical_schema import NUM_JOINTS, invert_transform, transform_points
from ..methods.baselines import BASELINES
from .depth_icp_eval import IcpRefiner, default_params
from .metrics import MM_PER_M


def _torch_predictor(path: Path, device: str):
    import torch

    from ..training import checkpoint as ckpt

    model, _ = ckpt.load(path, device)

    def predict(history, times, horizon, handedness):
        with torch.no_grad():
            return model(
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(times, dtype=torch.float32, device=device),
                torch.as_tensor(horizon, dtype=torch.float32, device=device),
                torch.as_tensor(handedness, dtype=torch.long, device=device),
                torch.ones(history.shape[:3], dtype=torch.float32, device=device),
            ).double().cpu().numpy()

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description="ICP 후 최종 자세 비교")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--history-frames", default="wilor_frames_val.parquet")
    ap.add_argument("--target-frames", default="canonical_frames_val.parquet")
    ap.add_argument("--windows-file", default="windows_wilor_val.parquet")
    ap.add_argument("--methods", nargs="*", default=["hold", "kalman_cv"])
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=[33, 66, 100, 133])
    ap.add_argument("--max-anchors", type=int, default=200,
                    help="시퀀스 전체를 다 돌면 느리다. anchor 를 균등 표본으로 제한한다")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="icp")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    dataset = cfg["dataset"]

    history = load_canonical_frames(out_dir / args.history_frames)
    target = load_canonical_frames(out_dir / args.target_frames)
    windows = pl.read_parquet(out_dir / args.windows_file).filter(
        pl.col("requested_horizon_ms").is_in(args.horizons_ms))

    models = {}
    for raw in args.checkpoint:
        path = Path(raw)
        models[f"{path.parent.parent.name}/{path.parent.name}"] = _torch_predictor(path, args.device)
    names = list(args.methods) + list(models)

    # anchor 를 균등 표본으로 줄인다(ICP 가 느리다). 같은 anchor 의 모든 horizon 을 함께 쓴다.
    anchors = np.unique(windows["anchor_row"].to_numpy())
    if len(anchors) > args.max_anchors:
        anchors = anchors[np.linspace(0, len(anchors) - 1, args.max_anchors).astype(int)]
    windows = windows.filter(pl.col("anchor_row").is_in(anchors.tolist()))
    print(f"anchor {len(anchors)}개, window {windows.height}개, 방법 {names}")

    batch = gather(history, windows, target)
    hist = batch["history_joints"].astype(np.float64)
    times = batch["history_time_ms"]
    horizon = batch["horizon_ms"]
    hand_index = (windows["handedness"].to_numpy() == "RIGHT").astype(np.int64)

    predictions = {}
    for name in args.methods:
        predictions[name] = BASELINES[name](hist, times, horizon)
    for name, fn in models.items():
        predictions[name] = fn(hist, times, horizon, hand_index)

    frame_rows = target.with_row_index("row")
    anchor_frame = frame_rows["frame_idx"].to_numpy()[windows["anchor_row"].to_numpy()]
    target_frame = frame_rows["frame_idx"].to_numpy()[windows["target_row"].to_numpy()]
    camera_to_world = frame_rows["camera_to_world"].to_numpy()[
        windows["target_row"].to_numpy()].reshape(-1, 4, 4).astype(np.float64)
    world_to_camera = invert_transform(camera_to_world)
    gt_camera = np.einsum("nij,nkj->nki", world_to_camera[:, :3, :3],
                          batch["target_joints"].astype(np.float64)) + world_to_camera[:, None, :3, 3]
    # anchor 자세(= history 마지막 프레임). skeleton rest 로 쓴다.
    anchor_camera = np.einsum("nij,nkj->nki", world_to_camera[:, :3, :3],
                              hist[:, -1]) + world_to_camera[:, None, :3, 3]

    manifest = {s["sequence_id"]: s for s in
                json.loads((out_dir / "dataset_manifest.json").read_text())["splits"][args.split]["sequences"]}
    params = default_params()
    refiners: dict[str, IcpRefiner] = {}
    records = []
    started = time.time()
    sequence_ids = windows["sequence_id"].to_numpy()
    for i in range(windows.height):
        sequence_id = sequence_ids[i]
        info = manifest[sequence_id]
        if sequence_id not in refiners:
            subject, sequence = sequence_id.split("/")
            refiners[sequence_id] = IcpRefiner(
                Path(dataset["calibration_root"]), Path(dataset["root"]) / subject / sequence,
                params)
        refiner = refiners[sequence_id]

        row = {"sequence_id": sequence_id, "handedness": info["handedness"],
               "horizon_ms": float(windows["requested_horizon_ms"][i])}
        for name in names:
            seed_world = predictions[name][i]
            seed_camera = (world_to_camera[i, :3, :3] @ seed_world.T).T + world_to_camera[i, :3, 3]
            row[f"{name}__seed"] = float(
                np.linalg.norm(seed_camera - gt_camera[i], axis=-1).mean() * MM_PER_M)
            try:
                refined, ok = refiner.refine(info["camera"], int(anchor_frame[i]),
                                             int(target_frame[i]), seed_camera,
                                             anchor_camera[i])
                row[f"{name}__icp"] = float(
                    np.linalg.norm(refined - gt_camera[i], axis=-1).mean() * MM_PER_M)
                row[f"{name}__guard_ok"] = bool(ok)
            except Exception as exc:                      # ICP 실패는 seed 유지로 처리
                row[f"{name}__icp"] = row[f"{name}__seed"]
                row[f"{name}__guard_ok"] = False
                row.setdefault("errors", []).append(f"{name}: {exc}")
        records.append(row)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{windows.height}  ({rate:.1f} window/s)", flush=True)

    table = pl.DataFrame([{k: v for k, v in r.items() if k != "errors"} for r in records])
    path = out_dir / f"{args.tag}_metrics_{args.split}.parquet"
    table.write_parquet(path)

    print(f"\n{'method':<26}{'seed':>10}{'ICP 후':>10}{'개선':>9}{'guard ok':>10}")
    for name in names:
        seed = table[f"{name}__seed"].mean()
        icp = table[f"{name}__icp"].mean()
        ok = table[f"{name}__guard_ok"].mean()
        print(f"  {name:<24}{seed:10.2f}{icp:10.2f}{(seed - icp) / seed * 100:8.1f}%{ok * 100:9.0f}%")
    print(f"\n표: {path}")


if __name__ == "__main__":
    main()
