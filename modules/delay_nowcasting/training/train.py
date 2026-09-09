"""모델 학습 (B 계획 §7.7, §11 Phase 2).

  python -m modules.delay_nowcasting.training.train \
      --config modules/delay_nowcasting/configs/model/residual_mlp.yaml --seed 0

checkpoint 선택 기준은 validation 의 selection horizon 평균 absolute MPJPE 다.
test split 은 이 스크립트에서 절대 열지 않는다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import REPO_ROOT, load_config, run_metadata
from ..evaluation.metrics import MM_PER_M
from . import checkpoint as ckpt
from .dataset import WindowTensors, load_split
from .losses import nowcast_loss

EVAL_CHUNK = 200_000


def _eval_chunk(model) -> int:
    """관절 차원을 유지하는 모델(tcn_graph)은 중간 텐서가 tcn_lite 의 21 배라
    같은 chunk 로는 평가에서 메모리가 터진다. 모델이 값을 지정하면 그것을 쓴다."""
    return int(getattr(model, "eval_chunk", EVAL_CHUNK))


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_split(model: torch.nn.Module, data: WindowTensors, noise=None,
                   seed: int = 12345) -> dict[float, float]:
    """requested horizon 별 absolute MPJPE (mm).

    noise 를 주면 validation history 도 같은 방식으로 열화시킨다. 이때 seed 를 고정해
    epoch 마다 같은 잡음을 보게 해야 early stopping 이 잡음 운에 흔들리지 않는다.
    """
    model.eval()
    errors = np.empty(len(data), dtype=np.float64)
    generator = (torch.Generator(device=data.joints.device).manual_seed(seed)
                 if noise is not None else None)
    chunk = _eval_chunk(model)
    for start in range(0, len(data), chunk):
        index = torch.arange(start, min(start + chunk, len(data)), device=data.joints.device)
        batch = data.batch(index, noise, generator)
        prediction = model(batch["history"], batch["history_time_ms"], batch["horizon_ms"],
                           batch["handedness"], batch["visibility"])
        error = torch.linalg.norm(prediction - batch["target"], dim=-1).mean(dim=-1)
        errors[index.cpu().numpy()] = error.double().cpu().numpy() * MM_PER_M
    return {float(h): float(errors[data.requested_horizon_ms == h].mean())
            for h in np.unique(data.requested_horizon_ms)}


@torch.no_grad()
def _normalization_stats(model: torch.nn.Module, data: WindowTensors, n_samples: int,
                         generator: torch.Generator, noise=None, geom=None) -> None:
    """normalization 통계는 학습이 실제로 보는 분포(= 열화 후)에서 뽑아야 한다."""
    from ..methods.features import build_features

    index = torch.randint(len(data), (min(n_samples, len(data)),),
                          device=data.joints.device, generator=generator)
    batch = data.batch(index, noise, generator, geom)
    if hasattr(model, "per_frame_features"):
        features = model.per_frame_features(batch["history"], batch["history_time_ms"],
                                            batch["visibility"]).reshape(-1, model.feature_mean.numel())
    else:
        features = build_features(batch["history"], batch["history_time_ms"],
                                  batch["horizon_ms"], model.accel_n_fit)
    model.set_normalization(features.mean(0), features.std(0))


def train(cfg, data_cfg, seed: int, out_dir: Path, device: str = "cuda") -> dict:
    training = cfg["training"]
    _set_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    cache_dir = REPO_ROOT / data_cfg["output_root"] / data_cfg.name
    train_source = training.get("train_data", {})
    train_data = load_split(cache_dir, "train", device,
                            history_file=train_source.get("history_frames"),
                            target_file=train_source.get("target_frames"),
                            windows_file=train_source.get("windows_file"))
    if train_source:
        print(f"seed {seed}: train history = {train_source['history_frames']}", flush=True)
    # 검증은 배포 조건에서 한다. 학습 입력이 무엇이든, 쓸 수 있는지는 실제 WiLoR
    # history 에서 결정되므로 checkpoint 선택 기준도 거기에 맞춘다.
    validation = training.get("validation", {})
    val_data = load_split(cache_dir, "val", device,
                          history_file=validation.get("history_frames"),
                          target_file=validation.get("target_frames"),
                          windows_file=validation.get("windows_file"))
    val_uses_noise = not validation.get("history_frames")
    print(f"seed {seed}: validation = "
          f"{validation.get('history_frames', 'GT history (+ 학습과 같은 noise)')}", flush=True)

    noise = None
    if training.get("noise"):
        from ..data.augmentations import NoiseConfig

        spec = dict(training["noise"])
        if "enabled" in spec:
            spec["enabled"] = tuple(spec["enabled"])
        # noise_scale 로 실측 보정값 전체를 한 번에 줄이거나 키운다.
        # "잡음이 과한가" 를 조건 C 검증으로 판정하기 위한 손잡이다.
        scale = float(spec.pop("noise_scale", 1.0))
        for key in ("jitter_m", "drift_m", "bias_m", "wrist_translation_m"):
            if key in spec:
                spec[key] = spec[key] * scale
        noise = NoiseConfig(**spec)
        print(f"seed {seed}: noise {noise}", flush=True)

    geom = None
    if training.get("geometry"):
        from ..data.augmentations import GeomConfig

        geom = GeomConfig(**training["geometry"])
        print(f"seed {seed}: 기하 증강 {geom}", flush=True)

    model = ckpt.build_model(cfg["model"]).to(device)
    # normalization 통계는 학습이 실제로 보는 분포에서 뽑아야 하므로 증강을 켜고 뽑는다.
    _normalization_stats(model, train_data, training.get("normalization_samples", 200_000),
                         generator, noise, geom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    # 지금까지는 lr 고정이라 epoch 5~13 에서 정체한 뒤 early stop 이 걸렸다. 그것이
    # 데이터 한계인지 최적화 한계인지 가르기 위해 plateau 감쇠를 선택지로 둔다.
    schedule = training.get("lr_schedule")
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=schedule.get("factor", 0.3),
        patience=schedule.get("patience", 3), min_lr=schedule.get("min_lr", 1e-6))
        if schedule else None)

    batch_size = int(training["batch_size"])
    steps_per_epoch = len(train_data) // batch_size
    selection = [float(h) for h in training["selection_horizons_ms"]]
    weights = training.get("loss_weights")
    uses_temporal = any((weights or {}).get(k, 0.0) > 0
                        for k in ("temporal_smooth", "temporal_track"))
    if uses_temporal:
        print(f"seed {seed}: 시간적 일관성 항 사용 (짝 window 를 함께 추론한다)", flush=True)

    print(f"seed {seed}: train={len(train_data)} val={len(val_data)} "
          f"params={model.num_parameters} steps/epoch={steps_per_epoch}", flush=True)

    best = {"metric": float("inf"), "epoch": -1}
    history, patience = [], 0
    for epoch in range(int(training["max_epochs"])):
        model.train()
        order = torch.randperm(len(train_data), device=device, generator=generator)
        started, running = time.time(), {}
        for step in range(steps_per_epoch):
            index = order[step * batch_size:(step + 1) * batch_size]
            step_geom = geom
            if geom is not None and uses_temporal:
                from ..data.augmentations import sample_geom

                step_geom = sample_geom(len(index), geom, device,
                                        train_data.joints.dtype, generator)
            batch = train_data.batch(index, noise, generator, step_geom)
            prediction = model(batch["history"], batch["history_time_ms"], batch["horizon_ms"],
                               batch["handedness"], batch["visibility"])
            # 시간적 일관성 항을 쓰면 anchor 가 한 프레임 앞선 window 도 함께 추론한다.
            previous = None
            if uses_temporal and train_data.prev_index is not None:
                prev_batch = train_data.batch(train_data.prev_index[index], noise,
                                              generator, step_geom)
                prev_pred = model(prev_batch["history"], prev_batch["history_time_ms"],
                                  prev_batch["horizon_ms"], prev_batch["handedness"],
                                  prev_batch["visibility"])
                previous = (prev_pred, prev_batch["target"])
            loss, terms = nowcast_loss(prediction, batch["target"], batch["anchor"],
                                       batch["horizon_ms"], weights=weights,
                                       previous=previous)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            optimizer.step()
            for name, value in terms.items():
                running[name] = running.get(name, 0.0) + value / steps_per_epoch

        val_errors = evaluate_split(model, val_data, noise if val_uses_noise else None)
        metric = float(np.mean([val_errors[h] for h in selection]))
        history.append({"epoch": epoch, "seconds": round(time.time() - started, 1),
                        "loss_terms": {k: round(v, 6) for k, v in running.items()},
                        "val_mpjpe_mm": {str(k): round(v, 4) for k, v in val_errors.items()},
                        "selection_metric_mm": round(metric, 4)})
        print(f"  epoch {epoch:>3} sel={metric:7.3f} mm  "
              + " ".join(f"{int(h)}ms={val_errors[h]:6.3f}" for h in sorted(val_errors)),
              flush=True)

        if scheduler is not None:
            scheduler.step(metric)
            history[-1]["lr"] = optimizer.param_groups[0]["lr"]

        if metric < best["metric"] - 1e-6:
            best = {"metric": metric, "epoch": epoch, "val_mpjpe_mm": val_errors}
            patience = 0
            ckpt.save(out_dir / "best_model.pt", model, cfg, seed,
                      cache_dir / "dataset_manifest.json",
                      {"best": best, "epoch": epoch})
        else:
            patience += 1
            if patience >= int(training["early_stopping_patience"]):
                print(f"  early stop at epoch {epoch} (best {best['epoch']})", flush=True)
                break

    ckpt.write_history(out_dir / "train_history.json", history)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="delay-conditioned nowcaster 학습")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="생략하면 config 의 seed 를 모두 돈다")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(REPO_ROOT / cfg["data_config"])
    root = (Path(args.out_dir) if args.out_dir
            else REPO_ROOT / data_cfg["output_root"] / data_cfg.name / cfg.name)
    seeds = [args.seed] if args.seed is not None else list(cfg["training"]["seeds"])

    print(f"config     : {cfg.path} (hash {cfg.hash})")
    print(f"data config: {data_cfg.path} (hash {data_cfg.hash})")
    print(f"output dir : {root}")

    results = {}
    for seed in seeds:
        out_dir = root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results[seed] = train(cfg, data_cfg, seed, out_dir, args.device)

    # seed 를 따로 실행해도 이전 seed 결과를 덮지 않도록 병합한다
    summary_path = root / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"config_hash": cfg.hash, "data_config_hash": data_cfg.hash,
                    "run": run_metadata(cfg)})
    summary["seeds"] = {**summary.get("seeds", {}),
                        **{str(k): v for k, v in results.items()}}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary    : {root / 'training_summary.json'}")
    for seed, best in results.items():
        print(f"  seed {seed}: best epoch {best['epoch']}, selection {best['metric']:.3f} mm")


if __name__ == "__main__":
    main()
