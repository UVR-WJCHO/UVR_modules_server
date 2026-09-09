"""Checkpoint 저장/복원 (B 계획 §7.7).

각 checkpoint 에는 git commit, config, seed, dataset manifest hash, normalization
statistics 가 함께 들어간다. 이게 없으면 결과를 나중에 재현할 수 없다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..config import ResolvedConfig, run_metadata
from ..methods.features import HORIZON_CLAMP_MS
from ..methods.residual_mlp import ResidualMLP
from ..methods.tcn_graph import TCNGraph
from ..methods.tcn_lite import TCNLite

MODEL_REGISTRY = {"residual_mlp": ResidualMLP, "tcn_lite": TCNLite,
                  "tcn_graph": TCNGraph}


def manifest_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_model(model_cfg: dict) -> torch.nn.Module:
    params = dict(model_cfg)
    model_type = params.pop("type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 model type: {model_type}")
    return MODEL_REGISTRY[model_type](**params)


def save(path: Path, model: torch.nn.Module, cfg: ResolvedConfig, seed: int,
         dataset_manifest: Path, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": cfg["model"],
        "config": cfg.data,
        "config_hash": cfg.hash,
        "seed": seed,
        "dataset_manifest_hash": manifest_hash(dataset_manifest),
        # horizon encoding 의 정규화 기준. 바뀌면 예전 checkpoint 는 호환되지 않는다.
        "horizon_clamp_ms": HORIZON_CLAMP_MS,
        "run": run_metadata(cfg, seed),
        **extra,
    }, path)


def load(path: Path, device: str = "cuda") -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
