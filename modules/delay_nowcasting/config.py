"""실험 config 로딩과 실행 identity.

B 계획 §7.7 / §13 이 요구하는 것:
  - 모든 CLI 가 resolved config, git commit, environment 정보, output directory 를 출력
  - checkpoint 에 config hash 와 seed 를 남겨 재현 가능

같은 config + seed 면 sample key 와 파생 seed 가 항상 같아야 하므로,
난수가 아니라 내용 해시로만 identity 를 만든다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResolvedConfig:
    name: str
    path: Path
    data: dict
    hash: str

    def __getitem__(self, key: str):
        return self.data[key]

    def get(self, key: str, default=None):
        return self.data.get(key, default)


def config_hash(data: dict) -> str:
    """config 내용의 안정적인 해시. key 순서/공백에 영향받지 않는다."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> ResolvedConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config 최상위는 mapping 이어야 한다")
    name = data.get("name", path.stem)
    return ResolvedConfig(name=name, path=path, data=data, hash=config_hash(data))


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict:
    import importlib

    versions = {}
    for module in ("torch", "polars", "numpy", "onnxruntime", "smplx", "yaml"):
        try:
            versions[module] = getattr(importlib.import_module(module), "__version__", "unknown")
        except ImportError:
            versions[module] = None
    return versions


def run_metadata(cfg: ResolvedConfig, seed: int | None = None) -> dict:
    """실행 로그/checkpoint 에 함께 저장할 재현 정보 (B 계획 §7.7)."""
    return {
        "config_name": cfg.name,
        "config_path": str(cfg.path),
        "config_hash": cfg.hash,
        "seed": seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def write_run_metadata(cfg: ResolvedConfig, out_dir: str | Path, seed: int | None = None) -> Path:
    """resolved config 와 run metadata 를 output directory 에 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.data, sort_keys=True, allow_unicode=True)
    )
    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(run_metadata(cfg, seed), indent=2, ensure_ascii=False))
    return meta_path


# --- 결정론적 identity ----------------------------------------------------------
def sample_key(sequence_id: str, handedness: str, target_frame_idx: int, horizon_ms: float) -> str:
    """window sample 의 안정적인 key. 같은 config/seed 면 항상 같은 문자열이 나온다."""
    return f"{sequence_id}|{handedness}|{int(target_frame_idx):07d}|h{float(horizon_ms):07.2f}"


def derive_seed(base_seed: int, *tags: str) -> int:
    """base seed 와 tag 로부터 결정론적인 32bit seed 를 만든다 (worker/fold 별 분리용)."""
    payload = "|".join((str(base_seed), *tags)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
