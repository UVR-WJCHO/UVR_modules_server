"""Subject-disjoint split 과 split manifest.

B 계획 §4.1 이 고정한 v1 split 을 config 에서 읽어 manifest 로 굳힌다.
기존 `train.txt`/`val.txt`/`test.txt` 는 서로 포함 관계라 사용하지 않는다.

  python -m modules.delay_nowcasting.data.splits \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..config import REPO_ROOT, ResolvedConfig, load_config, run_metadata

SPLITS: tuple[str, ...] = ("train", "val", "test")
_SEQUENCE_RE = re.compile(r"^(P\d{4})_[0-9a-f]+$")


def subject_of(sequence_id: str) -> str:
    match = _SEQUENCE_RE.match(sequence_id)
    if match is None:
        raise ValueError(f"sequence id 형식이 P####_<hex> 가 아니다: {sequence_id}")
    return match.group(1)


def discover_sequences(cfg: ResolvedConfig) -> list[str]:
    """config 의 sequence_list 에 적힌 sequence 를 읽고 raw 경로 존재를 확인한다."""
    dataset = cfg["dataset"]
    listing = Path(dataset["sequence_list"])
    raw_root = Path(dataset["raw_root"])
    sequences = [line.strip() for line in listing.read_text().splitlines() if line.strip()]

    missing = [s for s in sequences if not (raw_root / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"raw_root 에 없는 sequence {len(missing)}개: {missing[:5]}")
    return sorted(sequences)


def build_split_manifest(cfg: ResolvedConfig) -> dict:
    split_cfg = cfg["split"]
    sequences = discover_sequences(cfg)

    assignment: dict[str, list[str]] = {name: [] for name in SPLITS}
    subjects: dict[str, list[str]] = {name: list(split_cfg[name]) for name in SPLITS}
    subject_to_split = {
        subject: name for name in SPLITS for subject in subjects[name]
    }

    unassigned: list[str] = []
    for sequence_id in sequences:
        split_name = subject_to_split.get(subject_of(sequence_id))
        if split_name is None:
            unassigned.append(sequence_id)
        else:
            assignment[split_name].append(sequence_id)

    manifest = {
        "split_name": split_cfg["name"],
        "config_name": cfg.name,
        "config_hash": cfg.hash,
        "source_dataset": cfg["dataset"]["source_dataset"],
        "sequence_list": cfg["dataset"]["sequence_list"],
        "subjects": subjects,
        "sequences": assignment,
        "sequence_counts": {name: len(assignment[name]) for name in SPLITS},
        "unassigned_sequences": unassigned,
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict) -> None:
    """B 계획 §11 Phase 0 검증: train/val/test subject 교집합이 0."""
    subjects = manifest["subjects"]
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            overlap = set(subjects[a]) & set(subjects[b])
            if overlap:
                raise ValueError(f"subject 가 {a}/{b} 에 중복된다: {sorted(overlap)}")

    seen: dict[str, str] = {}
    for name in SPLITS:
        for sequence_id in manifest["sequences"][name]:
            if sequence_id in seen:
                raise ValueError(f"sequence {sequence_id} 가 {seen[sequence_id]}/{name} 에 중복된다")
            seen[sequence_id] = name
            if subject_of(sequence_id) not in subjects[name]:
                raise ValueError(f"sequence {sequence_id} 가 {name} subject 목록과 어긋난다")

    for name in SPLITS:
        if not manifest["sequences"][name]:
            raise ValueError(f"{name} split 이 비어 있다")


def write_split_manifest(cfg: ResolvedConfig, out_dir: Path) -> Path:
    manifest = build_split_manifest(cfg)
    manifest["run"] = run_metadata(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="subject-disjoint split manifest 생성")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    path = write_split_manifest(cfg, out_dir)

    manifest = json.loads(path.read_text())
    print(f"config      : {cfg.path} (hash {cfg.hash})")
    print(f"output dir  : {out_dir}")
    print(f"split       : {manifest['split_name']}")
    for name in SPLITS:
        print(f"  {name:<5} {manifest['sequence_counts'][name]:>3} sequences  "
              f"subjects={manifest['subjects'][name]}")
    if manifest["unassigned_sequences"]:
        print(f"  unassigned {len(manifest['unassigned_sequences'])} sequences")
    print(f"manifest    : {path}")


if __name__ == "__main__":
    main()
