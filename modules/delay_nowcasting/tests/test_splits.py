import json
from pathlib import Path

import pytest

from modules.delay_nowcasting.config import REPO_ROOT, load_config
from modules.delay_nowcasting.data import splits

CONFIG_PATH = REPO_ROOT / "modules/delay_nowcasting/configs/data/hot3d_v1.yaml"
CFG = load_config(CONFIG_PATH)
DATASET_AVAILABLE = Path(CFG["dataset"]["raw_root"]).is_dir()
needs_dataset = pytest.mark.skipif(not DATASET_AVAILABLE, reason="HOT3D raw_root 접근 불가")


def test_config_split_subjects_are_disjoint():
    """B 계획 §11 Phase 0 검증: train/val/test subject 교집합이 0."""
    sets = {name: set(CFG["split"][name]) for name in splits.SPLITS}
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()


def test_subject_of():
    assert splits.subject_of("P0014_9c030609") == "P0014"
    with pytest.raises(ValueError):
        splits.subject_of("not_a_sequence")


def test_validate_split_manifest_catches_subject_overlap():
    bad = {
        "subjects": {"train": ["P0001"], "val": ["P0001"], "test": ["P0014"]},
        "sequences": {"train": [], "val": [], "test": []},
    }
    with pytest.raises(ValueError, match="중복"):
        splits.validate_split_manifest(bad)


def test_validate_split_manifest_catches_sequence_in_wrong_split():
    bad = {
        "subjects": {"train": ["P0001"], "val": ["P0012"], "test": ["P0014"]},
        "sequences": {"train": ["P0012_abc123"], "val": ["P0012_def456"], "test": ["P0014_0a1b2c"]},
    }
    with pytest.raises(ValueError, match="subject 목록"):
        splits.validate_split_manifest(bad)


@needs_dataset
def test_discover_sequences_matches_cached_dataset():
    sequences = splits.discover_sequences(CFG)
    assert len(sequences) == 136
    assert len(set(sequences)) == len(sequences)


@needs_dataset
def test_build_split_manifest(tmp_path):
    manifest = splits.build_split_manifest(CFG)
    counts = manifest["sequence_counts"]
    assert counts == {"train": 83, "val": 22, "test": 31}
    assert sum(counts.values()) + len(manifest["unassigned_sequences"]) == 136
    assert manifest["unassigned_sequences"] == []

    path = splits.write_split_manifest(CFG, tmp_path)
    reloaded = json.loads(path.read_text())
    assert reloaded["sequences"] == manifest["sequences"]
    splits.validate_split_manifest(reloaded)
