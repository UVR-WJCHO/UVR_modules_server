import json

import pytest
import yaml

from modules.delay_nowcasting import config as cfgmod

CONFIG_PATH = cfgmod.REPO_ROOT / "modules/delay_nowcasting/configs/data/hot3d_v1.yaml"


def test_load_config_and_units_contract():
    cfg = cfgmod.load_config(CONFIG_PATH)
    assert cfg.name == "hot3d_v1"
    # B 계획 §4.3: 모든 거리 meter, timestamp integer nanosecond
    assert cfg["units"] == {"length": "meter", "time": "nanosecond"}


def test_config_hash_is_content_stable():
    a = {"b": 1, "a": {"y": 2, "x": [1, 2, 3]}}
    b = {"a": {"x": [1, 2, 3], "y": 2}, "b": 1}
    assert cfgmod.config_hash(a) == cfgmod.config_hash(b)
    assert cfgmod.config_hash(a) != cfgmod.config_hash({**a, "b": 2})


def test_sample_key_is_reproducible_and_discriminating():
    key = cfgmod.sample_key("P0014_abc123", "RIGHT", 42, 66.0)
    assert key == cfgmod.sample_key("P0014_abc123", "RIGHT", 42, 66.0)
    assert key != cfgmod.sample_key("P0014_abc123", "LEFT", 42, 66.0)
    assert key != cfgmod.sample_key("P0014_abc123", "RIGHT", 43, 66.0)
    assert key != cfgmod.sample_key("P0014_abc123", "RIGHT", 42, 100.0)
    # frame index 는 zero padding 되어 문자열 정렬이 시간 순서와 일치한다
    assert cfgmod.sample_key("s", "RIGHT", 9, 33.0) < cfgmod.sample_key("s", "RIGHT", 10, 33.0)


def test_derive_seed_is_deterministic():
    assert cfgmod.derive_seed(0, "train", "fold0") == cfgmod.derive_seed(0, "train", "fold0")
    assert cfgmod.derive_seed(0, "train") != cfgmod.derive_seed(1, "train")
    assert 0 <= cfgmod.derive_seed(0, "train") < 2 ** 32


def test_write_run_metadata(tmp_path):
    cfg = cfgmod.load_config(CONFIG_PATH)
    meta_path = cfgmod.write_run_metadata(cfg, tmp_path, seed=0)
    meta = json.loads(meta_path.read_text())
    assert meta["config_hash"] == cfg.hash
    assert meta["seed"] == 0
    assert meta["packages"]["numpy"] is not None
    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert cfgmod.config_hash(resolved) == cfg.hash


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        cfgmod.load_config(bad)
