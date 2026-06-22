from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

import modules_hotrack  # noqa: E402


def test_build_interactive_hotrack_segmentor_from_env(monkeypatch):
    captured = {}

    class DummySegmentor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(modules_hotrack, "InteractiveHoTrackSegmentor", DummySegmentor)
    monkeypatch.setenv("UVR_HOTRACK_OUTPUT_DIR", "out/hotrack")
    monkeypatch.setenv("UVR_HOTRACK_VIDEO_NAME", "win_test")
    monkeypatch.setenv("UVR_HOTRACK_YOLO_MODEL", "weights/yolo_100doh_best.pt")
    monkeypatch.setenv("UVR_HOTRACK_SAM2_VARIANT", "small")
    monkeypatch.setenv("UVR_HOTRACK_SAM2_CHECKPOINT", "weights/sam2_small.pt")
    monkeypatch.setenv("UVR_HOTRACK_MAX_SIDE", "960")
    monkeypatch.setenv("UVR_HOTRACK_HO_THRESH_HAND", "0.25")
    monkeypatch.setenv("UVR_HOTRACK_HO_THRESH_OBJ", "0.35")
    monkeypatch.setenv("UVR_HOTRACK_TARGET_CONTACT", "N")
    monkeypatch.setenv("UVR_HOTRACK_BACKFILL_WINDOW", "24")
    monkeypatch.setenv("UVR_HOTRACK_SAVE_TRACKING_JSON", "0")
    monkeypatch.setenv("UVR_HOTRACK_DINO_ALLOW_DOWNLOAD", "1")

    segmentor = modules_hotrack.build_interactive_hotrack_segmentor_from_env(interactive_window=False)

    assert isinstance(segmentor, DummySegmentor)
    assert captured["output_dir"] == "out/hotrack"
    assert captured["video_name"] == "win_test"
    assert captured["yolo_model_path"] == "weights/yolo_100doh_best.pt"
    assert captured["sam2_variant"] == "small"
    assert captured["sam2_checkpoint"] == "weights/sam2_small.pt"
    assert captured["max_side"] == 960
    assert captured["ho_thresh_hand"] == 0.25
    assert captured["ho_thresh_obj"] == 0.35
    assert captured["target_contact_code"] == "N"
    assert captured["backfill_window"] == 24
    assert captured["save_tracking_json"] is False
    assert captured["dino_allow_download"] is True
    assert captured["interactive_window"] is False
