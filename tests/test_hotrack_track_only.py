from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

from modules_hotrack import (  # noqa: E402
    _sanitize_track_only_result,
    _sanitize_track_only_tracking_json,
    apply_track_only_hotrack_runtime_flags,
)


class DummyHotrack:
    pass


def test_track_only_flags_match_hograph_plus_contract():
    hotrack = DummyHotrack()
    hotrack.use_dino_id = True
    hotrack.enable_id_recovery = True
    hotrack.compute_dino_similarity = False

    apply_track_only_hotrack_runtime_flags(hotrack)

    assert hotrack.use_dino_id is True
    assert hotrack.enable_id_recovery is True
    assert hotrack.compute_dino_similarity is False
    assert hotrack.enable_structural_ops is False
    assert hotrack.replay_enable_structural_split_logic is False
    assert hotrack.track_hand_masks is False
    assert hotrack.skip_existing_object_box_prompts is True
    assert hotrack.use_hand_matched_det_boxes_only is True
    assert hotrack.use_all_object_boxes is False
    assert hotrack.enable_component_promote_from_det is False
    assert hotrack.enable_detector_guided_single_component_split is True
    assert hotrack.skip_existing_object_box_prompts_for_split_candidates is False
    assert hotrack.attach_proxy_backfill_enabled is False
    assert hotrack.enable_post_track_duplicate_removal is False


def test_track_only_result_sanitizes_structural_fields():
    result = {
        "new_objects": [101],
        "struct_events": [{"type": "split_apply"}],
        "id_transitions": [{"type": "split"}],
        "id_remap": {"100": 101},
    }

    out = _sanitize_track_only_result(result)

    assert out["runtime_mode"] == "track_only"
    assert out["new_objects"] == [101]
    assert out["struct_events"] == []
    assert out["id_transitions"] == []
    assert out["id_remap"] == {}


def test_track_only_tracking_json_sanitizes_structural_fields(tmp_path: Path):
    track_path = tmp_path / "000000_track.json"
    track_path.write_text(
        json.dumps(
            {
                "new_objects": [101],
                "struct_events": [{"type": "merge_apply"}],
                "id_transitions": [{"type": "merge"}],
                "id_remap": {"101": 200},
            }
        ),
        encoding="utf-8",
    )

    _sanitize_track_only_tracking_json(track_path)

    payload = json.loads(track_path.read_text(encoding="utf-8"))
    assert payload["runtime_mode"] == "track_only"
    assert payload["new_objects"] == [101]
    assert payload["struct_events"] == []
    assert payload["id_transitions"] == []
    assert payload["id_remap"] == {}
