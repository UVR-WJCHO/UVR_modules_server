import numpy as np
import pytest

from modules.delay_nowcasting.data import canonical_schema as cs


def test_joint_order_is_21_and_unique():
    assert cs.NUM_JOINTS == 21
    assert len(set(cs.JOINT_NAMES)) == 21
    assert cs.JOINT_NAMES[cs.WRIST] == "wrist"
    assert tuple(cs.JOINT_INDEX[cs.JOINT_NAMES[i]] for i in range(21)) == tuple(range(21))


def test_fingertips_and_mcps_match_names():
    assert tuple(cs.JOINT_NAMES[i].endswith("_tip") for i in cs.FINGERTIPS) == (True,) * 5
    assert tuple(cs.JOINT_NAMES[i].endswith("_mcp") for i in cs.MCP_JOINTS) == (True,) * 5


def test_bones_form_a_tree_over_valid_indices():
    assert cs.NUM_BONES == 20
    for a, b in cs.BONES:
        assert 0 <= a < cs.NUM_JOINTS and 0 <= b < cs.NUM_JOINTS
    # 20 edge 로 21 node 를 잇고 wrist 가 root 인 tree 여야 한다
    assert len(set(cs.BONES)) == 20
    children = {b for _, b in cs.BONES}
    assert children == set(range(1, cs.NUM_JOINTS))


def test_joint_map_is_a_permutation():
    spec = cs.load_joint_map("mano_to_canonical")
    assert sorted(spec["index_map"].tolist()) == list(range(21))
    assert len(spec["tip_vertex_ids"]) == 5


def _synthetic_hand(scale=1.0):
    """wrist 원점, 손가락마다 일정 간격으로 뻗은 21-joint 손 (meter)."""
    joints = np.zeros((cs.NUM_JOINTS, 3), dtype=np.float64)
    for f, root in enumerate(cs.FINGER_ROOTS):
        base = np.array([0.02 * (f - 2), 0.09, 0.0])
        joints[root] = base
        for k in range(1, 4):
            joints[root + k] = base + np.array([0.0, 0.025 * k, 0.0])
    return joints * scale


def test_metric_unit_check_accepts_meter_rejects_mm():
    hand = _synthetic_hand()
    cs.assert_metric_units(hand)
    with pytest.raises(ValueError, match="meter"):
        cs.assert_metric_units(hand * 1000.0)


def test_joint_array_rejects_wrong_shape_and_nan():
    with pytest.raises(ValueError):
        cs.check_joint_array(np.zeros((20, 3)))
    bad = _synthetic_hand()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        cs.check_joint_array(bad)


def test_timestamps_must_be_integer_ns_and_increasing():
    cs.assert_timestamps_ns(np.array([0, 33_333_333, 66_666_666], dtype=np.int64))
    with pytest.raises(ValueError, match="integer"):
        cs.assert_timestamps_ns(np.array([0.0, 33.3], dtype=np.float64))
    with pytest.raises(ValueError, match="increasing"):
        cs.assert_timestamps_ns(np.array([0, 66_666_666, 33_333_333], dtype=np.int64))


def test_camera_world_camera_round_trip():
    rng = np.random.default_rng(0)
    quat = rng.normal(size=4)
    T = cs.make_transform(cs.quat_wxyz_to_matrix(quat), rng.normal(size=3))
    hand = _synthetic_hand()

    world = cs.transform_points(T, hand)
    back = cs.transform_points(cs.invert_transform(T), world)
    assert np.abs(back - hand).max() < 1e-9

    # rigid transform 은 bone 길이를 보존해야 한다
    assert np.abs(cs.bone_lengths(world) - cs.bone_lengths(hand)).max() < 1e-9


def test_quat_to_matrix_is_orthonormal_and_batched():
    rng = np.random.default_rng(1)
    quats = rng.normal(size=(7, 4))
    R = cs.quat_wxyz_to_matrix(quats)
    assert R.shape == (7, 3, 3)
    eye = np.einsum("nij,nkj->nik", R, R)
    assert np.abs(eye - np.eye(3)).max() < 1e-9
    assert np.abs(np.linalg.det(R) - 1.0).max() < 1e-9
