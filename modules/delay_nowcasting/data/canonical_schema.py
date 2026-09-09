"""Canonical skeleton / 단위 / 좌표계 계약.

B 계획 §3.1, §4.3, §5.1 이 요구하는 "모든 모듈이 공유하는 하나의 정의"를 여기서 고정한다.
adapter, baseline, model, metric 은 joint index 나 단위를 자체적으로 정의하지 않고
반드시 이 모듈의 상수를 import 한다.

고정 사항:
  - 21-joint canonical 순서 (= WiLoR/DexYCB 출력 순서, `_comm/SERVER_RESULT_PROTOCOL.md` §3)
  - 거리 meter, 시간 integer nanosecond
  - dataset -> canonical joint index mapping 은 코드가 아니라
    `configs/joint_maps/*.json` 에서 읽는다
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- Joint 순서 (B 계획 §3.1) ---------------------------------------------------
JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_mcp", "thumb_pip_or_ip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
)
NUM_JOINTS = len(JOINT_NAMES)
JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

WRIST = 0
FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)          # B 계획 §9.1
MCP_JOINTS: tuple[int, ...] = (1, 5, 9, 13, 17)
FINGER_ROOTS: tuple[int, ...] = MCP_JOINTS

# wrist -> 각 손가락 root, 그리고 손가락 내부 chain. 총 20 edge 의 tree.
BONES: tuple[tuple[int, int], ...] = tuple(
    [(WRIST, root) for root in FINGER_ROOTS]
    + [(j, j + 1) for root in FINGER_ROOTS for j in range(root, root + 3)]
)
NUM_BONES = len(BONES)

HANDEDNESS: tuple[str, ...] = ("LEFT", "RIGHT")

# --- 단위 계약 (B 계획 §4.3) ----------------------------------------------------
LENGTH_UNIT = "meter"
TIME_UNIT = "nanosecond"

# meter/mm 혼입 검출용 경계. 성인 손의 bone 은 meter 단위에서 이 범위 안에 있고,
# mm 로 들어오면 세 자릿수 커지므로 즉시 걸린다.
MAX_PLAUSIBLE_BONE_M = 0.15
MIN_PLAUSIBLE_PALM_M = 0.04
PALM_BONE = (WRIST, JOINT_INDEX["middle_mcp"])

_JOINT_MAP_DIR = Path(__file__).resolve().parent.parent / "configs" / "joint_maps"


def load_joint_map(name: str) -> dict:
    """`configs/joint_maps/<name>.json` 을 읽고 index_map 이 0..20 의 permutation 인지 확인."""
    path = _JOINT_MAP_DIR / f"{name}.json"
    spec = json.loads(path.read_text())
    index_map = spec["index_map"]
    if sorted(index_map) != list(range(NUM_JOINTS)):
        raise ValueError(
            f"{path}: index_map 이 0..{NUM_JOINTS - 1} 의 permutation 이 아니다 -> {index_map}"
        )
    spec["index_map"] = np.asarray(index_map, dtype=np.int64)
    return spec


# --- 검증 -----------------------------------------------------------------------
def check_joint_array(joints: np.ndarray) -> np.ndarray:
    """(..., 21, 3) float 배열인지, finite 인지 확인하고 그대로 돌려준다."""
    arr = np.asarray(joints)
    if arr.ndim < 2 or arr.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(f"joint 배열 shape 은 (..., {NUM_JOINTS}, 3) 이어야 한다: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("joint 배열에 NaN/Inf 가 있다")
    return arr


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """(..., 21, 3) -> (..., 20) bone 길이."""
    arr = check_joint_array(joints)
    a = arr[..., [b[0] for b in BONES], :]
    b = arr[..., [b[1] for b in BONES], :]
    return np.linalg.norm(a - b, axis=-1)


def assert_metric_units(joints: np.ndarray) -> None:
    """거리 단위가 meter 인지 bone 길이 분포로 검사한다 (mm 혼입 검출)."""
    lengths = bone_lengths(joints)
    flat = lengths.reshape(-1, NUM_BONES)
    median = np.median(flat, axis=0)
    if median.max() > MAX_PLAUSIBLE_BONE_M:
        raise ValueError(
            f"bone 길이 median 최대 {median.max():.4f} > {MAX_PLAUSIBLE_BONE_M} m. "
            f"단위가 meter 가 아닐 가능성이 높다 ({LENGTH_UNIT} 로 통일해야 한다)"
        )
    palm = median[BONES.index(PALM_BONE)]
    if palm < MIN_PLAUSIBLE_PALM_M:
        raise ValueError(
            f"wrist->middle_mcp median {palm:.4f} < {MIN_PLAUSIBLE_PALM_M} m. "
            "스케일이 무너졌거나 degenerate pose 다"
        )


def chirality(joints: np.ndarray) -> np.ndarray:
    """(..., 21, 3) -> (...) 부호가 손의 좌/우를 가르는 스칼라.

    palm 평면의 법선과 thumb 방향의 내적이다. palm frame(§3.3)은 x 축을
    `index_mcp - little_mcp` 로 잡아 chirality 를 정의상 지워버리므로, 좌/우가
    실제로 거울상인지는 이렇게 world 좌표에서 확인해야 한다.
    RIGHT 는 양수, LEFT 는 음수여야 한다.
    """
    arr = check_joint_array(joints)
    wrist = arr[..., WRIST, :]
    x = arr[..., JOINT_INDEX["index_mcp"], :] - arr[..., JOINT_INDEX["little_mcp"], :]
    y = arr[..., JOINT_INDEX["middle_mcp"], :] - wrist
    normal = np.cross(x, y)
    return np.einsum("...i,...i->...", normal, arr[..., JOINT_INDEX["thumb_mcp"], :] - wrist)


def assert_timestamps_ns(timestamps: np.ndarray) -> None:
    """timestamp 가 integer nanosecond 이고 sequence 안에서 strictly increasing 인지 확인."""
    ts = np.asarray(timestamps)
    if not np.issubdtype(ts.dtype, np.integer):
        raise ValueError(f"timestamp 는 integer {TIME_UNIT} 이어야 한다: dtype={ts.dtype}")
    if ts.size > 1 and not np.all(np.diff(ts) > 0):
        raise ValueError("timestamp 가 strictly increasing 이 아니다")


# --- Rigid transform (B 계획 §3.2 / §5.1) ---------------------------------------
def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """(..., 4) [w, x, y, z] -> (..., 3, 3) rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """(..., 3, 3) + (..., 3) -> (..., 4, 4) row-major homogeneous transform."""
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """(..., 4, 4) rigid transform 의 역변환."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_transform(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(4, 4) 또는 (..., 4, 4) transform 을 (..., N, 3) 점에 적용."""
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return np.einsum("...ij,...nj->...ni", R, p) + t[..., None, :]


# --- Canonical frame cache schema (B 계획 §4.3) ---------------------------------
def canonical_frame_schema() -> dict:
    """`canonical_frames.parquet` 의 polars schema.

    polars 를 import 하는 유일한 지점이라 lazy import 로 둔다(상수만 쓰는 모듈이
    polars 없이도 import 되도록).
    """
    import polars as pl

    return {
        "sequence_id": pl.Utf8,
        "subject_id": pl.Utf8,
        "frame_idx": pl.Int64,
        "timestamp_ns": pl.Int64,
        "handedness": pl.Utf8,
        "track_id": pl.Utf8,
        "valid": pl.Boolean,
        "joints_world": pl.Array(pl.Float32, NUM_JOINTS * 3),
        "joints_camera": pl.Array(pl.Float32, NUM_JOINTS * 3),
        "visibility": pl.Array(pl.UInt8, NUM_JOINTS),
        "camera_to_world": pl.Array(pl.Float32, 16),
        "source_dataset": pl.Utf8,
        "source_frame_key": pl.Utf8,
    }


def validate_canonical_frames(df) -> None:
    """canonical frame DataFrame 이 schema/단위/handedness 계약을 지키는지 확인."""
    expected = canonical_frame_schema()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"canonical frame 에 없는 column: {missing}")
    for col, dtype in expected.items():
        if df.schema[col] != dtype:
            raise ValueError(f"column {col} dtype 이 {dtype} 가 아니라 {df.schema[col]} 다")

    bad_hand = set(df["handedness"].unique().to_list()) - set(HANDEDNESS)
    if bad_hand:
        raise ValueError(f"handedness 값이 {HANDEDNESS} 밖에 있다: {sorted(bad_hand)}")

    valid = df.filter(df["valid"])
    if valid.height:
        joints = valid["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3)
        assert_metric_units(joints)
