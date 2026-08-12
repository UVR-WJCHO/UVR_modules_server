"""Bring several assembly solves into one frame and write them out.

Each solve places its units in the camera frame of the capture it was solved
against, so two solves do not share coordinates — until a unit appears in both.
That unit is the same object seen twice, so comparing its two placements gives
the transform between the frames, and one solve's units can be carried into the
other's. Chaining that way is what turns a set of pairwise assemblies into a
single object.

The written form is the `transforms.json` that `metaobj_wrapper` consumes: a
translation, a rotation and a scale per part, in Blender's world axes. That
format can only carry a rigid motion with one uniform size, which is what the
poses are — provided nothing downstream reintroduced a size that differs across
and along a part's axis. `part_entry` refuses rather than quietly rounding one
off, because the error would land in the assembled model with nothing to show
where it came from.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# OpenCV camera axes to Blender world axes, camera at the origin:
# OpenCV is +x right, +y down, +z forward; Blender is +x right, +y forward,
# +z up. A pose conjugates under the change of basis: R' = M R M^T, t' = M t.
M_CV_TO_BLENDER = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]], dtype=np.float64)

MAX_ANISOTROPY = 1e-4


class NotASimilarity(ValueError):
    """A pose that scales differently across and along some axis, which the
    output format has no way to express."""


def decompose_similarity(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(R, t, s) from a 4x4 pose, refusing anything that is not one size."""
    M = np.asarray(T, dtype=np.float64)[:3, :3]
    sv = np.linalg.svd(M, compute_uv=False)
    if sv.min() <= 1e-12:
        raise NotASimilarity("degenerate pose")
    if sv.max() / sv.min() - 1.0 > MAX_ANISOTROPY:
        raise NotASimilarity(
            f"pose scales by {sv.max():.6f} one way and {sv.min():.6f} another "
            f"({sv.max() / sv.min() - 1:.4%} apart). transforms.json carries a "
            f"single scale per axis of the part's own frame, so this cannot be "
            f"written without silently changing the shape.")
    s = float(sv.mean())
    U, _, Vt = np.linalg.svd(M / s)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, np.asarray(T, dtype=np.float64)[:3, 3].copy(), s


def chain_solves(solves: Sequence[Dict[str, np.ndarray]],
                 verbose: bool = True) -> Dict[str, np.ndarray]:
    """Merge per-assembly poses into the first solve's frame.

    `solves` is a list of {unit name: 4x4 pose}. Each solve after the first must
    share at least one unit with what has been merged so far; that unit is the
    bridge. Its two poses differ only by the change of frame, so the bridge is
    exact rather than fitted — and the size it implies is a free check on
    whether the two solves agreed, since a bridge that is not 1.0 means they
    measured the same part differently.
    """
    if not solves:
        return {}
    merged: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64).copy()
                                     for k, v in solves[0].items()}
    for solve in solves[1:]:
        shared = [k for k in solve if k in merged]
        if not shared:
            raise ValueError(
                f"no unit links {sorted(solve)} to {sorted(merged)} — the two "
                f"solves have no object in common, so their frames cannot be "
                f"related")
        via = shared[0]
        bridge = merged[via] @ np.linalg.inv(np.asarray(solve[via], dtype=np.float64))
        if verbose:
            s = float(np.cbrt(abs(np.linalg.det(bridge[:3, :3]))))
            print(f"[chain] via unit {via}: bridge scale {s:.5f} "
                  f"(1.0 = the two solves agree on its size)")
        for k, T in solve.items():
            if k not in merged:
                merged[k] = bridge @ np.asarray(T, dtype=np.float64)
    return merged


def part_entry(name: str, T: np.ndarray, target_frame: str = "blender",
               euler_order: str = "xyz") -> dict:
    """One transforms.json entry from a 4x4 pose in the OpenCV camera frame."""
    R, t, s = decompose_similarity(T)
    if target_frame == "blender":
        R = M_CV_TO_BLENDER @ R @ M_CV_TO_BLENDER.T
        t = M_CV_TO_BLENDER @ t
    elif target_frame != "opencv":
        raise ValueError("target_frame must be 'blender' or 'opencv'")
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()          # scipy gives (x, y, z, w)
    return {
        "name": name,
        "translation": [float(v) for v in t],
        "rotation_euler_degrees": [float(v) for v in rot.as_euler(euler_order,
                                                                 degrees=True)],
        "rotation_quaternion": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
        "scale": [s, s, s],
    }


def write_transforms(path: Path, names_and_poses: Iterable[Tuple[str, np.ndarray]],
                     target_frame: str = "blender", euler_order: str = "xyz",
                     ) -> List[dict]:
    parts = [part_entry(n, T, target_frame, euler_order) for n, T in names_and_poses]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"parts": parts}, indent=2, ensure_ascii=False))
    return parts


def stage_meshes(mesh_dir: Path, units: Sequence[str], out_dir: Path) -> List[Path]:
    """Copy each unit's mesh out as mesh_<i>.glb, in the given order.

    The corrected meshes are copied, not the reconstructions they came from:
    the poses were solved against the corrected shape and mean nothing applied
    to anything else.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, unit in enumerate(units):
        src = Path(mesh_dir) / f"part_{unit}" / "mesh.glb"
        if not src.exists():
            raise FileNotFoundError(f"unit {unit}: {src} missing — run Stage 1 first")
        dst = out_dir / f"mesh_{i}.glb"
        shutil.copyfile(src, dst)
        written.append(dst)
    return written
