"""Similarity-transform and rotation utilities.

Conventions used throughout the package:
  * Camera frame is OpenCV: +x right, +y down, +z forward. "Visual up" is -y.
  * A pose is a 4x4 Sim(3) matrix T with T[:3,:3] = s*R, T[:3,3] = t, det(R)=+1.
  * Quaternions are (w, x, y, z).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Sim(3)
# --------------------------------------------------------------------------

def compose(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = float(s) * np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def decompose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """T -> (R, t, s). Raises if the linear part is not a scaled rotation."""
    M = np.asarray(T, dtype=np.float64)[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(M), 1e-30)))
    if s <= 1e-12:
        raise ValueError("degenerate transform (non-positive determinant)")
    R = M / s
    # re-orthonormalise to kill accumulated drift
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, np.asarray(T, dtype=np.float64)[:3, 3].copy(), s


def apply(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=np.float64)
    return P @ np.asarray(T, dtype=np.float64)[:3, :3].T + np.asarray(T)[:3, 3]


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def rot_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation of `theta` radians about `axis`."""
    a = normalize(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rot_align(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Smallest rotation taking `from_vec` onto `to_vec`."""
    a, b = normalize(from_vec), normalize(to_vec)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # 180 deg: pick any axis perpendicular to a
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return rot_about_axis(normalize(np.cross(a, tmp)), math.pi)
    return rot_about_axis(np.cross(a, b), math.acos(max(-1.0, min(1.0, c))))


def quat_from_rot(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z). Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        k = math.sqrt(tr + 1.0) * 2
        q = [0.25 * k, (R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        k = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / k, 0.25 * k, (R[0, 1] + R[1, 0]) / k, (R[0, 2] + R[2, 0]) / k]
    elif R[1, 1] > R[2, 2]:
        k = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / k, (R[0, 1] + R[1, 0]) / k, 0.25 * k, (R[1, 2] + R[2, 1]) / k]
    else:
        k = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / k, (R[0, 2] + R[2, 0]) / k, (R[1, 2] + R[2, 1]) / k, 0.25 * k]
    q = np.array(q, dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_to_rot_torch(q: torch.Tensor) -> torch.Tensor:
    """Differentiable (w,x,y,z) quaternion -> 3x3 rotation."""
    q = q / (q.norm() + 1e-9)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]).reshape(3, 3)


def sim3_torch(q: torch.Tensor, t: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
    """Differentiable (q, t, log s) -> 4x4 Sim(3)."""
    R = quat_to_rot_torch(q)
    T = torch.zeros(4, 4, dtype=R.dtype, device=R.device)
    T[:3, :3] = log_s.exp() * R
    T[:3, 3] = t
    T[3, 3] = 1
    return T


def fibonacci_directions(n: int) -> np.ndarray:
    """`n` roughly uniform unit vectors on the sphere."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def rotation_ball(R0: np.ndarray, max_deg: float, n_shells: int = 3,
                  n_axes: int = 12) -> List[Tuple[str, np.ndarray]]:
    """Rotations within `max_deg` of `R0`, as concentric shells of perturbations.

    Used when a pose is already known to within a bounded reorientation: the
    search then only has to cover that neighbourhood instead of all of SO(3),
    which both removes far-away poses that happen to explain the silhouette and
    leaves far more budget per candidate.
    """
    out = [("solo", np.asarray(R0, dtype=np.float64).copy())]
    axes = fibonacci_directions(int(n_axes))
    for k in range(1, int(n_shells) + 1):
        ang = math.radians(max_deg * k / n_shells)
        for j, a in enumerate(axes):
            out.append((f"d{int(round(math.degrees(ang))):02d}a{j:02d}",
                        rot_about_axis(a, ang) @ R0))
    return out


def rot_log_torch(R: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle vector of a rotation. Well conditioned for the
    modest angles this package deals with; degenerate near a half turn."""
    c = torch.clamp((torch.diagonal(R).sum() - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    th = torch.acos(c)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * torch.sin(th) + 1e-9))


def symmetry_axis(points: np.ndarray, n_bins: int = 36
                  ) -> Tuple[np.ndarray, float]:
    """The mesh axis about which the shape is most rotationally symmetric, and
    how far from symmetric it still is (0 = a perfect surface of revolution).

    Measured on the outline: points are put in cylindrical coordinates about
    each principal axis and the outer radius is taken per angular bin. A shape
    of revolution has the same outer radius at every angle, so the spread of
    those radii, relative to their mean, says how much a turn about that axis
    would change the shape — which is exactly how much the observation can say
    about the turn.
    """
    P = np.asarray(points, dtype=np.float64)
    axes, _ = pca_axes(P)
    c = P.mean(axis=0)
    best = (None, np.inf)
    for i in range(3):
        a = normalize(axes[:, i])
        d = P - c
        along = d @ a
        perp = d - np.outer(along, a)
        u = normalize(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        v = np.cross(a, u)
        phi = np.arctan2(perp @ v, perp @ u)
        r = np.linalg.norm(perp, axis=1)
        bins = np.clip(((phi + np.pi) / (2 * np.pi) * n_bins).astype(int), 0, n_bins - 1)
        outer = np.array([np.percentile(r[bins == b], 90) if np.any(bins == b) else np.nan
                          for b in range(n_bins)])
        outer = outer[np.isfinite(outer)]
        if len(outer) < n_bins // 2 or outer.mean() <= 1e-9:
            continue
        asym = float(outer.std() / outer.mean())
        if asym < best[1]:
            best = (a, asym)
    return (best[0] if best[0] is not None else axes[:, 0]), float(best[1])


def centroid_axis(points: np.ndarray, n_dirs: int = 600, n_slices: int = 20
                  ) -> Tuple[np.ndarray, float]:
    """The axis about which perpendicular cross-sections stay centred.

    A body built around an axis — turned, or repeated in a ring about it — has
    every perpendicular cross-section centred on that axis, however many lobes
    the cross-section has. That makes this test blind to lobe count, unlike a
    radius-constancy test, which is what it is for: it recovers the axis of a
    five-nozzle engine cluster, where measuring how round the outline is would
    reject the true axis outright.

    It cannot stand alone. A cylinder also keeps its centroids in line when cut
    across its width, so this is the fallback for shapes that are not clean
    surfaces of revolution, where `symmetry_axis` already answers.
    """
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0)
    D = P - c
    best = (None, np.inf)
    for a in fibonacci_directions(int(n_dirs)):
        a = normalize(a)
        h = D @ a
        perp = D - np.outer(h, a)
        lo, hi = np.percentile(h, [3, 97])
        if hi - lo < 1e-9:
            continue
        b = np.clip(((h - lo) / (hi - lo) * n_slices).astype(int), 0, n_slices - 1)
        cen = [perp[b == k].mean(axis=0) for k in range(n_slices) if (b == k).sum() > 30]
        if len(cen) < n_slices // 2:
            continue
        cen = np.asarray(cen)
        spread = float(np.linalg.norm(cen - cen.mean(axis=0), axis=1).mean())
        radius = float(np.linalg.norm(perp, axis=1).mean())
        rel = spread / max(radius, 1e-9)
        if rel < best[1]:
            best = (a, rel)
    return (best[0] if best[0] is not None else np.array([0.0, 1.0, 0.0])), float(best[1])


CAMERA_UP = np.array([0.0, -1.0, 0.0])


def central_axis(points: np.ndarray, R_solo: Optional[np.ndarray] = None,
                 n_dirs: int = 1200, n_slices: int = 20, tol: float = 1.30
                 ) -> Tuple[np.ndarray, float]:
    """The axis a part is built around, in its own coordinates.

    Directions are scored by how still the perpendicular cross-sections keep
    their centres: a body turned about an axis, or built as a ring of features
    around one, has every such cross-section centred on it. That test is blind
    to how many lobes the cross-section has, which a roundness test is not — it
    would reject the true axis of a five-nozzle engine cluster outright.

    The test alone is degenerate. A cylinder keeps its centroids in line when
    cut across its width too, and a squat one is no longer across than it is
    tall, so neither the score nor the extent separates the real axis from a
    perpendicular one. What separates them is how the part was photographed:
    it stood up. `R_solo` — the pose Stage 1 found — carries that, and among
    the directions that score alike the one nearest upright is taken. The pose
    is only accurate to a couple of dozen degrees, far too coarse to *be* the
    axis, but the alternatives are a right angle apart, so it is ample to
    choose between them. Precision comes from the geometry, disambiguation
    from the pose.
    """
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0)
    D = P - c
    cands = []
    for a in fibonacci_directions(int(n_dirs)):
        a = normalize(a)
        h = D @ a
        perp = D - np.outer(h, a)
        lo, hi = np.percentile(h, [3, 97])
        if hi - lo < 1e-9:
            continue
        b = np.clip(((h - lo) / (hi - lo) * n_slices).astype(int), 0, n_slices - 1)
        cen = [perp[b == k].mean(axis=0) for k in range(n_slices) if (b == k).sum() > 30]
        if len(cen) < n_slices // 2:
            continue
        cen = np.asarray(cen)
        rel = (float(np.linalg.norm(cen - cen.mean(axis=0), axis=1).mean())
               / max(float(np.linalg.norm(perp, axis=1).mean()), 1e-9))
        cands.append((rel, a))
    if not cands:
        return np.array([0.0, 1.0, 0.0]), float("inf")
    best = min(c[0] for c in cands)
    keep = [c for c in cands if c[0] <= best * tol]
    if R_solo is None:
        return min(keep, key=lambda c: c[0])[1], best
    prior = normalize(np.asarray(R_solo, dtype=np.float64).T @ CAMERA_UP)
    a = max(keep, key=lambda c: abs(float(np.dot(c[1], prior))))[1]
    return (a if np.dot(a, prior) > 0 else -a), best


def tilt_twist(R: np.ndarray, R_ref: np.ndarray, axis_local: np.ndarray
               ) -> Tuple[float, float]:
    """Split the turn from `R_ref` to `R` into degrees about the part's own
    axis (twist) and degrees away from it (tilt)."""
    import torch as _t
    v = rot_log_torch(_t.tensor(R @ np.asarray(R_ref).T, dtype=_t.float32)).numpy()
    a = normalize(np.asarray(R_ref) @ np.asarray(axis_local))
    tw = float(np.dot(v, a))
    return float(np.degrees(np.linalg.norm(v - tw * a))), float(np.degrees(tw))


def sim3_aniso_torch(q: torch.Tensor, t: torch.Tensor, log_sr: torch.Tensor,
                     log_sa: torch.Tensor, axis_local: torch.Tensor) -> torch.Tensor:
    """Sim(3) widened to scale differently across and along the part's axis.

    A reconstruction can come out the right height and the wrong girth, and one
    number cannot fix that without spoiling the other. Scaling across the axis
    and along it separately can, and stays a symmetric positive map, so
    rendering, gradients and winding are unaffected.
    """
    R = quat_to_rot_torch(q)
    a = axis_local / (axis_local.norm() + 1e-9)
    P = torch.outer(a, a)
    S = log_sr.exp() * (torch.eye(3, dtype=R.dtype, device=R.device) - P) + log_sa.exp() * P
    T = torch.zeros(4, 4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R @ S
    T[:3, 3] = t
    T[3, 3] = 1
    return T


def rim_circles(points: np.ndarray, axis: np.ndarray, band: float = 0.06,
                outer_q: float = 0.65) -> List[Tuple[np.ndarray, float]]:
    """The circle each end of a part finishes on, as (centre, radius).

    Parts of an assembly meet on circular faces, so the joint is not really a
    surface-to-surface affair at all — it is two circles that have to become
    one. Reading them off each part turns seating, coaxiality and the step at
    the seam into a single registration with nothing left to trade against
    anything else.

    Each end is taken as a thin band, and within it only the outermost points,
    so an end that is capped is read by its edge rather than by its lid. The
    fit is algebraic (Kasa), which tolerates a rim that is only partly sampled.
    """
    P = np.asarray(points, dtype=np.float64)
    a = normalize(axis)
    h = (P - P.mean(axis=0)) @ a
    lo, hi = np.percentile(h, [1, 99])
    span = max(hi - lo, 1e-9)
    u = normalize(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
    v = np.cross(a, u)
    out = []
    for sel in ((h <= lo + band * span), (h >= hi - band * span)):
        R = P[sel]
        if len(R) < 30:
            out.append((P.mean(axis=0), 0.0))
            continue
        d = R - R.mean(axis=0)
        xy = np.stack([d @ u, d @ v], axis=1)
        keep = np.linalg.norm(xy, axis=1) >= np.quantile(np.linalg.norm(xy, axis=1), outer_q)
        xy = xy[keep]
        A = np.c_[2 * xy, np.ones(len(xy))]
        sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
        rad = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-12)))
        centre = R.mean(axis=0) + sol[0] * u + sol[1] * v
        centre = centre + (float((R @ a).mean()) - float(centre @ a)) * a
        out.append((centre, rad))
    return out          # [bottom, top] along `axis`


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle of the rotation taking R1 to R2, in degrees."""
    c = (float(np.trace(np.asarray(R1).T @ np.asarray(R2))) - 1.0) / 2.0
    return float(np.degrees(math.acos(max(-1.0, min(1.0, c)))))


def geodesic_torch(R: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    """Differentiable angle (radians) between two rotations."""
    c = (torch.einsum("ij,ij->", R, R_ref) - 1.0) / 2.0
    return torch.acos(torch.clamp(c, -1.0 + 1e-6, 1.0 - 1e-6))


def axis_from_name(name: str) -> np.ndarray:
    """'y' -> +Y, '-z' -> -Z, etc."""
    n = name.strip().lower()
    sign = -1.0 if n.startswith("-") else 1.0
    key = n.lstrip("+-")
    base = {"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}.get(key)
    if base is None:
        raise ValueError(f"bad axis name: {name!r}")
    return sign * np.array(base, dtype=np.float64)


# --------------------------------------------------------------------------
# point-cloud helpers
# --------------------------------------------------------------------------

def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Masked depth pixels -> Nx3 camera-frame points."""
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Principal axes (columns, descending eigenvalue) and their extents."""
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((P - c).T))
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    proj = (P - c) @ axes
    extent = np.percentile(proj, 97.5, axis=0) - np.percentile(proj, 2.5, axis=0)
    return axes, extent


def umeyama_rigid(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform (scale fixed at 1) taking src onto tgt."""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    H = (src - mu_s).T @ (tgt - mu_t) / len(src)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    R = R.T  # we want tgt ~ R @ src
    return compose(R, mu_t - R @ mu_s, 1.0)
