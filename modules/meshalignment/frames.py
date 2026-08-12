"""Frame loading and everything derived from a single capture.

A capture is RGB + metric depth + a foreground-mask source + intrinsics + the
mesh reconstructed for that frame. Two directory layouts are accepted:

  nested :  <data_dir>/part_<fid>/{rgb,rgb_masked,depth,intrinsic,mesh}.*
  flat   :  <data_dir>/{rgb,rgb_masked,depth,intrinsic,mesh}_<fid>.*

Both are read directly, so no symlink shuffling is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_fill_holes

from .geom import backproject, normalize

# Depth outside this band is sensor noise, not the scene.
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 6.0


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def _paths(data_dir: Path, fid: str) -> Optional[Dict[str, Path]]:
    """Resolve the five files for `fid` under either layout, or None."""
    nested = data_dir / f"part_{fid}"
    cands = [
        {"rgb": nested / "rgb.png", "masked": nested / "rgb_masked.png",
         "depth": nested / "depth.npy", "K": nested / "intrinsic.npy",
         "mesh": nested / "mesh.glb"},
        {"rgb": data_dir / f"rgb_{fid}.png", "masked": data_dir / f"rgb_masked_{fid}.png",
         "depth": data_dir / f"depth_{fid}.npy", "K": data_dir / f"intrinsic_{fid}.npy",
         "mesh": data_dir / f"mesh_{fid}.glb"},
    ]
    for c in cands:
        if all(p.exists() for p in c.values()):
            return c
    return None


def find_frame_ids(data_dir: Path) -> List[str]:
    """All frame ids with a complete file set, in natural order."""
    ids = set()
    for p in data_dir.glob("part_*"):
        if p.is_dir():
            ids.add(p.name[len("part_"):])
    for p in data_dir.glob("rgb_*.png"):
        m = re.fullmatch(r"rgb_([A-Za-z0-9_-]+)\.png", p.name)
        if m and not m.group(1).startswith("masked_"):
            ids.add(m.group(1))
    found = [f for f in sorted(ids) if _paths(data_dir, f) is not None]
    return sorted(found, key=lambda f: (len(f), f))


# --------------------------------------------------------------------------
# intrinsics
# --------------------------------------------------------------------------

def to_standard_K(K_raw: np.ndarray) -> np.ndarray:
    """Accept a 3x3 K or an hl2ss 4x4 (row-vector, cx/cy at [2,0],[2,1])."""
    K = np.asarray(K_raw, dtype=np.float64)
    if K.shape == (3, 3):
        return K.copy()
    if K.shape != (4, 4):
        raise ValueError(f"unexpected intrinsics shape {K.shape}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    # hl2ss stores the principal point in the third ROW; a plain 4x4 in the
    # third COLUMN. Whichever holds pixel-sized values is the real one.
    if abs(K[2, 0]) > 1.0 and abs(K[2, 1]) > 1.0:
        cx, cy = float(K[2, 0]), float(K[2, 1])
    else:
        cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _close_and_fill(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return binary_fill_holes(
        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0)


def foreground_mask(rgb: np.ndarray, masked_rgb: np.ndarray,
                    depth: Optional[np.ndarray] = None,
                    dark_level: int = 3, keep_ratio: float = 0.5,
                    depth_margin_m: float = 0.03) -> np.ndarray:
    """Boolean foreground, recovering parts of the object that are truly black.

    An alpha channel, when present, is authoritative and used directly.

    Otherwise the masked image is compared against the original photograph.
    Background is not erased but attenuated, so a background pixel keeps a
    small fraction of its original brightness while a foreground pixel is
    passed through untouched. Comparing the two images therefore *decides*
    every pixel bright enough to carry the evidence:

        masked ~= rgb          -> foreground
        masked << rgb          -> background

    A pixel that was already black in the photograph carries no evidence
    either way — attenuating zero yields zero. Thresholding brightness alone
    silently assigns all of those to the background, which removes any genuinely
    black region of the object (and, if that region reaches the silhouette
    boundary, hole-filling cannot put it back). Those undecidable pixels are
    instead resolved by depth: they join the foreground when they sit at the
    object's distance and connect to it. A bright distractor touching the
    object — a hand holding it — stays out regardless of its depth, because
    it was bright enough to be decided, and it was decided against.
    """
    arr = np.asarray(masked_rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return _close_and_fill(_largest_component(arr[..., 3] > 0))

    src = np.asarray(rgb)[..., :3].max(axis=-1).astype(np.int32)
    got = (arr[..., :3] if arr.ndim == 3 else arr[..., None]).max(axis=-1).astype(np.int32)

    decidable = src >= dark_level
    kept = decidable & (got >= keep_ratio * src)
    seed = _largest_component(kept)
    if depth is None or seed.sum() < 50:
        return _close_and_fill(seed)

    # Undecidable pixels: black in the photograph, so attenuation left no trace.
    undecidable = ~decidable
    valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
    seed_depth = depth[seed & valid]
    if seed_depth.size < 50:
        return _close_and_fill(seed)
    lo, hi = np.percentile(seed_depth, [1.0, 99.0])
    at_object = valid & (depth >= lo - depth_margin_m) & (depth <= hi + depth_margin_m)

    # Grow only into undecidable pixels that are both at the object's distance
    # and connected to it, so a same-distance surface elsewhere cannot join.
    grown = _largest_component(seed | (undecidable & at_object))
    return _close_and_fill(grown & (seed | undecidable))


def depth_correspondence_mask(
    depth: np.ndarray, mask: np.ndarray, erode_px: int = 2,
    p_low: float = 1.0, p_high: float = 99.0, margin_m: float = 0.02,
) -> np.ndarray:
    """Mask pixels whose depth is trustworthy enough for correspondence.

    Erodes the silhouette (boundary pixels mix object and background depth)
    and keeps a robust depth band so a stray far/near reading cannot drag the
    fit.
    """
    m = mask.astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        m = cv2.erode(m, k, iterations=1)
    m = m.astype(bool) & np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
    if m.sum() < 50:
        return m
    lo, hi = np.percentile(depth[m], [p_low, p_high])
    return m & (depth >= lo - margin_m) & (depth <= hi + margin_m)


# --------------------------------------------------------------------------
# support plane
# --------------------------------------------------------------------------

@dataclass
class Plane:
    """Support (table) plane: `normal . x + offset = 0`.

    `normal` points away from the table toward visual up, so
    `normal . x + offset > 0` means "above the table".
    """
    normal: np.ndarray
    offset: float
    n_inliers: int
    tilt_deg: float          # angle between normal and camera up
    ok: bool

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.offset


CAMERA_UP = np.array([0.0, -1.0, 0.0])


def estimate_plane(
    depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
    iters: int = 800, thresh_m: float = 0.008, max_points: int = 24000,
    seed: int = 0,
) -> Plane:
    """RANSAC the support plane from non-object pixels around the object.

    The ROI is a *ring* around the object rather than a band below it, so a
    capture whose object touches the bottom of the frame still has plane
    evidence. Both normal and offset come from the same inlier set, so they
    cannot disagree. `ok=False` means there was not enough evidence — the
    caller must decide what to do rather than silently receiving a guess.
    """
    H, W = depth.shape
    ys, xs = np.where(mask)
    bad = Plane(CAMERA_UP.copy(), 0.0, 0, 0.0, False)
    if len(xs) < 20:
        return bad

    # Ring: generous box around the object, minus the dilated object itself.
    pad_x = max(int(0.5 * (xs.max() - xs.min())), 120)
    pad_y = max(int(0.5 * (ys.max() - ys.min())), 120)
    roi = np.zeros_like(mask, dtype=bool)
    roi[max(int(ys.min()) - pad_y, 0):min(int(ys.max()) + pad_y, H - 1) + 1,
        max(int(xs.min()) - pad_x, 0):min(int(xs.max()) + pad_x, W - 1) + 1] = True
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    obj = cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
    valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
    pts = backproject(depth, roi & ~obj & valid, K)
    if len(pts) < 300:
        return bad

    rng = np.random.default_rng(int(seed))
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]

    best = (0, None, 0.0, None)
    for _ in range(int(iters)):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(n) < 1e-10:
            continue
        n = normalize(n)
        d = -float(np.dot(n, tri[0]))
        if np.dot(n, CAMERA_UP) < 0:
            n, d = -n, -d
        inl = np.abs(pts @ n + d) < float(thresh_m)
        if int(inl.sum()) > best[0]:
            best = (int(inl.sum()), n, d, inl)
    if best[1] is None or best[0] < 200:
        return bad

    # Refit normal AND offset on the inliers together.
    inpts = pts[best[3]]
    c = inpts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((inpts - c).T))
    n = normalize(evecs[:, int(np.argmin(evals))])
    if np.dot(n, CAMERA_UP) < 0:
        n = -n
    d = -float(np.dot(n, c))
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(n, CAMERA_UP), -1.0, 1.0))))
    return Plane(n, d, len(inpts), tilt, True)


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------

@dataclass
class Frame:
    fid: str
    rgb: np.ndarray            # HxWx3 uint8
    depth: np.ndarray          # HxW float32, metres
    K: np.ndarray              # 3x3
    mask: np.ndarray           # HxW bool, foreground
    corr: np.ndarray           # HxW bool, depth trustworthy inside foreground
    mesh: trimesh.Trimesh
    plane: Plane

    @property
    def shape(self) -> Tuple[int, int]:
        return self.depth.shape

    def target_points(self) -> np.ndarray:
        return backproject(self.depth, self.corr, self.K)


def load_frame(data_dir: Path, fid: str, *, plane_seed: int = 0) -> Frame:
    p = _paths(Path(data_dir), fid)
    if p is None:
        raise FileNotFoundError(f"frame {fid!r}: incomplete file set under {data_dir}")
    rgb = np.array(Image.open(p["rgb"]).convert("RGB"))
    depth = np.load(p["depth"]).astype(np.float32)
    K = to_standard_K(np.load(p["K"]))
    mask = foreground_mask(rgb, np.array(Image.open(p["masked"])), depth)
    if mask.shape != depth.shape:
        raise ValueError(f"frame {fid!r}: mask {mask.shape} != depth {depth.shape}")
    mesh = trimesh.load(p["mesh"], force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"frame {fid!r}: {p['mesh']} did not load as a mesh")
    return Frame(
        fid=fid, rgb=rgb, depth=depth, K=K, mask=mask,
        corr=depth_correspondence_mask(depth, mask), mesh=mesh,
        plane=estimate_plane(depth, mask, K, seed=plane_seed),
    )


def mask_touches_border(mask: np.ndarray) -> Dict[str, bool]:
    """Which image borders the foreground reaches — i.e. where the object is
    cropped and its silhouette carries no size information."""
    return {"top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
            "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())}
