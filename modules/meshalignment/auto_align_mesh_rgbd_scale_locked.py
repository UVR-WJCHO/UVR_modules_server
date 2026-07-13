#!/usr/bin/env python3
"""
auto_align_mesh_rgbd_scale_locked.py
================================

Scale-locked RGB-D + mesh alignment script for files named:
    rgb_{id}.png
    rgb_masked_{id}.png
    depth_{id}.npy
    mesh_{id}.glb

What this script fixes compared with the earlier attempts:
  1. Uses the PROVIDED rgb_masked image as the object mask source.
     It never expands the mask with a depth-band, so the desk plane is not
     pulled into the target point cloud.
  2. Splits masks into:
       - sil_mask: full object mask for silhouette/RGB scoring.
       - corr_mask: eroded + depth-cleaned mask for depth correspondence.
  3. Uses only z-buffer-visible mesh samples for depth correspondences.
     Hidden/back-side mesh points are not matched to front-surface depth.
  4. Uses only proper rotations, det(R)=+1. No reflection/post flip is saved.
  5. Searches many coarse rotations, runs Umeyama correction, then optional
     PyTorch refinement with a scale prior.
  6. Optionally uses nvdiffrast textured/depth rendering for candidate re-ranking
     when nvdiffrast is installed. The code still runs without it.

Example:
    python auto_align_mesh_rgbd_scale_locked.py \
        --data_dir /mnt/d/metaobj \
        --output_dir /mnt/d/metaobj/results_auto_final \
        --device cuda

CPU quick test:
    python auto_align_mesh_rgbd_scale_locked.py --data_dir ./metaobj --cpu --no_adam --n_spin 12
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.spatial.transform import Rotation

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    NVDIFFRAST_AVAILABLE = False


DEFAULT_HL2_K = np.array(
    [[1011.0, 0.0, 640.0],
     [0.0, 1006.0, 360.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


# ----------------------------- configuration -----------------------------

@dataclass
class AlignConfig:
    n_sample: int = 10000
    n_pair_max: int = 30000
    n_spin: int = 18
    top_candidates: int = 5
    render_top: int = 12
    min_target_points: int = 150
    min_pairs: int = 160
    mask_threshold: int = 5
    erode_px: int = 2
    depth_percentile_low: float = 1.0
    depth_percentile_high: float = 99.0
    depth_margin_m: float = 0.02
    depth_inlier_m: float = 0.03
    scale_mode: str = "per_frame"  # auto | global | per_frame
    scale_cov_threshold: float = 0.12
    scale_source: str = "bbox2d"   # bbox2d | bbox | pca | mixed
    scale_lock: bool = True         # keep physical scale fixed after initialization
    scale_refine_limit: float = 0.03 # if unlocked, clamp Adam scale changes to +/- this fraction
    bbox_scale_refine: bool = True  # final fast projected-bbox scale correction
    adam_iters: int = 120
    adam_lr: float = 0.006
    no_adam: bool = False
    seed: int = 7
    # Upright/table support prior. For tabletop objects, this prevents depth loss
    # from pitching a symmetric/slender mesh backward or forward.
    upright_mode: str = "table"      # off | camera | table
    upright_axis: str = "y"          # local mesh up axis: x,y,z,-x,-y,-z
    upright_weight: float = 2.0       # score penalty/bonus weight
    upright_hard: bool = True         # project each candidate back to the upright cone
    table_ransac_iters: int = 800
    table_plane_thresh_m: float = 0.008


@dataclass
class PoseMetrics:
    score: float
    f1: float
    depth_inlier: float
    depth_median_abs_m: float
    area_ratio: float
    scale: float
    det_R: float
    pairs: int = 0
    rgb_err: Optional[float] = None
    render_iou: Optional[float] = None
    render_depth_err_m: Optional[float] = None
    upright_angle_deg: Optional[float] = None


@dataclass
class Candidate:
    T: np.ndarray
    R_init: np.ndarray
    metrics: PoseMetrics
    name: str


# ----------------------------- IO helpers --------------------------------

def fid_to_seed(fid) -> int:
    """Stable integer seed from any frame ID (int or str).

    Plain decimal strings ('0', '34') become that integer for backwards
    compatibility (so cached seeds in old result folders stay valid).
    Anything else ('4-0', '0_check', '01') hashes to a stable integer.
    """
    try:
        return int(fid)
    except (ValueError, TypeError):
        return abs(hash(str(fid))) & 0xFFFFFF


def find_frames(data_dir: Path) -> List[str]:
    """Discover rgb_{fid}.{png,jpg} groups where fid is any identifier
    matching [A-Za-z0-9_-]+ (e.g. '0', '01', '34', '4-0', '0_check').
    Returns string IDs to preserve leading zeros and special chars.
    """
    pat = re.compile(r"^rgb_([A-Za-z0-9_-]+)\.png$")
    ids = []
    for p in sorted(data_dir.glob("rgb_*.png")):
        m = pat.match(p.name)
        if not m:
            continue
        fid = m.group(1)
        # Exclude rgb_masked_* matches (they also begin with rgb_).
        if fid.startswith("masked_"):
            continue
        required = [
            data_dir / f"rgb_masked_{fid}.png",
            data_dir / f"depth_{fid}.npy",
            data_dir / f"mesh_{fid}.glb",
        ]
        if all(x.exists() for x in required):
            ids.append(fid)
    return ids


def load_frame(data_dir: Path, fid: int):
    rgb = np.array(Image.open(data_dir / f"rgb_{fid}.png").convert("RGB"))
    masked = np.array(Image.open(data_dir / f"rgb_masked_{fid}.png"))
    depth = np.load(data_dir / f"depth_{fid}.npy").astype(np.float32)
    mesh = trimesh.load(data_dir / f"mesh_{fid}.glb", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"mesh_{fid}.glb could not be loaded as a Trimesh")
    return rgb, masked, depth, mesh


def K_to_standard(K_in: np.ndarray) -> np.ndarray:
    """Convert a 4x4 hl2ss intrinsics matrix to a 3x3 standard K.

    hl2ss saves intrinsics in row-vector right-multiply form:
        [[fx, 0,  0, 0],
         [0,  fy, 0, 0],
         [cx, cy, 1, 0],
         [0,  0,  0, 1]]
    Standard convention is:
        [[fx, 0,  cx],
         [0,  fy, cy],
         [0,  0,  1]]
    Detects the layout by checking where cx/cy live; falls through unchanged
    for an already-standard 3x3.
    """
    K_in = np.asarray(K_in, dtype=np.float64)
    if K_in.shape == (3, 3):
        return K_in.copy()
    if K_in.shape != (4, 4):
        raise ValueError(f"intrinsics has unexpected shape {K_in.shape}")
    # hl2ss layout: cx at [2,0], cy at [2,1], fx at [0,0], fy at [1,1]
    fx = float(K_in[0, 0])
    fy = float(K_in[1, 1])
    if abs(K_in[2, 0]) > 1.0 and abs(K_in[2, 1]) > 1.0:
        cx = float(K_in[2, 0])
        cy = float(K_in[2, 1])
    else:
        # Maybe stored as standard 4x4 with cx, cy at [0,2], [1,2]
        cx = float(K_in[0, 2])
        cy = float(K_in[1, 2])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_saved_K(data_dir: Path, fid) -> Optional[np.ndarray]:
    """Look for intrinsic_{fid}.npy (hl2ss layout) in data_dir and return
    the standard 3x3 K. Returns None if not found.
    """
    p = data_dir / f"intrinsic_{fid}.npy"
    if not p.exists():
        return None
    try:
        K_raw = np.load(p)
    except Exception:
        return None
    return K_to_standard(K_raw)


def parse_K(args, H: int, W: int) -> np.ndarray:
    if args.K is not None:
        vals = [float(x) for x in args.K.split(",")]
        if len(vals) != 9:
            raise ValueError("--K must contain 9 comma-separated numbers")
        K = np.array(vals, dtype=np.float64).reshape(3, 3)
    elif args.K_json is not None:
        obj = json.loads(Path(args.K_json).read_text())
        if isinstance(obj, dict) and "K" in obj:
            obj = obj["K"]
        K = np.array(obj, dtype=np.float64).reshape(3, 3)
    else:
        K = DEFAULT_HL2_K.copy()

    # If the image is not 1280x720, scale the default K to the actual image.
    if args.K is None and args.K_json is None and (W != 1280 or H != 720):
        sx, sy = W / 1280.0, H / 720.0
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return K


# ----------------------------- masks / point clouds ------------------------

def make_mask_from_provided(masked_rgb: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Use rgb_masked_{id}.png as the foreground source.

    Do NOT use a depth band to expand the mask; otherwise table/desk pixels get
    included. A small morphological close + hole fill recovers black texture
    holes inside the object.
    """
    arr = np.asarray(masked_rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        raw = (arr[..., 3] > 0).astype(np.uint8)
    elif arr.ndim == 3:
        raw = (arr[..., :3].max(axis=-1) > threshold).astype(np.uint8)
    else:
        raw = (arr > threshold).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    else:
        mask = raw

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = binary_fill_holes(mask > 0).astype(np.uint8)
    return mask.astype(bool)


def make_depth_corr_mask(
    depth: np.ndarray,
    object_mask: np.ndarray,
    erode_px: int = 2,
    p_low: float = 1.0,
    p_high: float = 99.0,
    margin_m: float = 0.02,
) -> np.ndarray:
    """Mask for depth correspondence.

    It is strictly a subset of object_mask. This prevents the desk/table surface
    from entering target depth points. Erosion removes boundary mixed pixels.
    """
    valid = np.isfinite(depth) & (depth > 0)
    m = object_mask.astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k, iterations=1)
    m = m.astype(bool)
    vals = depth[m & valid]
    if vals.size < 50:
        return m & valid
    lo, hi = np.percentile(vals, [p_low, p_high])
    return m & valid & (depth >= lo - margin_m) & (depth <= hi + margin_m)


# ----------------------------- upright/table helpers ----------------------

def local_axis_from_name(name: str) -> np.ndarray:
    """Return a signed local mesh axis vector from x/y/z/-x/-y/-z."""
    key = str(name).strip().lower()
    sign = -1.0 if key.startswith("-") else 1.0
    key = key[1:] if key.startswith("-") else key
    if key not in ("x", "y", "z"):
        raise ValueError(f"upright_axis must be one of x,y,z,-x,-y,-z, got {name!r}")
    v = np.zeros(3, dtype=np.float64)
    v[{"x": 0, "y": 1, "z": 2}[key]] = sign
    return v


def estimate_table_normal_from_depth(
    depth: np.ndarray,
    sil_mask: np.ndarray,
    K: np.ndarray,
    iters: int = 800,
    thresh_m: float = 0.008,
    max_points: int = 24000,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Estimate local table/support plane normal near the object.

    The returned normal is oriented toward visual image-up, i.e. close to
    camera [0,-1,0]. It uses only pixels outside a dilated object mask and a
    local ROI around/below the object, so object pixels are not used.
    """
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 6.0)
    ys, xs = np.where(sil_mask)
    dbg: Dict = {"ok": False}
    if len(xs) < 20:
        return None, dbg

    x0 = max(int(xs.min()) - 220, 0)
    x1 = min(int(xs.max()) + 220, W - 1)
    y0 = max(int(ys.min()) - 30, 0)
    y1 = min(int(ys.max()) + 260, H - 1)
    roi = np.zeros_like(sil_mask, dtype=bool)
    roi[y0:y1 + 1, x0:x1 + 1] = True

    # Exclude the object and immediate boundary, then bias toward lower/table pixels.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    obj_dil = cv2.dilate(sil_mask.astype(np.uint8), k, iterations=1).astype(bool)
    yy = np.indices(depth.shape)[0]
    obj_h = max(float(ys.max() - ys.min()), 1.0)
    lower_bias = yy > (float(ys.min()) + 0.30 * obj_h)
    m = roi & (~obj_dil) & lower_bias & valid

    pts = backproject(depth, m, K)
    dbg.update({"roi": [x0, y0, x1, y1], "plane_pixels": int(m.sum()), "plane_points": int(len(pts))})
    if len(pts) < 300:
        return None, dbg

    rng = np.random.default_rng(int(seed))
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), int(max_points), replace=False)]

    camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    best = None
    # RANSAC plane: n.x + d = 0
    for _ in range(int(iters)):
        ids = rng.choice(len(pts), 3, replace=False)
        tri = pts[ids]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n = n / nn
        d = -float(np.dot(n, tri[0]))
        if np.dot(n, camera_up) < 0.0:
            n = -n
            d = -d
        dist = np.abs(pts @ n + d)
        inliers = dist < float(thresh_m)
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, n, d, inliers)

    if best is None or best[0] < 200:
        return None, dbg

    inpts = pts[best[3]]
    c = inpts.mean(axis=0)
    cov = np.cov((inpts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, int(np.argmin(evals))]
    n = n / max(np.linalg.norm(n), 1e-12)
    if np.dot(n, camera_up) < 0.0:
        n = -n
    angle = float(np.degrees(np.arccos(np.clip(np.dot(n, camera_up), -1.0, 1.0))))
    dbg.update({
        "ok": True,
        "normal": n.tolist(),
        "inliers": int(len(inpts)),
        "angle_to_camera_up_deg": angle,
    })
    return n.astype(np.float64), dbg


def get_upright_normal(
    mode: str,
    depth: np.ndarray,
    sil_mask: np.ndarray,
    K: np.ndarray,
    cfg: AlignConfig,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], Dict]:
    mode = str(mode).lower()
    if mode == "off":
        return None, {"mode": "off", "ok": False}
    if mode == "camera":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64), {"mode": "camera", "ok": True, "normal": [0.0, -1.0, 0.0]}
    if mode == "table":
        n, dbg = estimate_table_normal_from_depth(
            depth, sil_mask, K,
            iters=cfg.table_ransac_iters,
            thresh_m=cfg.table_plane_thresh_m,
            seed=seed,
        )
        dbg["mode"] = "table"
        if n is None:
            # Safe fallback: still prevent severe pitch drift.
            return np.array([0.0, -1.0, 0.0], dtype=np.float64), {**dbg, "fallback": "camera_up"}
        return n, dbg
    raise ValueError(f"upright_mode must be off,camera,table, got {mode!r}")


def upright_angle_deg(R: np.ndarray, local_up: np.ndarray, normal: Optional[np.ndarray]) -> Optional[float]:
    if normal is None:
        return None
    u = R @ local_up
    u = u / max(np.linalg.norm(u), 1e-12)
    n = normal / max(np.linalg.norm(normal), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, n), -1.0, 1.0))))


def upright_project_T(
    T: np.ndarray,
    src_centroid: np.ndarray,
    local_up: np.ndarray,
    normal: Optional[np.ndarray],
) -> np.ndarray:
    """Hard-project a Sim3 pose so R @ local_up exactly equals normal.

    Translation is adjusted to keep the transformed source centroid fixed.
    """
    if normal is None:
        return T
    R, t, s, det_R = decompose_similarity(T)
    if det_R <= 0 or s <= 0:
        return T
    u = R @ local_up
    u = u / max(np.linalg.norm(u), 1e-12)
    n = normal / max(np.linalg.norm(normal), 1e-12)
    Q = rotation_align_vectors(u, n)
    R2 = Q @ R
    if np.linalg.det(R2) <= 0:
        return T
    cam_centroid = s * (R @ src_centroid) + t
    t2 = cam_centroid - s * (R2 @ src_centroid)
    return compose_similarity(R2, t2, s)


def refit_translation_from_visible_depth(
    T: np.ndarray,
    src_pts: np.ndarray,
    depth: np.ndarray,
    corr_mask: np.ndarray,
    K: np.ndarray,
    min_pairs: int = 120,
    max_iter: int = 2,
) -> np.ndarray:
    """After an upright projection, recover the best translation by median residual."""
    Tout = T.copy()
    for _ in range(int(max_iter)):
        src_cam = apply_T(src_pts, Tout)
        sp, tp = visible_projected_pairs(src_cam, depth, corr_mask, K, min_pairs=min_pairs, max_pairs=20000)
        if sp is None or len(sp) < min_pairs:
            break
        delta = np.median(tp - sp, axis=0)
        if not np.isfinite(delta).all() or np.linalg.norm(delta) > 0.20:
            break
        Tout[:3, 3] += delta
    return Tout


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return c, v[:, order], w[order]


def estimate_scale_pca(source_pts: np.ndarray, target_pts: np.ndarray) -> float:
    _, _, ws = pca_frame(source_pts)
    _, _, wt = pca_frame(target_pts)
    return float(np.sqrt(max(float(wt.sum()), 1e-12) / max(float(ws.sum()), 1e-12)))


def estimate_scale_bbox(object_mask: np.ndarray, depth: np.ndarray, K: np.ndarray, mesh: trimesh.Trimesh) -> Optional[float]:
    valid = object_mask & np.isfinite(depth) & (depth > 0)
    if int(object_mask.sum()) < 50 or int(valid.sum()) < 50:
        return None
    z_med = float(np.median(depth[valid]))
    ys, xs = np.where(object_mask)
    pix_w = float(np.percentile(xs, 97.5) - np.percentile(xs, 2.5))
    pix_h = float(np.percentile(ys, 97.5) - np.percentile(ys, 2.5))
    phys_w = pix_w * z_med / float(K[0, 0])
    phys_h = pix_h * z_med / float(K[1, 1])
    mesh_max = float(np.max(mesh.extents))
    if mesh_max <= 1e-9:
        return None
    return max(phys_w, phys_h) / mesh_max



def estimate_scale_bbox2d_for_R(
    model_pts: np.ndarray,
    R: np.ndarray,
    object_mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    q_model: Tuple[float, float] = (0.5, 99.5),
    q_mask: Tuple[float, float] = (1.0, 99.0),
) -> Optional[float]:
    """Estimate metric scale for a specific rotation from the 2D mask bbox.

    The previous code used a single mesh.extents.max scale estimate before yaw
    was known. That is often wrong for tapered/cylindrical/legged objects.
    For a candidate rotation R, this function compares the observed mask width
    and height in meters at the object's median depth against the rotated mesh
    extent along camera X and camera Y.
    """
    valid = object_mask & np.isfinite(depth) & (depth > 0.05)
    if int(valid.sum()) < 50 or int(object_mask.sum()) < 50:
        return None

    ys, xs = np.where(object_mask)
    xlo, xhi = np.percentile(xs.astype(np.float64), q_mask)
    ylo, yhi = np.percentile(ys.astype(np.float64), q_mask)
    pix_w = float(xhi - xlo)
    pix_h = float(yhi - ylo)
    if pix_w < 5 or pix_h < 5:
        return None

    z_med = float(np.median(depth[valid]))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    target_w_m = pix_w * z_med / max(fx, 1e-12)
    target_h_m = pix_h * z_med / max(fy, 1e-12)

    pts = np.asarray(model_pts, dtype=np.float64)
    if len(pts) < 8:
        return None
    pr = (np.asarray(R, dtype=np.float64) @ pts.T).T
    x0, x1 = np.percentile(pr[:, 0], q_model)
    y0, y1 = np.percentile(pr[:, 1], q_model)
    model_w = float(x1 - x0)
    model_h = float(y1 - y0)

    vals = []
    weights = []
    if model_w > 1e-9 and target_w_m > 1e-9:
        vals.append(target_w_m / model_w)
        weights.append(max(pix_w, 1.0))
    if model_h > 1e-9 and target_h_m > 1e-9:
        vals.append(target_h_m / model_h)
        # height usually constrains scale best for these upright objects.
        weights.append(max(pix_h, 1.0) * 1.25)
    if not vals:
        return None

    vals = np.asarray(vals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    # Weighted geometric mean is stable when width and height slightly disagree.
    log_s = float(np.sum(weights * np.log(np.maximum(vals, 1e-12))) / np.sum(weights))
    s = math.exp(log_s)
    if not np.isfinite(s) or s <= 0:
        return None
    return float(s)


def mask_bbox_pixels(mask: np.ndarray, q: Tuple[float, float] = (1.0, 99.0)) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < 20:
        return None
    x0, x1 = np.percentile(xs.astype(np.float64), q)
    y0, y1 = np.percentile(ys.astype(np.float64), q)
    return float(x0), float(y0), float(x1), float(y1)


def projected_bbox_pixels(
    model_pts: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    q: Tuple[float, float] = (0.5, 99.5),
) -> Optional[Tuple[float, float, float, float]]:
    pts_cam = apply_T(model_pts, T)
    u, v, ok = project_points(pts_cam, K)
    ok = ok & np.isfinite(u) & np.isfinite(v)
    # Do not require in-image for bbox; a slightly oversized candidate should
    # still be measurable. But reject wild projections.
    ok = ok & (u > -W) & (u < 2 * W) & (v > -H) & (v < 2 * H)
    if int(ok.sum()) < 8:
        return None
    x0, x1 = np.percentile(u[ok], q)
    y0, y1 = np.percentile(v[ok], q)
    return float(x0), float(y0), float(x1), float(y1)


def rescale_T_around_source_centroid(T: np.ndarray, src_centroid: np.ndarray, scale_mul: float) -> np.ndarray:
    R, t, s, det_R = decompose_similarity(T)
    if det_R <= 0 or s <= 0 or not np.isfinite(scale_mul) or scale_mul <= 0:
        return T
    cam_centroid = s * (R @ src_centroid) + t
    s2 = s * float(scale_mul)
    t2 = cam_centroid - s2 * (R @ src_centroid)
    return compose_similarity(R, t2, s2)


def refine_scale_to_mask_bbox(
    T: np.ndarray,
    model_pts: np.ndarray,
    src_centroid: np.ndarray,
    sil_mask: np.ndarray,
    K: np.ndarray,
    n_iter: int = 2,
    max_step: float = 0.18,
) -> np.ndarray:
    """Fast post-scale correction using projected mesh bbox vs provided mask bbox.

    This adjusts scale only from the provided mask silhouette, not from depth.
    It prevents depth residuals from making the mesh too large/small.
    """
    H, W = sil_mask.shape
    mb = mask_bbox_pixels(sil_mask)
    if mb is None:
        return T
    mx0, my0, mx1, my1 = mb
    tw = max(mx1 - mx0, 1.0)
    th = max(my1 - my0, 1.0)
    Tout = T.copy()
    for _ in range(int(n_iter)):
        pb = projected_bbox_pixels(model_pts, Tout, K, H, W)
        if pb is None:
            break
        px0, py0, px1, py1 = pb
        rw = max(px1 - px0, 1.0)
        rh = max(py1 - py0, 1.0)
        vals = np.array([tw / rw, th / rh], dtype=np.float64)
        weights = np.array([tw, th * 1.25], dtype=np.float64)
        mul = math.exp(float(np.sum(weights * np.log(np.maximum(vals, 1e-12))) / np.sum(weights)))
        mul = float(np.clip(mul, 1.0 - max_step, 1.0 + max_step))
        if not np.isfinite(mul) or abs(math.log(mul)) < 0.003:
            break
        Tout = rescale_T_around_source_centroid(Tout, src_centroid, mul)
    return Tout

def robust_median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no valid scale values")
    if arr.size < 4:
        return float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    kept = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    return float(np.median(kept if kept.size else arr))


def choose_frame_scale(s_pca: float, s_bbox: Optional[float]) -> float:
    if s_bbox is None or not np.isfinite(s_bbox) or s_bbox <= 0:
        return float(s_pca)
    if not np.isfinite(s_pca) or s_pca <= 0:
        return float(s_bbox)
    # If the two estimates are close, use their median. If they disagree a lot,
    # bbox is usually safer for visible-only depth clouds.
    ratio = max(s_pca, s_bbox) / max(min(s_pca, s_bbox), 1e-12)
    if ratio < 1.35:
        return float(np.median([s_pca, s_bbox]))
    return float(s_bbox)


# ----------------------------- transforms --------------------------------

def apply_T(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    return (T @ h.T).T[:, :3]


def compose_similarity(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = float(s) * np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def decompose_similarity(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    A = np.asarray(T[:3, :3], dtype=np.float64)
    detA = float(np.linalg.det(A))
    if detA <= 0:
        return np.eye(3), np.asarray(T[:3, 3], dtype=np.float64), 0.0, detA
    s = float(np.cbrt(detA))
    R_raw = A / max(s, 1e-12)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, np.asarray(T[:3, 3], dtype=np.float64), s, float(np.linalg.det(R))


def umeyama(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Similarity T mapping src -> tgt, with proper rotation."""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    n, d = src.shape
    ms = src.mean(axis=0)
    mt = tgt.mean(axis=0)
    sc = src - ms
    tc = tgt - mt
    var_src = float((sc * sc).sum() / max(n, 1))
    cov = tc.T @ sc / max(n, 1)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    t = mt - s * (R @ ms)
    return compose_similarity(R, t, s), s



def umeyama_rigid(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rigid correction T mapping src -> tgt, with scale fixed to 1.

    This is the important scale fix. The initial scale is estimated from the
    2D object mask and current rotation; the depth correspondence stage should
    only correct rotation/translation. A full Sim(3) Umeyama can shrink or grow
    the mesh to fit noisy visible-depth samples, which is what caused the
    over/under-sized overlays.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    ms = src.mean(axis=0)
    mt = tgt.mean(axis=0)
    sc = src - ms
    tc = tgt - mt
    cov = tc.T @ sc / max(len(src), 1)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    t = mt - R @ ms
    return compose_similarity(R, t, 1.0), 1.0

def rotation_align_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    a = np.asarray(from_vec, dtype=np.float64)
    b = np.asarray(to_vec, dtype=np.float64)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-10:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis /= max(np.linalg.norm(axis), 1e-12)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / max(s * s, 1e-12))


def rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    return Rotation.from_rotvec(theta * axis).as_matrix()


def unique_rotation_key(R: np.ndarray, decimals: int = 4) -> Tuple[float, ...]:
    return tuple(np.round(R.reshape(-1), decimals=decimals).tolist())


def generate_rotation_candidates(
    src_pts: np.ndarray,
    target_pts: np.ndarray,
    n_spin: int = 24,
    upright_normal: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Generate many proper rotations.

    We align plausible mesh long/up axes to either the target 3D PCA axis or
    camera up/down, then spin around that target axis. This avoids reflection
    flips while covering the 180/axis ambiguities.
    """
    _, src_axes, _ = pca_frame(src_pts)
    _, tgt_axes, _ = pca_frame(target_pts)

    canonical = [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]
    local_axes: List[Tuple[str, np.ndarray]] = []
    for i in range(3):
        local_axes.append((f"pca{i}+", src_axes[:, i]))
        local_axes.append((f"pca{i}-", -src_axes[:, i]))
    for name, ax in canonical:
        local_axes.append((f"{name}+", ax))
        local_axes.append((f"{name}-", -ax))

    # OpenCV camera: +y image-down, so visual up is camera -Y.
    camera_up = np.array([0.0, -1.0, 0.0])
    target_axes: List[Tuple[str, np.ndarray]] = [
        ("tgt_pca0+", tgt_axes[:, 0]),
        ("tgt_pca0-", -tgt_axes[:, 0]),
        ("cam_up", camera_up),
        ("cam_down", -camera_up),
    ]
    if upright_normal is not None:
        un = np.asarray(upright_normal, dtype=np.float64)
        un /= max(np.linalg.norm(un), 1e-12)
        target_axes.insert(0, ("upright", un))

    seen = set()
    out: List[Tuple[str, np.ndarray]] = []
    spin_angles = np.linspace(0.0, 2.0 * np.pi, int(n_spin), endpoint=False)
    for lname, laxis in local_axes:
        for tname, taxis in target_axes:
            R0 = rotation_align_vectors(laxis, taxis)
            for j, th in enumerate(spin_angles):
                Rspin = rotation_about_axis(taxis, float(th))
                R = Rspin @ R0
                if np.linalg.det(R) <= 0:
                    continue
                key = unique_rotation_key(R, decimals=3)
                if key in seen:
                    continue
                seen.add(key)
                out.append((f"{lname}->{tname}/spin{j:02d}", R))
    return out


# ----------------------------- projection / visible pairs ------------------

def project_points(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]
    ok = np.isfinite(z) & (z > 1e-6)
    u = np.full(len(pts), np.nan, dtype=np.float64)
    v = np.full(len(pts), np.nan, dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u[ok] = fx * pts[ok, 0] / z[ok] + cx
    v[ok] = fy * pts[ok, 1] / z[ok] + cy
    return u, v, ok


def zbuffer_visible_indices(points_cam: np.ndarray, K: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, ok = project_points(points_cam, K)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inb = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ids = np.where(inb)[0]
    if ids.size == 0:
        return ids, ui, vi
    keys = vi[ids].astype(np.int64) * int(W) + ui[ids].astype(np.int64)
    zvals = points_cam[ids, 2]
    order = np.lexsort((zvals, keys))
    keys_sorted = keys[order]
    first = np.r_[True, keys_sorted[1:] != keys_sorted[:-1]]
    visible = ids[order][first]
    return visible, ui, vi


def visible_projected_pairs(
    source_cam: np.ndarray,
    depth: np.ndarray,
    corr_mask: np.ndarray,
    K: np.ndarray,
    min_pairs: int = 160,
    max_pairs: int = 30000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    H, W = depth.shape
    vis_idx, ui, vi = zbuffer_visible_indices(source_cam, K, H, W)
    if vis_idx.size < min_pairs:
        return None, None

    us = ui[vis_idx]
    vs = vi[vis_idx]
    zt = depth[vs, us]
    good = corr_mask[vs, us] & np.isfinite(zt) & (zt > 0)
    if int(good.sum()) < min_pairs:
        return None, None

    ids = vis_idx[good]
    us = us[good].astype(np.float64)
    vs = vs[good].astype(np.float64)
    zt = zt[good].astype(np.float64)

    # Robust z-ratio filter before Umeyama.
    zs = source_cam[ids, 2].astype(np.float64)
    ratio = zt / np.maximum(zs, 1e-12)
    qlo, qhi = np.percentile(ratio, [5.0, 95.0])
    keep = (ratio >= qlo) & (ratio <= qhi)
    if int(keep.sum()) < min_pairs:
        return None, None
    ids = ids[keep]
    us = us[keep]
    vs = vs[keep]
    zt = zt[keep]

    if ids.size > max_pairs:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(ids.size, max_pairs, replace=False)
        ids = ids[sel]
        us = us[sel]
        vs = vs[sel]
        zt = zt[sel]

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    tgt = np.stack([(us - cx) * zt / fx, (vs - cy) * zt / fy, zt], axis=1)
    src = source_cam[ids]
    return src.astype(np.float64), tgt.astype(np.float64)


def f1_from_projection(point_mask: np.ndarray, target_mask: np.ndarray, dilate_px: int = 5) -> Tuple[float, float]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    pd = cv2.dilate(point_mask.astype(np.uint8), k, iterations=1).astype(bool)
    md = cv2.dilate(target_mask.astype(np.uint8), k, iterations=1).astype(bool)
    tp = float((pd & target_mask).sum())
    fp = float((pd & ~target_mask).sum())
    fn = float((~pd & md).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    area_ratio = float(pd.sum()) / max(float(target_mask.sum()), 1.0)
    return float(f1), area_ratio


def score_pose_points(
    src_pts: np.ndarray,
    T: np.ndarray,
    depth: np.ndarray,
    corr_mask: np.ndarray,
    sil_mask: np.ndarray,
    K: np.ndarray,
    depth_inlier_m: float = 0.03,
    local_up: Optional[np.ndarray] = None,
    upright_normal: Optional[np.ndarray] = None,
    upright_weight: float = 0.0,
) -> PoseMetrics:
    H, W = depth.shape
    R, _, s, det_R = decompose_similarity(T)
    if s <= 0 or det_R <= 0:
        return PoseMetrics(-1e9, 0, 0, 1.0, 0, 0, det_R)

    pts_cam = apply_T(src_pts, T)
    vis_idx, ui, vi = zbuffer_visible_indices(pts_cam, K, H, W)
    if vis_idx.size == 0:
        return PoseMetrics(-1e9, 0, 0, 1.0, 0, s, det_R)

    proj = np.zeros((H, W), dtype=np.uint8)
    proj[vi[vis_idx], ui[vis_idx]] = 1
    f1, area_ratio = f1_from_projection(proj, sil_mask, dilate_px=5)

    us = ui[vis_idx]
    vs = vi[vis_idx]
    dz = depth[vs, us]
    good = corr_mask[vs, us] & np.isfinite(dz) & (dz > 0)
    if int(good.sum()) > 20:
        err = np.abs(pts_cam[vis_idx[good], 2] - dz[good])
        med_err = float(np.median(err))
        inlier = float((err < depth_inlier_m).mean())
    else:
        med_err = 1.0
        inlier = 0.0

    # Scale-sensitive area penalty. Scale is now fixed, but this still helps
    # reject candidates whose projected footprint is visibly too large/small.
    area_pen = abs(math.log(max(area_ratio, 1e-4)))
    score = 1.5 * f1 + 1.0 * inlier - 4.0 * min(med_err, 0.25) - 0.65 * area_pen

    up_ang = None
    if local_up is not None and upright_normal is not None and upright_weight > 0:
        up_ang = upright_angle_deg(R, local_up, upright_normal)
        # Penalize pitch/roll away from the support normal. A 30Â° error costs
        # roughly upright_weight, enough to reject the backward-leaning solution
        # while still allowing depth/RGB to decide yaw.
        score -= float(upright_weight) * (min(float(up_ang), 90.0) / 30.0) ** 2

    return PoseMetrics(
        score=float(score),
        f1=float(f1),
        depth_inlier=float(inlier),
        depth_median_abs_m=float(med_err),
        area_ratio=float(area_ratio),
        scale=float(s),
        det_R=float(det_R),
        pairs=int(good.sum()),
        upright_angle_deg=up_ang,
    )


# ----------------------------- optional nvdiffrast renderer ----------------

def make_projection_matrix_torch(fx, fy, cx, cy, W, H, near=0.05, far=10.0, device="cuda"):
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / W
    P[0, 2] = -(1.0 - 2.0 * cx / W)
    P[1, 1] = -2.0 * fy / H
    P[1, 2] = 1.0 - 2.0 * cy / H
    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = 1.0
    return P


class NVTexturedRenderer:
    def __init__(self, mesh: trimesh.Trimesh, K: np.ndarray, H: int, W: int, device: str = "cuda"):
        if not NVDIFFRAST_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError("nvdiffrast/torch not available")
        if getattr(mesh.visual, "uv", None) is None:
            raise RuntimeError("mesh has no UV coordinates")
        tex_obj = getattr(mesh.visual.material, "baseColorTexture", None)
        if tex_obj is None:
            raise RuntimeError("mesh has no baseColorTexture")
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.verts = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
        faces_np = np.asarray(mesh.faces, dtype=np.int32)
        self.faces = torch.tensor(faces_np, dtype=torch.int32, device=device)
        # Reversed winding for poses whose 3x3 has negative determinant
        # (reflection from axis flips); without this nvdiffrast culls the
        # visible side and the textured render comes out inside-out.
        self.faces_rev = torch.tensor(np.ascontiguousarray(faces_np[:, ::-1]),
                                       dtype=torch.int32, device=device)
        self.uv = torch.tensor(np.asarray(mesh.visual.uv, dtype=np.float32), dtype=torch.float32, device=device)
        tex_img = np.asarray(tex_obj).astype(np.float32) / 255.0
        if tex_img.ndim == 2:
            tex_img = np.stack([tex_img, tex_img, tex_img], axis=-1)
        if tex_img.shape[-1] == 4:
            tex_img = tex_img[..., :3]
        tex_img = np.ascontiguousarray(tex_img[::-1])
        self.tex = torch.tensor(tex_img, dtype=torch.float32, device=device)
        self.P = make_projection_matrix_torch(
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
            W, H, device=device,
        )
        self.ctx = dr.RasterizeCudaContext(device=device)

    def render_T(self, T_np: np.ndarray):
        T = torch.tensor(T_np, dtype=torch.float32, device=self.device)
        # Pick face winding based on the pose's determinant. A pose with
        # det<0 (reflection) needs reversed winding so the front face is the
        # visible one in image space.
        det_T = float(np.linalg.det(T_np[:3, :3]))
        faces = self.faces_rev if det_T < 0 else self.faces
        vh = torch.cat([self.verts, torch.ones_like(self.verts[:, :1])], dim=1)
        vcam = (vh @ T.T)[:, :3]
        vh2 = torch.cat([vcam, torch.ones_like(vcam[:, :1])], dim=1)
        vclip = (vh2 @ self.P.T).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(self.ctx, vclip, faces, resolution=[self.H, self.W])
        rast_c = rast.contiguous()
        sil_hard = (rast[..., 3:4] > 0).float()
        sil = dr.antialias(sil_hard, rast, vclip, faces).squeeze(0).squeeze(-1)
        z_attr = vcam[:, 2:3].unsqueeze(0).contiguous()
        zr, _ = dr.interpolate(z_attr, rast_c, faces)
        zr = zr.squeeze(0).squeeze(-1)
        uv_attr = self.uv.unsqueeze(0).contiguous()
        uv_interp, _ = dr.interpolate(uv_attr, rast_c, faces)
        rgb = dr.texture(self.tex[None], uv_interp.contiguous(), filter_mode="linear").squeeze(0)
        # nvdiffrast outputs OpenGL convention (row 0 = bottom of NDC).
        # Flip vertically so the result is in image convention (row 0 = top),
        # matching depth/mask tensors and the saved RGB image.
        sil = torch.flip(sil, dims=[0])
        zr = torch.flip(zr, dims=[0])
        rgb = torch.flip(rgb, dims=[0])
        return sil, zr, rgb


def nv_metrics(
    renderer: NVTexturedRenderer,
    T: np.ndarray,
    sil_mask: np.ndarray,
    corr_mask: np.ndarray,
    depth: np.ndarray,
    target_rgb: np.ndarray,
) -> Tuple[float, float, float]:
    target_mask_t = torch.tensor(sil_mask.astype(np.bool_), device=renderer.device)
    corr_mask_t = torch.tensor(corr_mask.astype(np.bool_), device=renderer.device)
    depth_t = torch.tensor(depth.astype(np.float32), device=renderer.device)
    rgb_t = torch.tensor(target_rgb.astype(np.float32) / 255.0, dtype=torch.float32, device=renderer.device)
    with torch.no_grad():
        sil, zr, rgb = renderer.render_T(T)
        mr = sil > 0.5
        mt = target_mask_t
        inter = (mr & mt).sum().float()
        union = (mr | mt).sum().float().clamp(min=1.0)
        iou = float((inter / union).item())
        both = mr & corr_mask_t & (depth_t > 0.05)
        if int(both.sum().item()) > 80:
            d_err = float((zr[both] - depth_t[both]).abs().median().item())
            rgb_err = float((rgb[both] - rgb_t[both]).abs().mean().item())
        else:
            d_err = 1.0
            rgb_err = 1.0
    return iou, d_err, rgb_err


# ----------------------------- PyTorch optimizer ---------------------------

if TORCH_AVAILABLE:
    def quat_to_rot_torch(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ]).reshape(3, 3)

    def rot_to_quat_np(R: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]].astype(np.float64)

    def make_grid_torch(u, v, H, W):
        return torch.stack([2.0 * u / (W - 1) - 1.0,
                            2.0 * v / (H - 1) - 1.0], dim=-1).unsqueeze(0).unsqueeze(0)

    class DiffPoseSim3(nn.Module):
        def __init__(self, device: str):
            super().__init__()
            self.t = nn.Parameter(torch.zeros(3, device=device))
            self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
            self.log_s = nn.Parameter(torch.tensor(0.0, device=device))

        def init_from_T(self, T: np.ndarray):
            R, t, s, detR = decompose_similarity(T)
            q = rot_to_quat_np(R)
            with torch.no_grad():
                self.t.copy_(torch.tensor(t, dtype=torch.float32, device=self.t.device))
                self.q.copy_(torch.tensor(q, dtype=torch.float32, device=self.t.device))
                self.log_s.copy_(torch.tensor(math.log(max(s, 1e-8)), dtype=torch.float32, device=self.t.device))

        def forward(self, pts):
            q = F.normalize(self.q, dim=0)
            R = quat_to_rot_torch(q)
            s = self.log_s.exp()
            return s * (pts @ R.T) + self.t

        def matrix_np(self):
            with torch.no_grad():
                q = F.normalize(self.q, dim=0).detach().cpu().numpy()
                R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                s = float(self.log_s.exp().item())
                t = self.t.detach().cpu().numpy().astype(np.float64)
            return compose_similarity(R, t, s)


def optimize_pose_torch(
    src_pts_np: np.ndarray,
    target_pts_np: np.ndarray,
    depth_np: np.ndarray,
    corr_mask_np: np.ndarray,
    sil_mask_np: np.ndarray,
    T_init: np.ndarray,
    K: np.ndarray,
    n_iter: int,
    lr: float,
    device: str,
    local_up_np: Optional[np.ndarray] = None,
    upright_normal_np: Optional[np.ndarray] = None,
    upright_weight: float = 0.0,
    scale_lock: bool = False,
    scale_refine_limit: float = 0.03,
) -> np.ndarray:
    if not TORCH_AVAILABLE:
        return T_init
    H, W = depth_np.shape
    rng = np.random.default_rng(123)
    src_pts = src_pts_np
    if len(src_pts) > 8000:
        src_pts = src_pts[rng.choice(len(src_pts), 8000, replace=False)]
    tgt_pts = target_pts_np
    if len(tgt_pts) > 2000:
        tgt_pts = tgt_pts[rng.choice(len(tgt_pts), 2000, replace=False)]

    src_t = torch.tensor(src_pts.astype(np.float32), device=device)
    tgt_t = torch.tensor(tgt_pts.astype(np.float32), device=device)
    depth_t = torch.tensor(depth_np.astype(np.float32), device=device)
    corr_t = torch.tensor(corr_mask_np.astype(np.float32), device=device)
    sil_t = torch.tensor(sil_mask_np.astype(np.float32), device=device)
    if local_up_np is not None and upright_normal_np is not None and upright_weight > 0:
        local_up_t = torch.tensor(np.asarray(local_up_np, dtype=np.float32), device=device)
        local_up_t = local_up_t / (local_up_t.norm() + 1e-9)
        upright_normal_t = torch.tensor(np.asarray(upright_normal_np, dtype=np.float32), device=device)
        upright_normal_t = upright_normal_t / (upright_normal_t.norm() + 1e-9)
    else:
        local_up_t = None
        upright_normal_t = None

    R0, _, s0, _ = decompose_similarity(T_init)
    log_s_prior = torch.tensor(math.log(max(s0, 1e-8)), dtype=torch.float32, device=device)

    pose = DiffPoseSim3(device=device)
    pose.init_from_T(T_init)
    if scale_lock:
        pose.log_s.requires_grad_(False)
        opt_params = [
            {"params": [pose.t], "lr": lr},
            {"params": [pose.q], "lr": lr * 0.6},
        ]
    else:
        opt_params = [
            {"params": [pose.t], "lr": lr},
            {"params": [pose.q], "lr": lr * 0.6},
            {"params": [pose.log_s], "lr": lr * 0.10},
        ]
    opt = torch.optim.Adam(opt_params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(int(n_iter), 1), eta_min=lr * 0.05)

    fx, fy, cx, cy = [float(x) for x in (K[0, 0], K[1, 1], K[0, 2], K[1, 2])]
    best_T = T_init.copy()
    best_loss = float("inf")
    patience = 25
    wait = 0

    for _ in range(int(n_iter)):
        opt.zero_grad(set_to_none=True)
        p = pose(src_t)
        z = p[:, 2]
        valid = z > 0.01
        z_safe = z.clamp(min=0.01)
        u = fx * p[:, 0] / z_safe + cx
        v = fy * p[:, 1] / z_safe + cy
        grid = make_grid_torch(u, v, H, W)
        d_s = F.grid_sample(depth_t[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze()
        c_s = F.grid_sample(corr_t[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze()
        m_s = F.grid_sample(sil_t[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze()

        good = valid & (d_s > 0.01) & (c_s > 0.3)
        if int(good.sum().item()) > 20:
            L_depth = F.huber_loss(z[good], d_s[good], delta=0.03)
            ratio = z[good] / d_s[good].clamp(min=0.01)
            sr, _ = ratio.sort()
            lo = max(1, int(0.1 * len(sr)))
            hi = min(len(sr) - 1, int(0.9 * len(sr)))
            L_ratio = (sr[lo:hi].mean() - 1.0).pow(2) if hi - lo > 4 else torch.tensor(0.0, device=device)
            p_cham = p[good]
        else:
            L_depth = torch.tensor(0.0, device=device)
            L_ratio = torch.tensor(0.0, device=device)
            p_cham = p[valid]

        if int(valid.sum().item()) > 20:
            L_sil = (1.0 - m_s[valid]).mean()
        else:
            L_sil = torch.tensor(0.0, device=device)

        if len(tgt_t) > 50 and len(p_cham) > 50:
            if len(p_cham) > 1200:
                idx = torch.randperm(len(p_cham), device=device)[:1200]
                p_cham = p_cham[idx]
            tgt_sel = tgt_t
            if len(tgt_sel) > 1200:
                idx = torch.randperm(len(tgt_sel), device=device)[:1200]
                tgt_sel = tgt_sel[idx]
            d_fw = torch.cdist(p_cham, tgt_sel).min(dim=1).values.mean()
            d_bw = torch.cdist(tgt_sel, p_cham).min(dim=1).values.mean()
            L_cham = d_fw + d_bw
        else:
            L_cham = torch.tensor(0.0, device=device)

        L_scale = (pose.log_s - log_s_prior).pow(2)
        if local_up_t is not None:
            q_unit = F.normalize(pose.q, dim=0)
            R_cur = quat_to_rot_torch(q_unit)
            up_cur = R_cur @ local_up_t
            up_cur = up_cur / (up_cur.norm() + 1e-9)
            dot_up = torch.clamp((up_cur * upright_normal_t).sum(), -1.0, 1.0)
            L_upright = (1.0 - dot_up).pow(2)
        else:
            L_upright = torch.tensor(0.0, device=device)
        loss = (1.0 * L_depth + 0.5 * L_ratio + 0.20 * L_sil + 0.20 * L_cham
                + 4.0 * L_scale + float(upright_weight) * 8.0 * L_upright)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pose.parameters(), 1.0)
        opt.step()
        if not scale_lock and scale_refine_limit is not None and scale_refine_limit >= 0:
            lim = math.log(1.0 + float(scale_refine_limit))
            with torch.no_grad():
                pose.log_s.clamp_(log_s_prior - lim, log_s_prior + lim)
        sched.step()

        val = float(loss.item())
        if val < best_loss - 1e-6:
            best_loss = val
            best_T = pose.matrix_np()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    return best_T


# ----------------------------- core fitting --------------------------------

def fit_one_frame(
    fid: int,
    rgb: np.ndarray,
    masked: np.ndarray,
    depth: np.ndarray,
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    cfg: AlignConfig,
    device: str,
    global_scale: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, PoseMetrics, Dict]:
    rng = np.random.default_rng(cfg.seed + fid_to_seed(fid))
    H, W = depth.shape

    sil_mask = make_mask_from_provided(masked, threshold=cfg.mask_threshold)
    corr_mask = make_depth_corr_mask(
        depth,
        sil_mask,
        erode_px=cfg.erode_px,
        p_low=cfg.depth_percentile_low,
        p_high=cfg.depth_percentile_high,
        margin_m=cfg.depth_margin_m,
    )
    target_pts = backproject(depth, corr_mask, K)
    if len(target_pts) < cfg.min_target_points:
        raise RuntimeError(f"frame {fid}: too few target depth points: {len(target_pts)}")

    local_up = local_axis_from_name(cfg.upright_axis)
    upright_normal, upright_debug = get_upright_normal(cfg.upright_mode, depth, sil_mask, K, cfg, seed=cfg.seed + fid_to_seed(fid) * 17)

    src_pts_un, _ = trimesh.sample.sample_surface(mesh, cfg.n_sample,
                                                    seed=cfg.seed + fid_to_seed(fid) * 31)
    src_pts_un = np.asarray(src_pts_un, dtype=np.float64)
    # Use vertices for scale estimation so thin tips/antennas are not lost by
    # surface sampling percentiles. Use samples for robust correspondence.
    scale_pts_un = np.asarray(mesh.vertices, dtype=np.float64)
    if len(scale_pts_un) < 8:
        scale_pts_un = src_pts_un
    src_centroid = src_pts_un.mean(axis=0)
    target_centroid = target_pts.mean(axis=0)

    s_pca = estimate_scale_pca(src_pts_un, target_pts)
    s_bbox = estimate_scale_bbox(sil_mask, depth, K, mesh)
    s_frame = choose_frame_scale(s_pca, s_bbox)
    s0_fallback = float(global_scale) if global_scale is not None else float(s_frame)

    if verbose:
        print(f"  mask px={int(sil_mask.sum())} corr px={int(corr_mask.sum())} target_pts={len(target_pts)}")
        print(f"  scale pca={s_pca:.5f} bbox={(s_bbox if s_bbox is not None else float('nan')):.5f} "
              f"fallback={s0_fallback:.5f} source={cfg.scale_source} lock={cfg.scale_lock}")
        if upright_normal is not None:
            print(f"  upright mode={cfg.upright_mode} axis={cfg.upright_axis} normal={np.round(upright_normal, 4).tolist()} debug={upright_debug}")

    rot_candidates = generate_rotation_candidates(src_pts_un, target_pts, n_spin=cfg.n_spin, upright_normal=upright_normal)
    if verbose:
        print(f"  rotation candidates: {len(rot_candidates)}")

    candidates: List[Candidate] = []
    for name, R0 in rot_candidates:
        if global_scale is not None:
            s0 = float(global_scale)
        elif cfg.scale_source == "bbox2d":
            s_rot = estimate_scale_bbox2d_for_R(scale_pts_un, R0, sil_mask, depth, K)
            s0 = float(s_rot) if s_rot is not None else float(s0_fallback)
        elif cfg.scale_source == "bbox":
            s0 = float(s_bbox) if s_bbox is not None and np.isfinite(s_bbox) and s_bbox > 0 else float(s0_fallback)
        elif cfg.scale_source == "pca":
            s0 = float(s_pca) if np.isfinite(s_pca) and s_pca > 0 else float(s0_fallback)
        else:
            s0 = float(s0_fallback)

        t0 = target_centroid - s0 * (R0 @ src_centroid)
        T0 = compose_similarity(R0, t0, s0)
        src_cam0 = apply_T(src_pts_un, T0)
        src_pair, tgt_pair = visible_projected_pairs(
            src_cam0,
            depth,
            corr_mask,
            K,
            min_pairs=cfg.min_pairs,
            max_pairs=cfg.n_pair_max,
            rng=rng,
        )
        if src_pair is None:
            continue
        try:
            if cfg.scale_lock:
                T_corr, _ = umeyama_rigid(src_pair, tgt_pair)
            else:
                T_corr, _ = umeyama(src_pair, tgt_pair)
        except Exception:
            continue
        T = T_corr @ T0
        if cfg.upright_hard and upright_normal is not None:
            T = upright_project_T(T, src_centroid, local_up, upright_normal)
            T = refit_translation_from_visible_depth(T, src_pts_un, depth, corr_mask, K, min_pairs=max(80, cfg.min_pairs // 2), max_iter=2)
        if cfg.bbox_scale_refine:
            T = refine_scale_to_mask_bbox(T, scale_pts_un, src_centroid, sil_mask, K, n_iter=2)
            T = refit_translation_from_visible_depth(T, src_pts_un, depth, corr_mask, K, min_pairs=max(80, cfg.min_pairs // 2), max_iter=1)
        R_clean, _, _, det_R = decompose_similarity(T)
        if det_R <= 0:
            continue
        metrics = score_pose_points(src_pts_un, T, depth, corr_mask, sil_mask, K, cfg.depth_inlier_m,
                                    local_up=local_up, upright_normal=upright_normal, upright_weight=cfg.upright_weight)
        metrics.pairs = int(len(src_pair))
        if metrics.det_R <= 0 or metrics.scale <= 0:
            continue
        candidates.append(Candidate(T=T, R_init=R0, metrics=metrics, name=name))

    if not candidates:
        raise RuntimeError(f"frame {fid}: no candidate produced enough visible depth pairs")

    candidates.sort(key=lambda c: c.metrics.score, reverse=True)

    # Optional textured/depth render re-ranking for the best geometric candidates.
    nv_renderer = None
    if (NVDIFFRAST_AVAILABLE and TORCH_AVAILABLE and device.startswith("cuda")
            and cfg.render_top > 0):
        try:
            nv_renderer = NVTexturedRenderer(mesh, K, H, W, device=device)
        except Exception as e:
            if verbose:
                print(f"  nvdiffrast texture scoring skipped: {e}")
            nv_renderer = None

    if nv_renderer is not None:
        rerank = candidates[: min(cfg.render_top, len(candidates))]
        for c in rerank:
            iou, d_err, rgb_err = nv_metrics(nv_renderer, c.T, sil_mask, corr_mask, depth, rgb)
            c.metrics.render_iou = iou
            c.metrics.render_depth_err_m = d_err
            c.metrics.rgb_err = rgb_err
            c.metrics.score = c.metrics.score + 0.8 * iou - 3.0 * min(d_err, 0.25) - 0.8 * min(rgb_err, 1.0)
        candidates.sort(key=lambda c: c.metrics.score, reverse=True)

    if verbose:
        print("  top candidates:")
        for c in candidates[: min(5, len(candidates))]:
            m = c.metrics
            extra = ""
            if m.rgb_err is not None:
                extra = f" render_iou={m.render_iou:.3f} rgb={m.rgb_err:.3f}"
            print(f"    {c.name:28s} score={m.score:+.3f} f1={m.f1:.3f} "
                  f"dinl={m.depth_inlier:.3f} derr={m.depth_median_abs_m*1000:.1f}mm "
                  f"s={m.scale:.5f} pairs={m.pairs}{extra}")

    best_candidates = candidates[: max(1, min(cfg.top_candidates, len(candidates)))]

    # Optional PyTorch refinement. Refine several top candidates and pick by the same hard score.
    if (not cfg.no_adam) and TORCH_AVAILABLE and cfg.adam_iters > 0:
        refined: List[Candidate] = []
        for c in best_candidates:
            T_ref = optimize_pose_torch(
                src_pts_un,
                target_pts,
                depth,
                corr_mask,
                sil_mask,
                c.T,
                K,
                n_iter=cfg.adam_iters,
                lr=cfg.adam_lr,
                device=device,
                local_up_np=local_up,
                upright_normal_np=upright_normal,
                upright_weight=cfg.upright_weight,
                scale_lock=cfg.scale_lock,
                scale_refine_limit=cfg.scale_refine_limit,
            )
            if cfg.upright_hard and upright_normal is not None:
                T_ref = upright_project_T(T_ref, src_centroid, local_up, upright_normal)
                T_ref = refit_translation_from_visible_depth(T_ref, src_pts_un, depth, corr_mask, K, min_pairs=max(80, cfg.min_pairs // 2), max_iter=1)
            if cfg.bbox_scale_refine:
                T_ref = refine_scale_to_mask_bbox(T_ref, scale_pts_un, src_centroid, sil_mask, K, n_iter=2)
                T_ref = refit_translation_from_visible_depth(T_ref, src_pts_un, depth, corr_mask, K, min_pairs=max(80, cfg.min_pairs // 2), max_iter=1)
            m_ref = score_pose_points(src_pts_un, T_ref, depth, corr_mask, sil_mask, K, cfg.depth_inlier_m,
                                      local_up=local_up, upright_normal=upright_normal, upright_weight=cfg.upright_weight)
            if nv_renderer is not None:
                iou, d_err, rgb_err = nv_metrics(nv_renderer, T_ref, sil_mask, corr_mask, depth, rgb)
                m_ref.render_iou = iou
                m_ref.render_depth_err_m = d_err
                m_ref.rgb_err = rgb_err
                m_ref.score = m_ref.score + 0.8 * iou - 3.0 * min(d_err, 0.25) - 0.8 * min(rgb_err, 1.0)
            refined.append(Candidate(T=T_ref, R_init=c.R_init, metrics=m_ref, name=c.name + "/adam"))
        all_final = candidates[:1] + refined
        all_final.sort(key=lambda c: c.metrics.score, reverse=True)
        best = all_final[0]
    else:
        best = candidates[0]

    R, t, s, det_R = decompose_similarity(best.T)
    best.metrics.scale = s
    best.metrics.det_R = det_R

    debug = {
        "s_pca": float(s_pca),
        "s_bbox": float(s_bbox) if s_bbox is not None else None,
        "s_init_fallback": float(s0_fallback),
        "scale_source": cfg.scale_source,
        "scale_lock": bool(cfg.scale_lock),
        "bbox_scale_refine": bool(cfg.bbox_scale_refine),
        "mask_pixels": int(sil_mask.sum()),
        "corr_pixels": int(corr_mask.sum()),
        "target_points": int(len(target_pts)),
        "num_candidates": int(len(candidates)),
        "best_name": best.name,
        "upright_mode": cfg.upright_mode,
        "upright_axis": cfg.upright_axis,
        "upright_normal": upright_normal.tolist() if upright_normal is not None else None,
        "upright_debug": upright_debug,
        "sil_mask": sil_mask,
        "corr_mask": corr_mask,
    }
    return best.T, best.metrics, debug


# ----------------------------- visualization -------------------------------

def render_overlay_cpu(
    mesh: trimesh.Trimesh,
    T: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    rgb: np.ndarray,
    alpha: float = 0.55,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU z-buffer overlay with per-triangle local masks.

    This is for visualization only. It uses a face-average z-buffer, which is
    sufficient for checking alignment and much faster than allocating a full
    HxW mask for every triangle.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    pts_cam = apply_T(verts, T)
    z = pts_cam[:, 2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    safe_z = np.maximum(z, 1e-6)
    u = fx * pts_cam[:, 0] / safe_z + cx
    v = fy * pts_cam[:, 1] / safe_z + cy
    uv = np.stack([u, v], axis=1).astype(np.int32)
    valid_v = z > 1e-6

    canvas = rgb.copy()
    sil = np.zeros((H, W), dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    face_z = z[faces].mean(axis=1)
    order = np.argsort(face_z)[::-1]
    col = np.array(color, dtype=np.float32)

    for fi in order:
        f = faces[fi]
        if not valid_v[f].all():
            continue
        tri = uv[f]
        xmin = max(int(tri[:, 0].min()), 0)
        xmax = min(int(tri[:, 0].max()), W - 1)
        ymin = max(int(tri[:, 1].min()), 0)
        ymax = min(int(tri[:, 1].max()), H - 1)
        if xmax < xmin or ymax < ymin:
            continue
        # Skip extremely large bogus triangles caused by near-plane crossings.
        if (xmax - xmin + 1) * (ymax - ymin + 1) > H * W * 0.50:
            continue
        d = float(face_z[fi])
        tri_local = tri - np.array([xmin, ymin], dtype=np.int32)
        sub_h, sub_w = ymax - ymin + 1, xmax - xmin + 1
        tmp = np.zeros((sub_h, sub_w), dtype=np.uint8)
        cv2.fillConvexPoly(tmp, tri_local, 1)
        if tmp.sum() == 0:
            continue
        z_sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        m = (tmp > 0) & (z_sub > d)
        if not m.any():
            continue
        z_sub[m] = d
        zbuf[ymin:ymax + 1, xmin:xmax + 1] = z_sub
        sil_sub = sil[ymin:ymax + 1, xmin:xmax + 1]
        sil_sub[m] = 1
        sil[ymin:ymax + 1, xmin:xmax + 1] = sil_sub
        canvas_sub = canvas[ymin:ymax + 1, xmin:xmax + 1]
        canvas_sub[m] = (canvas_sub[m].astype(np.float32) * (1.0 - alpha) + col * alpha).astype(np.uint8)
        canvas[ymin:ymax + 1, xmin:xmax + 1] = canvas_sub
    return canvas, sil.astype(bool)

def save_visuals(
    out_dir: Path,
    fid: int,
    rgb: np.ndarray,
    masked: np.ndarray,
    depth: np.ndarray,
    mesh: trimesh.Trimesh,
    T: np.ndarray,
    K: np.ndarray,
    sil_mask: np.ndarray,
    corr_mask: np.ndarray,
    metrics: PoseMetrics,
):
    H, W = depth.shape
    overlay, render_sil = render_overlay_cpu(mesh, T, K, H, W, rgb)

    contour = overlay.copy()
    target_cnts, _ = cv2.findContours((sil_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    render_cnts, _ = cv2.findContours((render_sil.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, target_cnts, -1, (255, 0, 0), 2)   # target mask contour
    cv2.drawContours(contour, render_cnts, -1, (0, 255, 0), 2)   # rendered contour
    up_txt = "" if metrics.upright_angle_deg is None else f" up={metrics.upright_angle_deg:.1f}deg"
    label = f"F1={metrics.f1:.3f} d_in={metrics.depth_inlier:.2f} d_med={metrics.depth_median_abs_m*1000:.1f}mm s={metrics.scale:.4f}{up_txt}"
    cv2.putText(contour, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    mask_vis = rgb.copy()
    mask_vis[sil_mask] = (0.45 * mask_vis[sil_mask] + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
    # mark corr mask in blue-ish/white to verify depth pixels are object-only
    mask_vis[corr_mask] = (0.55 * mask_vis[corr_mask] + np.array([0, 180, 255]) * 0.45).astype(np.uint8)

    masked_rgb = masked[..., :3] if masked.ndim == 3 else np.stack([masked] * 3, axis=-1)
    if masked_rgb.shape[:2] != rgb.shape[:2]:
        masked_rgb = cv2.resize(masked_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

    compare = np.concatenate([rgb, masked_rgb.astype(np.uint8), mask_vis, contour], axis=1)
    Image.fromarray(overlay).save(out_dir / f"overlay_{fid}.png")
    Image.fromarray(contour).save(out_dir / f"contour_{fid}.png")
    Image.fromarray(compare).save(out_dir / f"compare_{fid}.png")


# ----------------------------- global scale prepass ------------------------

def scale_prepass(frames_data: Dict, K: np.ndarray, cfg: AlignConfig) -> Dict:
    per = {}
    pca_vals = []
    bbox_vals = []
    chosen_vals = []
    for fid, (rgb, masked, depth, mesh) in frames_data.items():
        sil_mask = make_mask_from_provided(masked, threshold=cfg.mask_threshold)
        corr_mask = make_depth_corr_mask(
            depth, sil_mask, cfg.erode_px, cfg.depth_percentile_low,
            cfg.depth_percentile_high, cfg.depth_margin_m,
        )
        target_pts = backproject(depth, corr_mask, K)
        if len(target_pts) < cfg.min_target_points:
            continue
        src_pts, _ = trimesh.sample.sample_surface(mesh, min(5000, cfg.n_sample),
                                                    seed=cfg.seed + fid_to_seed(fid) * 13)
        src_pts = np.asarray(src_pts, dtype=np.float64)
        sp = estimate_scale_pca(src_pts, target_pts)
        sb = estimate_scale_bbox(sil_mask, depth, K, mesh)
        sc = choose_frame_scale(sp, sb)
        per[str(fid)] = {"pca": float(sp), "bbox": float(sb) if sb is not None else None, "chosen": float(sc)}
        pca_vals.append(float(sp))
        if sb is not None:
            bbox_vals.append(float(sb))
        chosen_vals.append(float(sc))

    cv = float(np.std(chosen_vals) / max(np.mean(chosen_vals), 1e-12)) if len(chosen_vals) >= 2 else 1.0
    global_scale = robust_median(chosen_vals) if chosen_vals else None
    return {
        "per_frame": per,
        "global_scale": float(global_scale) if global_scale is not None else None,
        "chosen_cov": cv,
        "pca_median": robust_median(pca_vals) if pca_vals else None,
        "bbox_median": robust_median(bbox_vals) if bbox_vals else None,
    }


# ----------------------------- main ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--output_dir", default=None, type=str)
    ap.add_argument("--K", default=None, type=str, help="9 comma-separated values, row-major")
    ap.add_argument("--K_json", default=None, type=str)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--device", default=None, type=str, help="cuda or cpu; overrides auto")
    ap.add_argument("--no_adam", action="store_true")
    ap.add_argument("--n_spin", default=18, type=int)
    ap.add_argument("--n_sample", default=10000, type=int)
    ap.add_argument("--top_candidates", default=5, type=int)
    ap.add_argument("--adam_iters", default=120, type=int)
    ap.add_argument("--scale_mode", choices=["auto", "global", "per_frame"], default="per_frame")
    ap.add_argument("--scale_cov_threshold", default=0.12, type=float)
    ap.add_argument("--scale_source", choices=["bbox2d", "bbox", "pca", "mixed"], default="bbox2d")
    ap.add_argument("--unlock_scale", action="store_true", help="allow Umeyama/Adam to change scale; not recommended for these RGB-D masks")
    ap.add_argument("--no_bbox_scale_refine", action="store_true", help="disable final projected-bbox scale correction")
    ap.add_argument("--scale_refine_limit", default=0.03, type=float, help="if unlocked, max fractional Adam scale drift")
    ap.add_argument("--mask_threshold", default=5, type=int)
    ap.add_argument("--erode_px", default=2, type=int)
    ap.add_argument("--render_top", default=12, type=int, help="nvdiffrast re-rank count; ignored if unavailable")
    ap.add_argument("--upright_mode", choices=["off", "camera", "table"], default="table")
    ap.add_argument("--upright_axis", default="y", help="local mesh up axis: x,y,z,-x,-y,-z")
    ap.add_argument("--upright_weight", default=2.0, type=float)
    ap.add_argument("--no_upright_hard", action="store_true", help="disable hard projection to support normal")
    ap.add_argument("--table_plane_thresh_m", default=0.008, type=float)
    ap.add_argument("--frame_ids", default=None, type=str,
                    help="comma-separated list of frame IDs to align "
                         "(default: all found). Example: --frame_ids 0,3,4")
    ap.add_argument("--use_saved_K", action="store_true",
                    help="load per-frame K from intrinsic_{fid}.npy in data_dir "
                         "(hl2ss layout, auto-converted). Overrides --K/--K_json.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "results_auto_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AlignConfig(
        n_sample=int(args.n_sample),
        n_spin=int(args.n_spin),
        top_candidates=int(args.top_candidates),
        adam_iters=int(args.adam_iters),
        no_adam=bool(args.no_adam),
        scale_mode=str(args.scale_mode),
        scale_cov_threshold=float(args.scale_cov_threshold),
        scale_source=str(args.scale_source),
        scale_lock=not bool(args.unlock_scale),
        scale_refine_limit=float(args.scale_refine_limit),
        bbox_scale_refine=not bool(args.no_bbox_scale_refine),
        mask_threshold=int(args.mask_threshold),
        erode_px=int(args.erode_px),
        render_top=int(args.render_top),
        upright_mode=str(args.upright_mode),
        upright_axis=str(args.upright_axis),
        upright_weight=float(args.upright_weight),
        upright_hard=not bool(args.no_upright_hard),
        table_plane_thresh_m=float(args.table_plane_thresh_m),
    )

    # Device selection.
    if args.device:
        device = args.device
    elif args.cpu or not (TORCH_AVAILABLE and torch.cuda.is_available()):
        device = "cpu"
    else:
        device = "cuda"
    if device == "cpu" and not args.no_adam and cfg.adam_iters > 0:
        # CPU Adam can be slow but is valid. Keep it, just warn.
        pass
    if not TORCH_AVAILABLE:
        cfg.no_adam = True

    # Enforce name/behavior consistency: when scale is locked, no other code
    # path may move scale either. refine_scale_to_mask_bbox would otherwise
    # rescale by up to 18% per call, which silently contradicts the lock.
    # Same for the Adam scale_refine_limit. User can still opt out
    # explicitly with --unlock_scale.
    if cfg.scale_lock and cfg.bbox_scale_refine:
        if not bool(args.no_bbox_scale_refine):
            print("  [info] scale_lock active -> disabling bbox_scale_refine "
                  "for consistency (use --unlock_scale to allow scale moves)")
        cfg.bbox_scale_refine = False

    frame_ids = find_frames(data_dir)
    if not frame_ids:
        raise RuntimeError(f"No rgb_{{id}}.png / depth_{{id}}.npy / mesh_{{id}}.glb frames found in {data_dir}")

    if args.frame_ids:
        wanted = [x.strip() for x in args.frame_ids.split(",") if x.strip()]
        frame_ids = [f for f in frame_ids if str(f) in wanted]
        missing = [w for w in wanted if w not in frame_ids]
        if missing:
            print(f"  [warn] frame_ids requested but missing: {missing}")
        if not frame_ids:
            raise RuntimeError(f"After --frame_ids filter, no frames left")

    # Load first frame to set K.
    rgb0, masked0, depth0, mesh0 = load_frame(data_dir, frame_ids[0])
    H0, W0 = depth0.shape
    if args.use_saved_K:
        K_saved = load_saved_K(data_dir, frame_ids[0])
        if K_saved is None:
            print(f"  [warn] --use_saved_K but intrinsic_{frame_ids[0]}.npy not "
                  f"found; falling back to default K")
            K = parse_K(args, H0, W0)
        else:
            K = K_saved
            print(f"  [info] using saved K for frame {frame_ids[0]} "
                  f"(fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                  f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f})")
    else:
        K = parse_K(args, H0, W0)

    print("\n" + "=" * 72)
    print("  auto_align_mesh_rgbd_scale_locked")
    print("=" * 72)
    print(f"  data_dir    : {data_dir}")
    print(f"  output_dir  : {out_dir}")
    print(f"  frames      : {frame_ids}")
    print(f"  device      : {device}")
    print(f"  torch       : {TORCH_AVAILABLE}  nvdiffrast: {NVDIFFRAST_AVAILABLE}")
    print(f"  K           : {K.tolist()}")
    print(f"  upright     : mode={cfg.upright_mode} axis={cfg.upright_axis} weight={cfg.upright_weight} hard={cfg.upright_hard}")
    print(f"  scale       : mode={cfg.scale_mode} source={cfg.scale_source} lock={cfg.scale_lock} bbox_refine={cfg.bbox_scale_refine}")

    frames_data = {frame_ids[0]: (rgb0, masked0, depth0, mesh0)}
    for fid in frame_ids[1:]:
        frames_data[fid] = load_frame(data_dir, fid)

    # Global scale pre-pass.
    calib = scale_prepass(frames_data, K, cfg)
    use_global = False
    if cfg.scale_mode == "global":
        use_global = True
    elif cfg.scale_mode == "auto":
        use_global = calib["global_scale"] is not None and calib["chosen_cov"] <= cfg.scale_cov_threshold
    elif cfg.scale_mode == "per_frame":
        use_global = False

    print("\n--- scale prepass ---")
    for fid in frame_ids:
        v = calib["per_frame"].get(str(fid))
        if not v:
            print(f"  frame {fid}: no scale estimate")
            continue
        print(f"  frame {fid}: pca={v['pca']:.5f} bbox={(v['bbox'] if v['bbox'] is not None else float('nan')):.5f} chosen={v['chosen']:.5f}")
    print(f"  global_scale={calib['global_scale']}  CoV={calib['chosen_cov']:.3f}  use_global={use_global}")

    results = []
    t_all0 = time.perf_counter()
    for fid in frame_ids:
        rgb, masked, depth, mesh = frames_data[fid]
        H, W = depth.shape
        if args.use_saved_K:
            K_saved = load_saved_K(data_dir, fid)
            if K_saved is not None:
                K_frame = K_saved
            elif (H, W) != (H0, W0):
                K_frame = parse_K(args, H, W)
            else:
                K_frame = K
        elif (H, W) != (H0, W0):
            K_frame = parse_K(args, H, W)
        else:
            K_frame = K
        global_scale = calib["global_scale"] if use_global else None
        print(f"\n--- frame {fid} ---")
        t0 = time.perf_counter()
        T, metrics, debug = fit_one_frame(
            fid, rgb, masked, depth, mesh, K_frame, cfg, device,
            global_scale=global_scale, verbose=True,
        )
        dt = time.perf_counter() - t0
        R, t, s, det_R = decompose_similarity(T)

        np.savez(
            out_dir / f"pose_{fid}.npz",
            T=T,
            R=R,
            t=t,
            s=np.array(s),
            K=K_frame,
            det_R=np.array(det_R),
            f1=np.array(metrics.f1),
            depth_inlier=np.array(metrics.depth_inlier),
            depth_median_abs_m=np.array(metrics.depth_median_abs_m),
            upright_angle_deg=np.array(-1.0 if metrics.upright_angle_deg is None else metrics.upright_angle_deg),
        )
        save_visuals(
            out_dir, fid, rgb, masked, depth, mesh, T, K_frame,
            debug["sil_mask"], debug["corr_mask"], metrics,
        )

        item = {
            "frame_id": str(fid),
            "time_s": float(dt),
            "T": T.tolist(),
            "R": R.tolist(),
            "t_m": t.tolist(),
            "scale": float(s),
            "det_R": float(det_R),
            "rotation_euler_xyz_deg": Rotation.from_matrix(R).as_euler("xyz", degrees=True).tolist(),
            "metrics": asdict(metrics),
            "debug": {k: v for k, v in debug.items() if k not in ("sil_mask", "corr_mask")},
        }
        results.append(item)
        print(f"  FINAL: F1={metrics.f1:.3f} depth_inlier={metrics.depth_inlier:.3f} "
              f"d_med={metrics.depth_median_abs_m*1000:.1f}mm scale={s:.5f} "
              f"det_R={det_R:.3f} time={dt:.1f}s")
        print(f"  saved: pose_{fid}.npz, compare_{fid}.png")

    summary = {
        "config": asdict(cfg),
        "camera_K": K.tolist(),
        "scale_prepass": calib,
        "use_global_scale": bool(use_global),
        "frames": results,
        "aggregate": {
            "mean_f1": float(np.mean([r["metrics"]["f1"] for r in results])),
            "mean_depth_inlier": float(np.mean([r["metrics"]["depth_inlier"] for r in results])),
            "mean_depth_median_abs_mm": float(np.mean([r["metrics"]["depth_median_abs_m"] for r in results]) * 1000.0),
            "total_time_s": float(time.perf_counter() - t_all0),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("  done")
    print("=" * 72)
    print(f"  mean F1              : {summary['aggregate']['mean_f1']:.3f}")
    print(f"  mean depth inlier    : {summary['aggregate']['mean_depth_inlier']:.3f}")
    print(f"  mean median depth err: {summary['aggregate']['mean_depth_median_abs_mm']:.1f} mm")
    print(f"  outputs              : {out_dir}")
    print("  check compare_{id}.png first: columns are rgb | rgb_masked | masks | final overlay")


if __name__ == "__main__":
    main()
