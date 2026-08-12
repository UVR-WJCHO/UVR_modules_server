"""Stage 1 — fit one mesh to the frame it was reconstructed from.

The mesh shape is trusted; its pose and metric scale are not. So we search
Sim(3): a coarse sweep of orientations, each given a scale and translation from
the observation, then gradient refinement of the survivors.

The search is deliberately flat: *every* hypothesis is rendered and scored with
the full objective, including texture. A part that is nearly rotationally
symmetric has almost no silhouette signal about its yaw, so a screening pass
that only looks at geometry ranks the right yaw no higher than the wrong ones
and can drop it before texture is ever consulted. Rendering everything is
affordable because the sweep runs at reduced resolution.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from . import geom
from .frames import Frame
from .render import MeshRenderer
from .score import FitMetrics, Observation, fit_loss, score, upright_penalty


@dataclass
class SoloResult:
    T: np.ndarray
    metrics: FitMetrics
    hypothesis: str
    n_hypotheses: int
    seconds: float
    runners_up: List[Tuple[str, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# hypothesis generation
# --------------------------------------------------------------------------

def _local_axes(mesh_points: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Candidate "this direction of the mesh points up" axes."""
    axes, _ = geom.pca_axes(mesh_points)
    out = [(f"pca{i}{sg}", sign * axes[:, i])
           for i in range(3) for sg, sign in (("+", 1.0), ("-", -1.0))]
    out += [(f"{nm}{sg}", sign * geom.axis_from_name(nm))
            for nm in "xyz" for sg, sign in (("+", 1.0), ("-", -1.0))]
    return out


def _target_axes(frame: Frame, target_points: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """World directions a mesh axis may be aligned to."""
    axes, _ = geom.pca_axes(target_points)
    out = [("obs+", axes[:, 0]), ("obs-", -axes[:, 0])]
    if frame.plane.ok:
        out.insert(0, ("up", frame.plane.normal))
    else:
        out.insert(0, ("up", np.array([0.0, -1.0, 0.0])))
    return out


def rotation_hypotheses(mesh_points: np.ndarray, frame: Frame,
                        target_points: np.ndarray, n_yaw: int
                        ) -> List[Tuple[str, np.ndarray]]:
    """Every (mesh axis -> world axis) pairing, spun about the world axis."""
    out: List[Tuple[str, np.ndarray]] = []
    seen = set()
    yaws = np.linspace(0.0, 2 * np.pi, int(n_yaw), endpoint=False)
    for lname, laxis in _local_axes(mesh_points):
        for tname, taxis in _target_axes(frame, target_points):
            R0 = geom.rot_align(laxis, taxis)
            for j, th in enumerate(yaws):
                R = geom.rot_about_axis(taxis, float(th)) @ R0
                key = tuple(np.round(R.ravel(), 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append((f"{lname}->{tname}/y{j:02d}", R))
    return out


# --------------------------------------------------------------------------
# placing a hypothesis: scale, then translation from the observation
# --------------------------------------------------------------------------

def _initial_scale(mesh_points: np.ndarray, R: np.ndarray, frame: Frame,
                   z_ref: float) -> float:
    """Scale that makes the rotated mesh project to the observed bounding box.

    Matching the box *area* rather than one side keeps a slightly wrong
    rotation from producing a wildly wrong size.
    """
    ys, xs = np.where(frame.mask)
    w_px = float(np.percentile(xs, 98) - np.percentile(xs, 2))
    h_px = float(np.percentile(ys, 98) - np.percentile(ys, 2))
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    rot = mesh_points @ R.T
    w_m = float(np.percentile(rot[:, 0], 98) - np.percentile(rot[:, 0], 2))
    h_m = float(np.percentile(rot[:, 1], 98) - np.percentile(rot[:, 1], 2))
    if w_m <= 1e-9 or h_m <= 1e-9:
        return 1.0
    return float(np.sqrt((w_px * z_ref / fx / w_m) * (h_px * z_ref / fy / h_m)))


def _visible_pairs(points_cam: np.ndarray, frame: Frame, max_pairs: int,
                   rng: np.random.Generator, corr: Optional[np.ndarray] = None):
    """Front-most mesh point per pixel, paired with the measured 3D point there.

    Only the nearest mesh point in each pixel can be what the depth sensor saw,
    so pairing every projected point would match hidden back-surface points to
    front-surface measurements and bias the fit backwards.
    """
    H, W = frame.shape
    z = points_cam[:, 2]
    ok = z > 1e-6
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    cx, cy = float(frame.K[0, 2]), float(frame.K[1, 2])
    u = np.rint(fx * points_cam[:, 0] / np.maximum(z, 1e-6) + cx).astype(np.int64)
    v = np.rint(fy * points_cam[:, 1] / np.maximum(z, 1e-6) + cy).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not ok.any():
        return None, None
    idx = np.where(ok)[0]
    flat = v[idx] * W + u[idx]
    # keep, per pixel, the index with the smallest z
    order = np.lexsort((z[idx], flat))
    idx, flat = idx[order], flat[order]
    first = np.ones(len(flat), dtype=bool)
    first[1:] = flat[1:] != flat[:-1]
    idx = idx[first]

    hit = (frame.corr if corr is None else corr)[v[idx], u[idx]]
    idx = idx[hit]
    if len(idx) < 50:
        return None, None
    if len(idx) > max_pairs:
        idx = idx[rng.choice(len(idx), max_pairs, replace=False)]
    zs = frame.depth[v[idx], u[idx]].astype(np.float64)
    tgt = np.stack([(u[idx] - cx) * zs / fx, (v[idx] - cy) * zs / fy, zs], axis=1)
    return points_cam[idx], tgt


def place(mesh_points: np.ndarray, R: np.ndarray, frame: Frame, z_ref: float,
          centroid_px: Tuple[float, float], rng: np.random.Generator,
          n_snap: int = 2, scale: Optional[float] = None,
          corr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Give a rotation a scale and a translation, then snap it onto the depth.

    `scale` fixes the metric size when it is already known (a part measured in
    its own capture keeps that size in every other one); the snapping step is
    rigid, so it is preserved. `corr` restricts which measured pixels may be
    matched — used to keep a part from snapping onto depth that another part
    already accounts for.
    """
    s = _initial_scale(mesh_points, R, frame, z_ref) if scale is None else float(scale)
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    cx, cy = float(frame.K[0, 2]), float(frame.K[1, 2])
    # centroid of the rotated+scaled mesh lands on the mask centroid at z_ref
    target_c = np.array([(centroid_px[0] - cx) * z_ref / fx,
                         (centroid_px[1] - cy) * z_ref / fy, z_ref])
    t = target_c - s * (R @ mesh_points.mean(axis=0))
    T = geom.compose(R, t, s)
    for _ in range(n_snap):
        src, tgt = _visible_pairs(geom.apply(mesh_points, T), frame, 20000, rng, corr)
        if src is None:
            return T
        T = geom.umeyama_rigid(src, tgt) @ T
    return T


# --------------------------------------------------------------------------
# gradient refinement
# --------------------------------------------------------------------------

def refine(T0: np.ndarray, renderer: MeshRenderer, obs: Observation,
           local_up: np.ndarray, plane_normal: np.ndarray, upright_weight: float,
           iters: int, lr: float, device: str) -> np.ndarray:
    R0, t0, s0 = geom.decompose(T0)
    q = torch.tensor(geom.quat_from_rot(R0), dtype=torch.float32, device=device,
                     requires_grad=True)
    t = torch.tensor(t0, dtype=torch.float32, device=device, requires_grad=True)
    log_s = torch.tensor(math.log(max(s0, 1e-9)), dtype=torch.float32,
                         device=device, requires_grad=True)
    up_t = torch.tensor(local_up, dtype=torch.float32, device=device)
    n_t = torch.tensor(plane_normal, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([
        {"params": [t], "lr": lr},
        {"params": [q], "lr": lr * 0.5},
        {"params": [log_s], "lr": lr * 0.2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1))
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        T = geom.sim3_torch(q, t, log_s)
        loss = fit_loss(renderer.render(T), obs)
        if upright_weight > 0:
            loss = loss + upright_weight * upright_penalty(
                geom.quat_to_rot_torch(q), up_t, n_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q, t, log_s], 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        return geom.sim3_torch(q, t, log_s).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def align_frame(
    frame: Frame,
    *,
    device: str = "cuda",
    n_yaw: int = 24,
    n_sample: int = 8000,
    screen_scale: float = 0.25,
    top_k: int = 5,
    iters: int = 150,
    lr: float = 0.01,
    upright_axis: str = "y",
    upright_weight: float = 0.0,
    seed: int = 7,
    verbose: bool = True,
) -> SoloResult:
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    H, W = frame.shape

    target = frame.target_points()
    if len(target) < 150:
        raise RuntimeError(f"frame {frame.fid}: only {len(target)} usable depth points")
    mesh_pts = np.asarray(
        trimesh_sample(frame.mesh, n_sample, seed), dtype=np.float64)

    z_ref = float(np.median(frame.depth[frame.corr]))
    ys, xs = np.where(frame.mask)
    centroid_px = (float(xs.mean()), float(ys.mean()))

    hyps = rotation_hypotheses(mesh_pts, frame, target, n_yaw)
    if verbose:
        print(f"  plane: ok={frame.plane.ok} tilt={frame.plane.tilt_deg:.1f}deg "
              f"inliers={frame.plane.n_inliers}")
        print(f"  hypotheses: {len(hyps)}  (screen at {screen_scale:g}x, "
              f"refine top {top_k})")

    placed: List[Tuple[str, np.ndarray]] = []
    for name, R in hyps:
        T = place(mesh_pts, R, frame, z_ref, centroid_px, rng)
        if T is None:
            continue
        try:
            _, _, s = geom.decompose(T)
        except ValueError:
            continue
        if s > 0:
            placed.append((name, T))
    if not placed:
        raise RuntimeError(f"frame {frame.fid}: no hypothesis could be placed")

    obs_full = Observation.build(frame.mask, frame.depth, frame.corr, frame.rgb, device)
    full = MeshRenderer(frame.mesh, frame.K, H, W, device=device)

    # --- screen every hypothesis with the full objective, cheaply -----------
    obs_small = obs_full.downscale(screen_scale)
    small = full.at(frame.K, screen_scale)
    ranked = sorted(
        ((score(small.render_np(T), obs_small).score, name, T) for name, T in placed),
        key=lambda r: -r[0])
    if verbose:
        for sc, name, _ in ranked[:5]:
            print(f"    screen {name:22s} {sc:+.3f}")

    # --- refine the survivors at full resolution ----------------------------
    local_up = geom.axis_from_name(upright_axis)
    best: Optional[Tuple[FitMetrics, np.ndarray, str]] = None
    runners: List[Tuple[str, float]] = []
    for _, name, T0 in ranked[:top_k]:
        T = refine(T0, full, obs_full, local_up, frame.plane.normal,
                   upright_weight if frame.plane.ok else 0.0, iters, lr, device)
        m = score(full.render_np(T), obs_full)
        runners.append((name, m.score))
        if verbose:
            print(f"    refine {name:22s} {m.line()}")
        if best is None or m.score > best[0].score:
            best = (m, T, name)

    return SoloResult(T=best[1], metrics=best[0], hypothesis=best[2],
                      n_hypotheses=len(placed),
                      seconds=time.perf_counter() - t_start,
                      runners_up=sorted(runners, key=lambda r: -r[1]))


def trimesh_sample(mesh, n: int, seed: int) -> np.ndarray:
    """Surface samples, falling back to vertices for degenerate meshes."""
    import trimesh as _tm
    try:
        pts, _ = _tm.sample.sample_surface(mesh, int(n), seed=seed)
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) >= 100:
            return pts
    except Exception:
        pass
    return np.asarray(mesh.vertices, dtype=np.float64)
