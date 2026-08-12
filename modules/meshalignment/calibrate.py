"""Correcting a mesh's girth against the capture it was reconstructed from.

A reconstruction can come out the right length and the wrong width. Left
uncorrected that shows up later as a step where two parts meet, and the
assembly is the wrong place to deal with it: the same step is closed just as
well by thinning the part that was already right, and an optimiser takes
whichever is cheaper. The width is only unambiguous while the part is alone and
fully in view — which is Stage 1 — so it belongs there, alongside the pose,
rather than in a stage of its own.

The correction is radial: distances from the part's central axis are scaled,
distances along it are not. That is the shape of the error a turned
reconstruction makes, and it leaves the length, which came out right, alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

from . import geom
from .frames import Frame
from .render import MeshRenderer
from .score import FitMetrics, Observation, score
from .solo import refine, trimesh_sample


@dataclass
class Calibration:
    fid: str
    factor: float             # cumulative radial scaling applied to the mesh
    width_before: float       # measured / rendered girth, before
    width_after: float
    metrics_before: FitMetrics
    metrics_after: FitMetrics
    T: np.ndarray             # pose refined against the corrected mesh


def inflate_radially(mesh: trimesh.Trimesh, axis: np.ndarray, factor: float
                     ) -> trimesh.Trimesh:
    """Scale distances from the part's central axis, leaving lengths alone."""
    out = mesh.copy()
    v = np.asarray(out.vertices, dtype=np.float64)
    c = v.mean(axis=0)
    a = geom.normalize(axis)
    d = v - c
    along = np.outer(d @ a, a)
    out.vertices = c + along + (d - along) * float(factor)
    return out


def girth_ratio(frame: Frame, renderer: MeshRenderer, T: np.ndarray,
                axis_local: np.ndarray, n_bands: int = 24) -> float:
    """Measured girth over rendered girth, across the part's length.

    Width is taken perpendicular to the axis as it appears in the image and
    sampled in bands along it, so a part photographed at a slant is measured
    across its actual girth rather than across a bounding box that mixes the
    two directions. The median over bands ignores the ends, where the outline
    turns and neither silhouette is really reporting width.
    """
    r = renderer.render_np(T)
    sil = r.visible().cpu().numpy()
    if sil.sum() < 100:
        return 1.0
    K = frame.K
    a_cam = geom.normalize(np.asarray(T)[:3, :3] @ axis_local)
    # axis direction in the image, at the part's depth
    ys, xs = np.where(sil)
    cz = float(np.median(r.depth.cpu().numpy()[sil]))
    du = np.array([K[0, 0] * a_cam[0] / cz, K[1, 1] * a_cam[1] / cz])
    if np.linalg.norm(du) < 1e-6:
        du = np.array([0.0, 1.0])
    du = du / np.linalg.norm(du)
    dv = np.array([-du[1], du[0]])

    def widths(mask):
        yy, xx = np.where(mask)
        p = np.stack([xx, yy], 1).astype(np.float64)
        t = p @ du
        w = p @ dv
        lo, hi = np.percentile(t, [8, 92])
        if hi - lo < 1e-6:
            return None
        b = np.clip(((t - lo) / (hi - lo) * n_bands).astype(int), 0, n_bands - 1)
        out = []
        for k in range(n_bands):
            m = b == k
            if m.sum() > 30:
                out.append(np.percentile(w[m], 98) - np.percentile(w[m], 2))
        return np.array(out) if out else None

    wm, wr = widths(frame.mask), widths(sil)
    if wm is None or wr is None:
        return 1.0
    n = min(len(wm), len(wr))
    return float(np.median(wm[:n] / np.maximum(wr[:n], 1e-6)))


def calibrate_frame(frame: Frame, T0: np.ndarray, *, device: str = "cuda",
                    rounds: int = 4, iters: int = 80, lr: float = 0.006,
                    max_factor: float = 0.35, tol: float = 0.005
                    ) -> Tuple[trimesh.Trimesh, Calibration]:
    """Alternate: measure the girth error, scale it out, re-fit the pose.

    Re-fitting matters — a wider mesh no longer sits where the narrow one did,
    and measuring the next round against a stale pose would chase its own tail.
    """
    H, W = frame.shape
    pts = trimesh_sample(frame.mesh, 12000, 7)
    R0, _, _ = geom.decompose(T0)
    axis, _ = geom.central_axis(pts, R_solo=R0)

    obs = Observation.build(frame.mask, frame.depth, frame.corr, frame.rgb, device)
    base = MeshRenderer(frame.mesh, frame.K, H, W, device=device)
    m_before = score(base.render_np(T0), obs)
    w_before = girth_ratio(frame, base, T0, axis)

    mesh, T, factor = frame.mesh, np.asarray(T0, dtype=np.float64), 1.0
    ratio = w_before
    for _ in range(int(rounds)):
        if abs(ratio - 1.0) < tol:
            break
        factor = float(np.clip(factor * ratio, 1 - max_factor, 1 + max_factor))
        mesh = inflate_radially(frame.mesh, axis, factor)
        rend = MeshRenderer(mesh, frame.K, H, W, device=device)
        T = refine(T, rend, obs, geom.axis_from_name("y"), frame.plane.normal,
                   0.0, iters, lr, device)
        ratio = girth_ratio(frame, rend, T, axis)

    rend = MeshRenderer(mesh, frame.K, H, W, device=device)
    return mesh, Calibration(
        fid=frame.fid, factor=factor, width_before=w_before, width_after=ratio,
        metrics_before=m_before, metrics_after=score(rend.render_np(T), obs), T=T)
