"""The fit objective.

One objective is used everywhere: to screen coarse pose hypotheses, to drive
gradient refinement, and to report the final quality. Screening and refining
with different objectives is what lets a hypothesis that only the *reported*
objective likes get eliminated before it is ever measured — so they are kept
deliberately identical here, one as a hard score and one as its differentiable
twin with the same cue weights.

Cues, all computed from a render against the observation:
  silhouette  — do the rendered and observed foregrounds coincide
  depth       — does the rendered surface sit at the measured distance
  rgb         — does the rendered texture match the photograph

RGB is what resolves the orientation of a near-rotationally-symmetric part,
whose silhouette barely changes with yaw. It therefore has to be present from
the very first screening pass, not bolted on at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .render import Render

# Cue weights, shared by `score` and `fit_loss` so the two agree by construction.
W_DEPTH = 3.0        # per metre of median depth error
W_RGB = 0.8          # per unit of mean |rgb| error in [0,1]
DEPTH_ERR_CAP_M = 0.10
MIN_EVAL_PX = 100


@dataclass
class Observation:
    """The measured frame, on the GPU, at one resolution."""
    mask: torch.Tensor      # HxW float {0,1}
    depth: torch.Tensor     # HxW float, metres
    corr: torch.Tensor      # HxW bool, depth trustworthy
    rgb: torch.Tensor       # HxWx3 float [0,1]

    @staticmethod
    def build(mask, depth, corr, rgb, device: str) -> "Observation":
        return Observation(
            mask=torch.tensor(mask.astype(np.float32), device=device),
            depth=torch.tensor(depth.astype(np.float32), device=device),
            corr=torch.tensor(corr.astype(bool), device=device),
            rgb=torch.tensor(rgb.astype(np.float32) / 255.0, device=device),
        )

    def downscale(self, factor: float) -> "Observation":
        if factor >= 1.0:
            return self
        H, W = self.mask.shape
        h, w = max(int(round(H * factor)), 8), max(int(round(W * factor)), 8)

        def near(x, is_float=True):
            t = x[None, None].float()
            return F.interpolate(t, size=(h, w), mode="nearest")[0, 0]

        return Observation(
            mask=near(self.mask),
            depth=near(self.depth),
            corr=near(self.corr.float()) > 0.5,
            rgb=F.interpolate(self.rgb.permute(2, 0, 1)[None], size=(h, w),
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0),
        )

    def depth_valid(self) -> torch.Tensor:
        return self.corr & (self.depth > 0.05) & (self.depth < 5.0)


@dataclass
class FitMetrics:
    score: float          # higher is better; the single number decisions use
    iou: float
    depth_err_m: float    # median |rendered - measured| inside the overlap
    depth_inlier: float   # fraction within 30 mm
    rgb_err: float
    n_eval: int
    # diagnostics — never part of `score`, but they say *how* a fit is wrong
    area_ratio: float     # sqrt(rendered px / observed px); 1 = right size
    width_ratio: float    # rendered bbox width / observed bbox width
    height_ratio: float
    dx_px: float          # rendered centroid - observed centroid
    dy_px: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def line(self) -> str:
        return (f"score={self.score:+.3f} iou={self.iou:.3f} "
                f"dep={self.depth_err_m * 1000:.1f}mm/{self.depth_inlier:.2f} "
                f"rgb={self.rgb_err:.3f} area={self.area_ratio:.3f} "
                f"wh={self.width_ratio:.3f}/{self.height_ratio:.3f} "
                f"d=({self.dx_px:+.1f},{self.dy_px:+.1f})px")


def _bbox_ratio(rendered: torch.Tensor, observed: torch.Tensor):
    """(width ratio, height ratio, dx, dy) between two boolean images."""
    def stats(m):
        idx = m.nonzero()
        if idx.numel() < 20:
            return None
        ys, xs = idx[:, 0].float(), idx[:, 1].float()
        return (torch.quantile(xs, 0.02), torch.quantile(xs, 0.98),
                torch.quantile(ys, 0.02), torch.quantile(ys, 0.98),
                xs.mean(), ys.mean())
    a, b = stats(rendered), stats(observed)
    if a is None or b is None:
        return 0.0, 0.0, 0.0, 0.0
    wr = float((a[1] - a[0]) / torch.clamp(b[1] - b[0], min=1e-6))
    hr = float((a[3] - a[2]) / torch.clamp(b[3] - b[2], min=1e-6))
    return wr, hr, float(a[4] - b[4]), float(a[5] - b[5])


@torch.no_grad()
def score(render: Render, obs: Observation) -> FitMetrics:
    """Hard evaluation of a rendered pose. Used for every ranking decision."""
    vis = render.visible()
    mask_b = obs.mask > 0.5
    inter = float((vis & mask_b).sum())
    union = float((vis | mask_b).sum())
    iou = inter / max(union, 1.0)

    dv = vis & mask_b & obs.depth_valid() & (render.depth > 0.05)
    n = int(dv.sum())
    if n > MIN_EVAL_PX:
        err = (render.depth[dv] - obs.depth[dv]).abs()
        depth_err = float(err.median())
        inlier = float((err < 0.03).float().mean())
    else:
        depth_err, inlier = DEPTH_ERR_CAP_M, 0.0

    rv = vis & mask_b
    rgb_err = (float((render.rgb[rv] - obs.rgb[rv]).abs().mean())
               if int(rv.sum()) > MIN_EVAL_PX else 1.0)

    area = float(np.sqrt(float(vis.sum()) / max(float(mask_b.sum()), 1.0)))
    wr, hr, dx, dy = _bbox_ratio(vis, mask_b)
    return FitMetrics(
        score=iou - W_DEPTH * min(depth_err, DEPTH_ERR_CAP_M) - W_RGB * rgb_err,
        iou=iou, depth_err_m=depth_err, depth_inlier=inlier, rgb_err=rgb_err,
        n_eval=n, area_ratio=area, width_ratio=wr, height_ratio=hr,
        dx_px=dx, dy_px=dy,
    )


def fit_loss(render: Render, obs: Observation) -> torch.Tensor:
    """Differentiable twin of `score` (lower is better, same cue weights).

    The silhouette term uses the antialiased coverage directly so gradients
    reach the pose through the object boundary; depth and rgb are evaluated
    only where the render actually landed on the observed object, which keeps
    a badly-placed hypothesis from being rewarded for having nothing to
    compare.
    """
    sil = render.sil
    mask = obs.mask
    inter = (sil * mask).sum()
    union = (sil + mask - sil * mask).sum()
    loss = 1.0 - inter / (union + 1e-6)

    vis = render.visible()
    mask_b = mask > 0.5
    dv = vis & mask_b & obs.depth_valid() & (render.depth > 0.05)
    if int(dv.sum()) > MIN_EVAL_PX:
        loss = loss + W_DEPTH * (render.depth[dv] - obs.depth[dv]).abs().mean()

    rv = vis & mask_b
    if int(rv.sum()) > MIN_EVAL_PX:
        loss = loss + W_RGB * (render.rgb[rv] - obs.rgb[rv]).abs().mean()
    return loss


def upright_penalty(R: torch.Tensor, local_up: torch.Tensor,
                    plane_normal: torch.Tensor) -> torch.Tensor:
    """Soft prior: the mesh's own up axis points along the support normal.

    Deliberately soft. Objects photographed in a hand are genuinely tilted,
    and hard-projecting them onto the support normal discards real information
    the depth and silhouette cues would otherwise recover.
    """
    up = R @ local_up
    up = up / (up.norm() + 1e-9)
    return (1.0 - torch.clamp((up * plane_normal).sum(), -1.0, 1.0)).pow(2)
