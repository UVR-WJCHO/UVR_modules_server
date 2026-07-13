"""Joint two-part alignment for combined frames.

Goal: align two single-part meshes (e.g. mesh_0 and mesh_1) simultaneously to
a combined frame's RGB + depth + mask (e.g. frame _01) so that the two meshes
together cover the mask, each occupying its own region. Returns a world pose
per part.

Strategy
--------
- 14 free parameters: (q_a, t_a, log_s_a, q_b, t_b, log_s_b), one Sim(3) each
- Differentiable rendering via nvdiffrast for each mesh independently
- Per-pixel z-buffer to fuse the two renders into a single union
- Loss combines:
    * silhouette IoU of (sil_a OR sil_b) vs the input mask  → union covers mask
    * depth L1 of union depth vs real depth                 → 3D fit
    * non-overlap penalty (sil_a × sil_b)                   → keeps the two
                                                              meshes from
                                                              landing in the
                                                              same spot
    * rgb L1 of union rgb vs real rgb                       → texture cue
    * per-mesh scale anchor (each mesh stays near its single-frame scale)
    * per-mesh upright prior (each mesh's local up aligned to table normal)

Initialisation
--------------
- R, s start from the combined frame's single-mesh alignment (e.g. pose_01)
- t is recomputed: project two starting points around the mask centroid +
  depth median so the two meshes begin at different locations in the mask
  (e.g. mesh_a slightly above mask centroid, mesh_b slightly below)
- Non-overlap penalty then naturally pushes them into disjoint regions

Usage
-----
    python joint_two_part_align.py \\
        --data_dir D:/metaobj/data/2606_samples \\
        --combined_fid 01 \\
        --part_mesh_fids 0,1 \\
        --combined_pose_npz D:/metaobj/results_2606_v16/pose_01.npz \\
        --part_pose_npzs D:/metaobj/results_2606_v16/pose_0.npz,D:/metaobj/results_2606_v16/pose_1.npz \\
        --output_dir D:/metaobj/results_2606_joint_01 \\
        --device cuda
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
from PIL import Image
from scipy.spatial.transform import Rotation

_BASE = Path("/mnt/d/metaobj") if Path("/mnt/d/metaobj").exists() else Path("D:/metaobj")
sys.path.insert(0, str(_BASE))

from auto_align_mesh_rgbd_scale_locked import (
    NVTexturedRenderer, load_saved_K,
    make_mask_from_provided, make_depth_corr_mask,
    decompose_similarity, compose_similarity,
    backproject, estimate_table_normal_from_depth,
    AlignConfig, make_projection_matrix_torch,
    local_axis_from_name,
)


# ----------------------------- helpers --------------------------------------

def quat_axis_angle(axis_np: np.ndarray, angle_rad: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for rotation around `axis_np` by angle."""
    a = np.asarray(axis_np, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    half = 0.5 * float(angle_rad)
    s = math.sin(half)
    return np.array([math.cos(half), a[0] * s, a[1] * s, a[2] * s], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in (w, x, y, z) order.
    Corresponds to R1 @ R2 (apply q2 first, then q1)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w, x, y, z) → 3x3 rotation, differentiable."""
    q = q / (q.norm() + 1e-9)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]).reshape(3, 3)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation → quaternion (w, x, y, z)."""
    q = Rotation.from_matrix(R).as_quat()  # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def grid_sample_grid(u, v, H, W):
    """Make a normalised sampling grid for F.grid_sample."""
    return torch.stack([
        2.0 * u / (W - 1) - 1.0,
        2.0 * v / (H - 1) - 1.0,
    ], dim=-1)


# ----------------------------- joint renderer -------------------------------

class TwoMeshJointRenderer:
    """Wraps two NVTexturedRenderer instances and provides a soft union."""

    def __init__(self, mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                 K: np.ndarray, H: int, W: int, device: str):
        self.device = device
        self.H, self.W = int(H), int(W)
        self.renderer_a = NVTexturedRenderer(mesh_a, K, H, W, device=device)
        self.renderer_b = NVTexturedRenderer(mesh_b, K, H, W, device=device)

    def render(self, T_a: torch.Tensor, T_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Render both meshes. Returns soft silhouettes, depths, and RGB.
        T_a, T_b are 4x4 differentiable Sim(3) matrices in OpenCV camera frame.
        """
        sil_a, depth_a, rgb_a = self._render_one(self.renderer_a, T_a)
        sil_b, depth_b, rgb_b = self._render_one(self.renderer_b, T_b)
        return {
            "sil_a": sil_a, "depth_a": depth_a, "rgb_a": rgb_a,
            "sil_b": sil_b, "depth_b": depth_b, "rgb_b": rgb_b,
        }

    def _render_one(self, r: NVTexturedRenderer, T: torch.Tensor):
        """Differentiable render of one mesh under pose T (4x4)."""
        # T already includes scale: T[:3,:3] = s * R, T[:3,3] = t
        # Choose face winding based on det(s*R)
        det = torch.linalg.det(T[:3, :3])
        faces = r.faces_rev if float(det.detach().item()) < 0 else r.faces
        vh = torch.cat([r.verts, torch.ones_like(r.verts[:, :1])], dim=1)
        vcam = (vh @ T.T)[:, :3]
        vh2 = torch.cat([vcam, torch.ones_like(vcam[:, :1])], dim=1)
        vclip = (vh2 @ r.P.T).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(r.ctx, vclip, faces, resolution=[r.H, r.W])
        sil_hard = (rast[..., 3:4] > 0).float()
        sil = dr.antialias(sil_hard, rast, vclip, faces).squeeze(0).squeeze(-1)
        z_attr = vcam[:, 2:3].unsqueeze(0).contiguous()
        zr, _ = dr.interpolate(z_attr, rast.contiguous(), faces)
        zr = zr.squeeze(0).squeeze(-1)
        uv_attr = r.uv.unsqueeze(0).contiguous()
        uv_interp, _ = dr.interpolate(uv_attr, rast.contiguous(), faces)
        rgb = dr.texture(r.tex[None], uv_interp.contiguous(),
                          filter_mode="linear").squeeze(0)
        # row 0 is NDC bottom → flip vertically for image convention
        sil = torch.flip(sil, dims=[0])
        zr = torch.flip(zr, dims=[0])
        rgb = torch.flip(rgb, dims=[0])
        return sil, zr, rgb


# ----------------------------- main optimisation ----------------------------

class JointTwoPartAlignment(nn.Module):
    def __init__(self, T_a_init: np.ndarray, T_b_init: np.ndarray,
                 s_anchor_a: float, s_anchor_b: float,
                 local_up_np: np.ndarray, upright_normal_np: np.ndarray,
                 device: str):
        super().__init__()
        # decompose to (q, t, log_s) per mesh
        R_a, t_a, s_a, _ = decompose_similarity(T_a_init)
        R_b, t_b, s_b, _ = decompose_similarity(T_b_init)
        q_a0 = rot_to_quat(R_a)
        q_b0 = rot_to_quat(R_b)

        self.q_a = nn.Parameter(torch.tensor(q_a0, dtype=torch.float32, device=device))
        self.t_a = nn.Parameter(torch.tensor(t_a, dtype=torch.float32, device=device))
        self.log_s_a = nn.Parameter(torch.tensor(math.log(max(s_a, 1e-8)),
                                                  dtype=torch.float32, device=device))
        self.q_b = nn.Parameter(torch.tensor(q_b0, dtype=torch.float32, device=device))
        self.t_b = nn.Parameter(torch.tensor(t_b, dtype=torch.float32, device=device))
        self.log_s_b = nn.Parameter(torch.tensor(math.log(max(s_b, 1e-8)),
                                                  dtype=torch.float32, device=device))

        # anchors (rotation seeds and scales)
        self.q_a_anchor = torch.tensor(q_a0, dtype=torch.float32, device=device)
        self.q_b_anchor = torch.tensor(q_b0, dtype=torch.float32, device=device)
        self.log_s_a_anchor = torch.tensor(math.log(max(s_anchor_a, 1e-8)),
                                            dtype=torch.float32, device=device)
        self.log_s_b_anchor = torch.tensor(math.log(max(s_anchor_b, 1e-8)),
                                            dtype=torch.float32, device=device)
        self.local_up = torch.tensor(local_up_np, dtype=torch.float32, device=device)
        self.local_up = self.local_up / (self.local_up.norm() + 1e-9)
        self.upright_normal = torch.tensor(upright_normal_np, dtype=torch.float32, device=device)
        self.upright_normal = self.upright_normal / (self.upright_normal.norm() + 1e-9)

    def T_a(self) -> torch.Tensor:
        return self._compose(self.q_a, self.t_a, self.log_s_a)

    def T_b(self) -> torch.Tensor:
        return self._compose(self.q_b, self.t_b, self.log_s_b)

    @staticmethod
    def _compose(q, t, log_s):
        R = quat_to_rot(q)
        s = log_s.exp()
        T = torch.zeros(4, 4, dtype=R.dtype, device=R.device)
        T[:3, :3] = s * R
        T[:3, 3] = t
        T[3, 3] = 1
        return T

    def T_a_np(self) -> np.ndarray:
        with torch.no_grad():
            return self.T_a().detach().cpu().numpy().astype(np.float64)

    def T_b_np(self) -> np.ndarray:
        with torch.no_grad():
            return self.T_b().detach().cpu().numpy().astype(np.float64)


def project_q_to_upright(
    model: "JointTwoPartAlignment",
    mesh_which: str,
    local_up_np: np.ndarray,
    upright_normal_np: np.ndarray,
    device: str,
) -> float:
    """Pre-multiply the mesh's quaternion by the smallest-angle rotation that
    maps its current local-up axis (in the camera frame) onto
    `upright_normal_np`. This removes the roll/pitch tilt that an `R_solo`
    seed (from a different camera view) can introduce when used as a yaw
    screening seed, while leaving yaw around the upright axis unchanged.

    Returns the tilt correction angle in degrees (informational)."""
    q = (model.q_a if mesh_which == "a" else model.q_b
          ).detach().cpu().numpy().astype(np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    up_cam = R @ local_up_np
    up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-12)
    target = upright_normal_np / (np.linalg.norm(upright_normal_np) + 1e-12)
    axis = np.cross(up_cam, target)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(up_cam, target))
    if sin_a < 1e-6:
        return 0.0
    axis = axis / sin_a
    angle = math.atan2(sin_a, cos_a)
    q_correct = quat_axis_angle(axis, angle)
    q_new = quat_mul(q_correct, q)
    q_new = q_new / (np.linalg.norm(q_new) + 1e-12)
    with torch.no_grad():
        tgt = model.q_a if mesh_which == "a" else model.q_b
        tgt.copy_(torch.tensor(q_new, dtype=torch.float32, device=device))
    return float(np.degrees(angle))


def screen_yaw_for_mesh(
    model: "JointTwoPartAlignment",
    renderer: "TwoMeshJointRenderer",
    mesh_which: str,            # 'a' or 'b'
    local_up_np: np.ndarray,
    w_self_t: torch.Tensor,
    rgb_t_real: torch.Tensor,
    yaw_candidates_deg: List[float],
    n_iter: int,
    lr: float,
    device: str,
    extra_q_seeds: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> Tuple[str, float]:
    """For each (seed_q, yaw) pair, set the mesh's q to seed_q ⊗ q_yaw(local_up),
    run a short Adam refine on (q, t) of the chosen mesh, score by region-
    aware silhouette+RGB matching. The other mesh is held at its current
    state during screening so it doesn't pull the screened mesh off-target.

    `extra_q_seeds` is a list of (name, q_wxyz_np) pairs. The default seed
    (combined-frame R + soft-assign t) is always tried in addition.

    Score: lower is better. Combines (1 - IoU(rendered sil ∩ assigned region))
    and an L1 RGB error inside (rendered sil ∩ assigned region). After the
    sweep, the model is updated to the winning candidate's refined (q, t).
    Returns (best_label, best_score)."""
    # snapshot full state so we can restore between candidates and at end
    state0 = {
        "q_a": model.q_a.detach().clone(), "t_a": model.t_a.detach().clone(),
        "log_s_a": model.log_s_a.detach().clone(),
        "q_b": model.q_b.detach().clone(), "t_b": model.t_b.detach().clone(),
        "log_s_b": model.log_s_b.detach().clone(),
    }
    # default seed = the post-init q (combined-frame R)
    default_seed_q_np = (model.q_a if mesh_which == "a"
                          else model.q_b).detach().cpu().numpy()
    seed_list: List[Tuple[str, np.ndarray]] = [("R_combined", default_seed_q_np)]
    if extra_q_seeds:
        seed_list.extend([(n, np.asarray(q, dtype=np.float64)) for n, q in extra_q_seeds])

    best = None  # (score, label, snapshot dict)
    for seed_name, seed_q_np in seed_list:
      for yaw_deg in yaw_candidates_deg:
        # set candidate q = seed_q ⊗ q_yaw(local_up, yaw)
        q_yaw_np = quat_axis_angle(local_up_np, math.radians(float(yaw_deg)))
        q_new_np = quat_mul(seed_q_np, q_yaw_np)
        with torch.no_grad():
            if mesh_which == "a":
                model.q_a.copy_(torch.tensor(q_new_np, dtype=torch.float32, device=device))
                # restore everything else
                model.t_a.copy_(state0["t_a"]); model.log_s_a.copy_(state0["log_s_a"])
                model.q_b.copy_(state0["q_b"]); model.t_b.copy_(state0["t_b"]); model.log_s_b.copy_(state0["log_s_b"])
                params_train = [model.q_a, model.t_a]
            else:
                model.q_b.copy_(torch.tensor(q_new_np, dtype=torch.float32, device=device))
                model.q_a.copy_(state0["q_a"]); model.t_a.copy_(state0["t_a"]); model.log_s_a.copy_(state0["log_s_a"])
                model.t_b.copy_(state0["t_b"]); model.log_s_b.copy_(state0["log_s_b"])
                params_train = [model.q_b, model.t_b]
        opt = torch.optim.Adam([{"params": params_train, "lr": lr}])
        for _ in range(int(n_iter)):
            opt.zero_grad(set_to_none=True)
            out = renderer.render(model.T_a(), model.T_b())
            sil_mine = out["sil_a"] if mesh_which == "a" else out["sil_b"]
            rgb_mine = out["rgb_a"] if mesh_which == "a" else out["rgb_b"]
            L_sil = (sil_mine - w_self_t).abs().mean()
            sil_b_mask = sil_mine > 0.5
            region_b = w_self_t > 0.3
            eval_b = sil_b_mask & region_b
            if int(eval_b.sum().item()) > 100:
                L_rgb = (rgb_mine[eval_b] - rgb_t_real[eval_b]).abs().mean()
            else:
                L_rgb = torch.tensor(0.0, device=device)
            (L_sil + 0.5 * L_rgb).backward()
            opt.step()
        with torch.no_grad():
            out = renderer.render(model.T_a(), model.T_b())
            sil_mine = out["sil_a"] if mesh_which == "a" else out["sil_b"]
            rgb_mine = out["rgb_a"] if mesh_which == "a" else out["rgb_b"]
            sil_b_mask = sil_mine > 0.5
            region_b = w_self_t > 0.3
            eval_b = sil_b_mask & region_b
            n_eval = int(eval_b.sum().item())
            rgb_err = (float((rgb_mine[eval_b] - rgb_t_real[eval_b]).abs().mean().item())
                       if n_eval > 100 else 1.0)
            inter = float((sil_b_mask & region_b).sum().item())
            union = float((sil_b_mask | region_b).sum().item())
            iou = inter / max(union, 1.0)
            # RGB is the dominant cue for yaw selection (silhouette is rotation-
            # invariant for near-symmetric shapes). Weight RGB heavily so a
            # yaw with clearly better texture match wins even when its
            # silhouette IoU is a few percent worse.
            score = (1.0 - iou) + 6.0 * rgb_err
            snap = {
                "q": (model.q_a if mesh_which == "a" else model.q_b).detach().clone(),
                "t": (model.t_a if mesh_which == "a" else model.t_b).detach().clone(),
                "log_s": (model.log_s_a if mesh_which == "a" else model.log_s_b).detach().clone(),
            }
        label = f"{seed_name}+yaw{int(yaw_deg):>3}"
        print(f"   [{mesh_which}/{label}]  iou(sil,region)={iou:.3f}  "
              f"rgb={rgb_err:.3f}  score={score:.3f}")
        if best is None or score < best[0]:
            best = (score, label, snap)

    # restore all, then apply winning yaw mesh state
    with torch.no_grad():
        model.q_a.copy_(state0["q_a"]); model.t_a.copy_(state0["t_a"]); model.log_s_a.copy_(state0["log_s_a"])
        model.q_b.copy_(state0["q_b"]); model.t_b.copy_(state0["t_b"]); model.log_s_b.copy_(state0["log_s_b"])
        if mesh_which == "a":
            model.q_a.copy_(best[2]["q"]); model.t_a.copy_(best[2]["t"]); model.log_s_a.copy_(best[2]["log_s"])
        else:
            model.q_b.copy_(best[2]["q"]); model.t_b.copy_(best[2]["t"]); model.log_s_b.copy_(best[2]["log_s"])
    return best[1], best[0]


def estimate_table_d(depth: np.ndarray, sil_mask: np.ndarray, K: np.ndarray,
                      n_table: np.ndarray) -> float:
    """Given the table normal `n_table`, fit a single plane offset `d` such
    that `n_table . v + d = 0` for points near (and around) the object's
    base. Convention: positive `n_table . v + d` means above the table.

    The plane points come from a ROI around the bottom of the object's mask,
    excluding the object itself."""
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 6.0)
    ys, xs = np.where(sil_mask)
    if len(xs) < 20 or not valid.any():
        return -0.5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    obj_dil = cv2.dilate(sil_mask.astype(np.uint8), k, iterations=1).astype(bool)
    x0 = max(int(xs.min()) - 200, 0)
    x1 = min(int(xs.max()) + 200, W - 1)
    y_obj_max = int(ys.max())
    y0 = max(y_obj_max - 30, 0)
    y1 = min(y_obj_max + 300, H - 1)
    roi = np.zeros_like(sil_mask, dtype=bool)
    roi[y0:y1 + 1, x0:x1 + 1] = True
    m = roi & ~obj_dil & valid
    pts = backproject(depth, m, K)
    if len(pts) < 100:
        return -0.5
    ds = -(pts @ n_table.astype(np.float64))
    return float(np.median(ds))


def _part_top_height_solo(mesh: trimesh.Trimesh, pose_npz_path: Path) -> float:
    """Top of the mesh in its single-part frame's camera coordinates, using
    that frame's R, t, s. Camera y points down (OpenCV), so a vertex high above
    the table has very negative y_cam; we return `-min(y_cam)` so larger means
    higher. This is a metric height proxy that does not depend on the combined
    frame's rotation (which can differ between parts)."""
    d = np.load(pose_npz_path, allow_pickle=True)
    R = np.asarray(d["R"], dtype=np.float64)
    t = np.asarray(d["t"], dtype=np.float64).reshape(3)
    s = float(d["s"])
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v_cam = s * (v @ R.T) + t
    return float(-np.min(v_cam[:, 1]))


def determine_upper_mesh(
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
    R_combined: np.ndarray, upright_normal_np: np.ndarray,
    part_pose_a: Optional[Path] = None,
    part_pose_b: Optional[Path] = None,
) -> Tuple[bool, float, float, str]:
    """Decide which mesh sits on top in the combined object.

    Primary signal (when both `part_pose_x` files are provided): each mesh's
    top height above the table in its own single-frame alignment. The mesh
    whose top reaches farther above the table when placed alone is the
    "upper" piece of the assembly. This is robust because each part already
    has a known orientation in its own frame.

    Fallback: project mesh vertices through `R_combined` and dot with
    `upright_normal_np`. Less reliable when the two part meshes were
    reconstructed with different local-frame conventions.

    Returns (upper_is_a, score_a, score_b, mode)."""
    if part_pose_a is not None and part_pose_b is not None \
            and Path(part_pose_a).exists() and Path(part_pose_b).exists():
        score_a = _part_top_height_solo(mesh_a, Path(part_pose_a))
        score_b = _part_top_height_solo(mesh_b, Path(part_pose_b))
        return (score_a > score_b), score_a, score_b, "solo-height"
    va = np.asarray(mesh_a.vertices, dtype=np.float64)
    vb = np.asarray(mesh_b.vertices, dtype=np.float64)
    proj_a = (va @ R_combined.T) @ upright_normal_np
    proj_b = (vb @ R_combined.T) @ upright_normal_np
    top_a = float(np.percentile(proj_a, 90))
    top_b = float(np.percentile(proj_b, 90))
    return (top_a > top_b), top_a, top_b, "combined-proj"


def make_geometric_region_weight(
    sil_mask: np.ndarray, upper_is_a: bool,
) -> np.ndarray:
    """Build a per-pixel region weight `w_a_geom in [0,1]` based on vertical
    position inside the mask: top half belongs more to whichever mesh is the
    upper one, bottom half to the lower one. Used as a fallback/blend with
    the RGB-based soft assignment so frames where the two parts have similar
    colours still get sensible region targets."""
    H, W = sil_mask.shape
    ys, _xs = np.where(sil_mask)
    if len(ys) < 20:
        return sil_mask.astype(np.float32) * 0.5
    y_min, y_max = float(ys.min()), float(ys.max())
    yy = np.indices((H, W))[0].astype(np.float32)
    y_norm = np.clip((yy - y_min) / max(y_max - y_min, 1.0), 0.0, 1.0)  # 0=top, 1=bot
    w_a_geom = (1.0 - y_norm) if upper_is_a else y_norm
    return w_a_geom * sil_mask.astype(np.float32)


def make_depth_compare_vis(
    real_depth: np.ndarray,
    rendered_depth: np.ndarray,
    sil_mask: np.ndarray,
    combined_fid: str,
) -> np.ndarray:
    """3-panel depth diagnostic:
        REAL  |  RENDERED  |  DIFF (rendered - real, mm)
    All restricted to the mask; outside is black. Colours:
        depth panels: VIRIDIS, range = depth values inside mask
        diff panel:   blue = mesh is in front of real (rendered z too small,
                              would look "pulled toward camera"),
                      red  = mesh is behind real."""
    H, W = real_depth.shape
    inside = sil_mask.astype(bool) & np.isfinite(real_depth) & (real_depth > 0.05)

    # Common depth scale from real values inside mask
    if inside.any():
        d_lo = float(np.percentile(real_depth[inside], 5))
        d_hi = float(np.percentile(real_depth[inside], 95))
    else:
        d_lo, d_hi = 0.4, 1.2

    def colour_depth(d, valid):
        n = np.clip((d - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1) * 255
        n = n.astype(np.uint8)
        out = cv2.applyColorMap(n, cv2.COLORMAP_VIRIDIS)
        out[~valid] = (0, 0, 0)
        return out

    rendered_valid = inside & np.isfinite(rendered_depth) & (rendered_depth > 0.05)
    real_panel = colour_depth(real_depth, inside)
    rend_panel = colour_depth(rendered_depth, rendered_valid)

    # Diff: rendered - real (mm). Negative = mesh in front (closer to camera).
    diff_mm = np.zeros_like(real_depth)
    both = inside & rendered_valid
    diff_mm[both] = (rendered_depth[both] - real_depth[both]) * 1000.0
    diff_range_mm = 100.0
    d_norm = np.clip(diff_mm / diff_range_mm, -1, 1)
    # Map to colour: -1 (mesh in front) → blue, 0 → grey, +1 (behind) → red
    r = ((d_norm > 0) * d_norm * 255).astype(np.uint8)
    b = ((d_norm < 0) * (-d_norm) * 255).astype(np.uint8)
    g = (128 - np.abs(d_norm) * 128).astype(np.uint8)
    diff_panel = np.stack([b, g, r], axis=-1)
    diff_panel[~both] = (0, 0, 0)

    # Stats text
    if int(both.sum()) > 50:
        med_mm = float(np.median(diff_mm[both]))
        mean_mm = float(diff_mm[both].mean())
    else:
        med_mm = 0.0; mean_mm = 0.0

    # Crop to mask region
    ys, xs = np.where(sil_mask)
    y0, y1 = max(int(ys.min()) - 40, 0), min(int(ys.max()) + 40, H)
    x0, x1 = max(int(xs.min()) - 60, 0), min(int(xs.max()) + 60, W)
    real_c = real_panel[y0:y1, x0:x1].copy()
    rend_c = rend_panel[y0:y1, x0:x1].copy()
    diff_c = diff_panel[y0:y1, x0:x1].copy()
    cv2.putText(real_c, "REAL depth", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(rend_c, "RENDERED depth", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(diff_c, "DIFF (rendered-real)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(diff_c, f"med {med_mm:+.0f} mean {mean_mm:+.0f} mm",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(diff_c, "BLUE: in front (toward camera)",
                (10, diff_c.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    cv2.putText(diff_c, "RED:  behind  (away from camera)",
                (10, diff_c.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    return np.concatenate([real_c, rend_c, diff_c], axis=1)


def make_four_panel_vis(
    rgb: np.ndarray,
    sil_a: np.ndarray, sil_b: np.ndarray,
    rgb_a_render: np.ndarray, rgb_b_render: np.ndarray,
    sil_mask: np.ndarray, combined_fid: str, part_fids: List[str],
) -> np.ndarray:
    """4-panel side-by-side: REAL | JOINT(both + outlines) | part_a only |
    part_b only. Outlines are red for part_a and yellow for part_b."""
    H, W = rgb.shape[:2]
    kernel = np.ones((3, 3), np.uint8)

    def contour(mask_bool):
        d = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
        return d.astype(bool) & ~mask_bool

    cont_a = contour(sil_a)
    cont_b = contour(sil_b)
    joint = rgb.copy()
    joint[sil_b] = rgb_b_render[sil_b]
    joint[sil_a] = rgb_a_render[sil_a]
    joint[cont_a] = [255, 0, 0]
    joint[cont_b] = [255, 255, 0]

    a_only = (rgb.astype(np.float32) * 0.30).astype(np.uint8)
    a_only[sil_a] = rgb_a_render[sil_a]
    a_only[cont_a] = [255, 0, 0]

    b_only = (rgb.astype(np.float32) * 0.30).astype(np.uint8)
    b_only[sil_b] = rgb_b_render[sil_b]
    b_only[cont_b] = [255, 255, 0]

    ys, xs = np.where(sil_mask)
    y0, y1 = max(int(ys.min()) - 40, 0), min(int(ys.max()) + 40, H)
    x0, x1 = max(int(xs.min()) - 60, 0), min(int(xs.max()) + 60, W)
    real_c = rgb[y0:y1, x0:x1].copy()
    joint_c = joint[y0:y1, x0:x1].copy()
    a_c = a_only[y0:y1, x0:x1].copy()
    b_c = b_only[y0:y1, x0:x1].copy()
    cv2.putText(real_c, "REAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)
    cv2.putText(joint_c, f"JOINT -> _{combined_fid}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
    cv2.putText(a_c, f"part {part_fids[0]} (red)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
    cv2.putText(b_c, f"part {part_fids[1]} (yellow)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 80), 2)
    return np.concatenate([real_c, joint_c, a_c, b_c], axis=1)


def init_translation_from_region(
    depth: np.ndarray,
    region_weight: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project the weighted centroid of `region_weight` to camera-frame
    3D using the median depth inside that region. region_weight in [0,1]."""
    H, W = depth.shape
    valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 6.0)
    w = region_weight.astype(np.float32)
    w_sum = float(w.sum())
    if w_sum < 1.0:
        return np.array([0.0, 0.0, 0.5], dtype=np.float64)
    ys, xs = np.indices((H, W))
    cx_pix = float((w * xs).sum() / w_sum)
    cy_pix = float((w * ys).sum() / w_sum)
    depth_region = depth[(w > 0.05) & valid_depth]
    z = float(np.median(depth_region)) if depth_region.size > 0 else 0.5
    if not np.isfinite(z) or z <= 0:
        z = 0.5
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx_k, cy_k = float(K[0, 2]), float(K[1, 2])
    x = (cx_pix - cx_k) * z / fx
    y = (cy_pix - cy_k) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def _fit_rgb_gaussian(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 3-D Gaussian (mean, cov_inv) to RGB samples in [0,1]."""
    mu = samples.mean(axis=0).astype(np.float64)
    cov = np.cov(samples.T) + 1e-3 * np.eye(3)
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv


def _mahalanobis_sq(pix: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    d = pix - mu[None, :]
    return np.einsum("ij,jk,ik->i", d, cov_inv, d)


def soft_assign_pixels(
    rgb_combined: np.ndarray,
    mask_combined: np.ndarray,
    data_dir: Path,
    part_fids: List[str],
    tau: float = 4.0,
    sample_cap: int = 4000,
) -> np.ndarray:
    """Per-pixel soft assignment to part_a/part_b inside the combined mask.

    Builds a Gaussian RGB model from each single-part frame's masked pixels,
    then scores combined-mask pixels by Mahalanobis distance and softmaxes
    the negative half-distances with temperature `tau`.

    Returns w_a in [0,1] same shape as mask_combined. w_a = 0 outside mask,
    w_b = mask * (1 - w_a).
    """
    H, W = mask_combined.shape
    rgb01 = rgb_combined.astype(np.float32) / 255.0
    samples_per_part = []
    rng = np.random.default_rng(0)
    for fid in part_fids:
        rgb = np.array(Image.open(data_dir / f"rgb_{fid}.png").convert("RGB"))
        masked = np.array(Image.open(data_dir / f"rgb_masked_{fid}.png"))
        m = make_mask_from_provided(masked)
        s = (rgb[m].astype(np.float32) / 255.0)
        if len(s) > sample_cap:
            s = s[rng.choice(len(s), sample_cap, replace=False)]
        samples_per_part.append(s)
        print(f"[soft] mesh_{fid}: {len(s)} RGB samples  "
              f"mean={s.mean(0).round(3).tolist()}  std={s.std(0).round(3).tolist()}")
    mu_a, ci_a = _fit_rgb_gaussian(samples_per_part[0])
    mu_b, ci_b = _fit_rgb_gaussian(samples_per_part[1])
    pix = rgb01[mask_combined]
    d_a = _mahalanobis_sq(pix, mu_a, ci_a)
    d_b = _mahalanobis_sq(pix, mu_b, ci_b)
    log_w_a = -0.5 * d_a / tau
    log_w_b = -0.5 * d_b / tau
    m_lse = np.maximum(log_w_a, log_w_b)
    e_a = np.exp(log_w_a - m_lse); e_b = np.exp(log_w_b - m_lse)
    w_a = e_a / (e_a + e_b)
    out = np.zeros((H, W), dtype=np.float32)
    out[mask_combined] = w_a
    # report a simple separation diagnostic
    p_a = float((w_a > 0.7).mean())
    p_b = float((w_a < 0.3).mean())
    print(f"[soft] confident-A pixels={p_a:.2f}  confident-B pixels={p_b:.2f}  "
          f"ambiguous={1.0 - p_a - p_b:.2f}  (1.0=fully separated, 0.0=identical)")
    return out


def run(args) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # --- load combined frame data
    combined_fid = args.combined_fid
    rgb = np.array(Image.open(data_dir / f"rgb_{combined_fid}.png").convert("RGB"))
    masked = np.array(Image.open(data_dir / f"rgb_masked_{combined_fid}.png"))
    depth = np.load(data_dir / f"depth_{combined_fid}.npy").astype(np.float32)
    K = load_saved_K(data_dir, combined_fid)
    if K is None:
        raise RuntimeError(f"intrinsic_{combined_fid}.npy not found")
    H, W = depth.shape
    sil_mask = make_mask_from_provided(masked)
    corr_mask = make_depth_corr_mask(depth, sil_mask)
    print(f"[load] combined frame {combined_fid}: mask_px={int(sil_mask.sum())}, "
          f"K fx={K[0,0]:.1f} cx={K[0,2]:.1f}")

    # --- part meshes
    part_fids = [p.strip() for p in args.part_mesh_fids.split(",")]
    if len(part_fids) != 2:
        raise ValueError("joint two-part alignment requires exactly 2 part_mesh_fids")
    mesh_a = trimesh.load(data_dir / f"mesh_{part_fids[0]}.glb", force="mesh")
    mesh_b = trimesh.load(data_dir / f"mesh_{part_fids[1]}.glb", force="mesh")

    # --- table normal (upright prior)
    cfg = AlignConfig()
    upright_normal, _ = estimate_table_normal_from_depth(
        depth, sil_mask, K, iters=cfg.table_ransac_iters,
        thresh_m=cfg.table_plane_thresh_m, seed=0,
    )
    if upright_normal is None:
        upright_normal = np.array([0.0, -1.0, 0.0])  # fallback to camera up
    local_up = local_axis_from_name(args.upright_axis)
    d_table = estimate_table_d(depth, sil_mask, K, upright_normal)
    print(f"[upright] normal={np.round(upright_normal, 3).tolist()}  "
          f"d_table={d_table:.3f}m")

    # --- initial poses ---------------------------------------------------------
    # combined-mesh single alignment as the seed for R and s (option (i))
    T_combined = np.eye(4)
    s_combined = 1.0
    if args.combined_pose_npz and Path(args.combined_pose_npz).exists():
        d = np.load(args.combined_pose_npz, allow_pickle=True)
        R = np.asarray(d["R"]); t_c = np.asarray(d["t"]).reshape(3); s_combined = float(d["s"])
        T_combined[:3, :3] = s_combined * R
        T_combined[:3, 3] = t_c
        print(f"[init] using combined pose from {args.combined_pose_npz} "
              f"(s={s_combined:.4f})")
    else:
        print(f"[init] no combined pose given — will use identity rotation")

    # scale anchors from single-part frame results, if available
    s_anchor_a, s_anchor_b = s_combined, s_combined
    if args.part_pose_npzs:
        npzs = [p.strip() for p in args.part_pose_npzs.split(",") if p.strip()]
        if len(npzs) == 2:
            for i, p in enumerate(npzs):
                if Path(p).exists():
                    d = np.load(p, allow_pickle=True)
                    s_p = float(d["s"])
                    if i == 0:
                        s_anchor_a = s_p
                    else:
                        s_anchor_b = s_p
                    print(f"[init] part {part_fids[i]} scale anchor = {s_p:.4f} "
                          f"(from {p})")

    # combined frame's R direction is the common seed for both meshes
    R_combined = T_combined[:3, :3] / max(s_combined, 1e-8)

    # --- geometric upper/lower decision -------------------------------------
    part_pose_paths: List[Optional[Path]] = [None, None]
    if args.part_pose_npzs:
        npzs = [p.strip() for p in args.part_pose_npzs.split(",") if p.strip()]
        if len(npzs) == 2:
            part_pose_paths = [Path(npzs[0]), Path(npzs[1])]
    upper_is_a, top_a_proj, top_b_proj, ud_mode = determine_upper_mesh(
        mesh_a, mesh_b, R_combined, upright_normal,
        part_pose_a=part_pose_paths[0], part_pose_b=part_pose_paths[1],
    )
    upper_name = part_fids[0] if upper_is_a else part_fids[1]
    lower_name = part_fids[1] if upper_is_a else part_fids[0]
    print(f"[geometry/{ud_mode}] "
          f"part_{part_fids[0]}={top_a_proj:.3f}, part_{part_fids[1]}={top_b_proj:.3f}  "
          f"-> upper=part_{upper_name}, lower=part_{lower_name}")

    # --- soft pixel assignment from single-part RGB models ------------------
    w_a_soft = soft_assign_pixels(
        rgb, sil_mask, data_dir, part_fids,
        tau=float(args.soft_tau), sample_cap=int(args.soft_sample_cap),
    )
    # RGB-only confidence: fraction of mask pixels with a clear soft winner
    inside_mask = sil_mask.astype(bool)
    if inside_mask.any():
        w_in = w_a_soft[inside_mask]
        conf_rgb = float(((w_in > 0.7) | (w_in < 0.3)).mean())
    else:
        conf_rgb = 0.0
    # Vertical coherence: does the soft assignment respect the vertical
    # geometric prior (upper mesh higher in image, lower mesh lower)?
    # Frames where the two parts share appearance but stack vertically (e.g.
    # both white with horizontal bands) show high RGB-confidence but ZERO
    # vertical coherence -- the RGB winner-take-all would produce horizontal
    # stripes, not a vertical split. Use Pearson(y_pix, w_a_soft) inside
    # mask, signed by the expected direction from `upper_is_a`.
    coherence = 0.0
    if inside_mask.sum() > 100 and w_a_soft[inside_mask].std() > 0.01:
        ys_arr, _ = np.indices(sil_mask.shape)
        y_in = ys_arr[inside_mask].astype(np.float32)
        w_in_f = w_a_soft[inside_mask].astype(np.float32)
        pearson = float(np.corrcoef(y_in, w_in_f)[0, 1])
        # If upper_is_a: a is upper, so smaller y -> larger w_a -> Pearson < 0
        # If lower_is_a: bigger y -> larger w_a -> Pearson > 0
        expected_sign = -1.0 if upper_is_a else +1.0
        coherence = float(max(0.0, expected_sign * pearson))
    # geometric prior (always-on fallback): top half = upper mesh
    w_a_geom = make_geometric_region_weight(sil_mask, upper_is_a)
    # Decouple two decisions:
    # (1) Region weight (soft vs geom): based on RGB confidence. When soft
    #     separates pixels reliably, use mostly soft regardless of vertical
    #     coherence (horizontal-stripe soft assignment is still informative,
    #     just unsuitable for binary winner-take-all).
    # (2) Loss type (binary vs fractional, set later): based on coherence.
    alpha = float(np.clip(0.2 + 0.7 * conf_rgb, 0.2, 0.9))
    w_a_np = (alpha * w_a_soft + (1.0 - alpha) * w_a_geom)
    w_a_np = (w_a_np * sil_mask.astype(np.float32)).clip(0, 1)
    w_b_np = sil_mask.astype(np.float32) * (1.0 - w_a_np)
    print(f"[blend] rgb_conf={conf_rgb:.2f}  vert_coherence={coherence:.2f}  "
          f"alpha(soft)={alpha:.2f}  (1-alpha)*geom={1.0 - alpha:.2f}")
    # Save assignment debug PNG: red=mesh_a, blue=mesh_b
    soft_vis = np.zeros_like(rgb)
    soft_vis[..., 0] = (w_a_np * 255).clip(0, 255).astype(np.uint8)
    soft_vis[..., 2] = (w_b_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(soft_vis).save(out_dir / f"soft_assign_{combined_fid}.png")
    print(f"[saved] soft_assign_{combined_fid}.png (R=part_a, B=part_b)")
    # init translations from the blended region centroids
    t_a_np = init_translation_from_region(depth, w_a_np, K)
    t_b_np = init_translation_from_region(depth, w_b_np, K)
    print(f"[init] t_a={t_a_np.round(3).tolist()}  t_b={t_b_np.round(3).tolist()}")

    T_a_init = compose_similarity(R_combined, t_a_np, s_anchor_a)
    T_b_init = compose_similarity(R_combined, t_b_np, s_anchor_b)

    # --- model + renderer
    renderer = TwoMeshJointRenderer(mesh_a, mesh_b, K, H, W, device=device)
    model = JointTwoPartAlignment(
        T_a_init, T_b_init, s_anchor_a, s_anchor_b,
        local_up, upright_normal, device=device,
    ).to(device)

    # --- input tensors
    depth_t = torch.tensor(depth, dtype=torch.float32, device=device)
    mask_t = torch.tensor(sil_mask.astype(np.float32), device=device)
    corr_t = torch.tensor(corr_mask.astype(np.float32), device=device)
    rgb_t_real = torch.tensor(rgb.astype(np.float32) / 255.0,
                                dtype=torch.float32, device=device)
    w_a_t = torch.tensor(w_a_np, dtype=torch.float32, device=device)
    w_b_t = torch.tensor(w_b_np, dtype=torch.float32, device=device)
    # Region target for the per-mesh silhouette loss:
    # - If the soft assignment aligns with the vertical prior (high
    #   `coherence`), the winner-take-all split is meaningful — binary loss
    #   pulls each mesh to cover its assigned region precisely.
    # - If coherence is low (soft has horizontal stripes), binary would
    #   force each mesh to cover stripe-shaped regions, distorting
    #   silhouettes (e.g. a thin antenna can't fill the whole top half).
    #   Use the fractional region weight: a weak attractor that lets other
    #   cues (depth, RGB, table contact, scale anchor) drive the layout.
    use_binary_region = (coherence >= 0.4)
    if use_binary_region:
        region_a_t = ((w_a_t > w_b_t) & (mask_t > 0.5)).float()
        region_b_t = ((w_b_t >= w_a_t) & (mask_t > 0.5)).float()
    else:
        region_a_t = w_a_t
        region_b_t = w_b_t
    print(f"[loss] region target = {'binary (coherence ok)' if use_binary_region else 'fractional (low coherence)'}")
    n_table_t = torch.tensor(upright_normal, dtype=torch.float32, device=device)
    d_table_t = torch.tensor(float(d_table), dtype=torch.float32, device=device)
    verts_a_t = torch.tensor(np.asarray(mesh_a.vertices, dtype=np.float32),
                              dtype=torch.float32, device=device)
    verts_b_t = torch.tensor(np.asarray(mesh_b.vertices, dtype=np.float32),
                              dtype=torch.float32, device=device)

    # --- yaw screening: pick best (seed, yaw) per mesh ----------------------
    # seed candidates: R_combined (combined-frame alignment) + R_p_solo (each
    # part's single-frame alignment, transferred verbatim). R_p_solo provides
    # the correct mesh face orientation when the combined-frame alignment
    # happened to pick a yaw that hides the textured face.
    if int(args.yaw_screen_iters) > 0:
        yaw_cands = [float(s) for s in str(args.yaw_screen_candidates).split(",") if s.strip()]
        if not yaw_cands:
            yaw_cands = [0.0, 90.0, 180.0, 270.0]
        extra_seeds_a: List[Tuple[str, np.ndarray]] = []
        extra_seeds_b: List[Tuple[str, np.ndarray]] = []
        if args.part_pose_npzs:
            npzs = [p.strip() for p in args.part_pose_npzs.split(",") if p.strip()]
            if len(npzs) == 2:
                for i, p in enumerate(npzs):
                    if Path(p).exists():
                        d = np.load(p, allow_pickle=True)
                        R_solo = np.asarray(d["R"], dtype=np.float64)
                        q_solo = rot_to_quat(R_solo)
                        (extra_seeds_a if i == 0 else extra_seeds_b).append(
                            ("R_solo", q_solo)
                        )
                        print(f"[yaw screen] added R_solo seed for mesh_{part_fids[i]}")
        print(f"[yaw screen] mesh_{part_fids[0]}  yaws={yaw_cands}  "
              f"iters={int(args.yaw_screen_iters)}")
        best_a, _ = screen_yaw_for_mesh(
            model, renderer, "a", local_up, w_a_t, rgb_t_real,
            yaw_cands, int(args.yaw_screen_iters), float(args.lr), device,
            extra_q_seeds=extra_seeds_a,
        )
        print(f"[yaw screen] -> mesh_{part_fids[0]} best: {best_a}")
        print(f"[yaw screen] mesh_{part_fids[1]}  yaws={yaw_cands}")
        best_b, _ = screen_yaw_for_mesh(
            model, renderer, "b", local_up, w_b_t, rgb_t_real,
            yaw_cands, int(args.yaw_screen_iters), float(args.lr), device,
            extra_q_seeds=extra_seeds_b,
        )
        print(f"[yaw screen] -> mesh_{part_fids[1]} best: {best_b}")
        # NOTE: we deliberately do NOT update q_anchor here. Keeping the
        # anchor at the original R_combined provides a weak pull-back that
        # discourages a screened-but-misaligned yaw from drifting the mesh
        # outside the mask. The main loop will start from the screening
        # winner (since model.q_a/q_b were set to the winner) and the soft
        # L_rot just keeps yaws sensible. The rot anchor weight is also
        # reduced (--w_rot_anchor default 0.5) so the pull is gentle.

        # Project each mesh's q so its local-up axis exactly aligns with
        # upright_normal in camera frame. This removes the residual tilt
        # (roll/pitch) introduced when the yaw screening picks an R_solo
        # seed whose original camera view wasn't perfectly upright. Yaw is
        # preserved because we apply the smallest-angle rotation correction.
        tilt_a = project_q_to_upright(model, "a", local_up, upright_normal, device)
        tilt_b = project_q_to_upright(model, "b", local_up, upright_normal, device)
        print(f"[yaw screen] tilt corrected: "
              f"mesh_{part_fids[0]}={tilt_a:.1f}deg  mesh_{part_fids[1]}={tilt_b:.1f}deg")

    # --- optimiser
    opt = torch.optim.Adam([
        {"params": [model.t_a, model.t_b], "lr": args.lr},
        {"params": [model.q_a, model.q_b], "lr": args.lr * 0.3},
        {"params": [model.log_s_a, model.log_s_b], "lr": args.lr * 0.05},
    ])

    n_iter = int(args.iters)
    print(f"[adam] {n_iter} iters, lr={args.lr}")
    t0 = time.perf_counter()
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        out = renderer.render(model.T_a(), model.T_b())
        sa, sb = out["sil_a"], out["sil_b"]
        za, zb = out["depth_a"], out["depth_b"]
        ra, rb = out["rgb_a"], out["rgb_b"]

        # union silhouette (soft OR via max)
        sil_union = torch.maximum(sa, sb)

        # union depth: per-pixel nearest visible mesh
        # where both visible → take min; where one only → take that; else 0
        za_v = (sa > 0.5)
        zb_v = (sb > 0.5)
        both = za_v & zb_v
        a_only = za_v & ~zb_v
        b_only = zb_v & ~za_v
        # for back-prop continuity use a soft selector based on smaller depth
        # but apply gradient only where one of the two is visible
        depth_pick = torch.where(both, torch.minimum(za, zb),
                       torch.where(a_only, za,
                       torch.where(b_only, zb, torch.zeros_like(za))))
        # rgb pick: pixel-wise nearest mesh
        rgb_pick = torch.where(both.unsqueeze(-1),
                                torch.where((za < zb).unsqueeze(-1), ra, rb),
                                torch.where(a_only.unsqueeze(-1), ra,
                                torch.where(b_only.unsqueeze(-1), rb,
                                            torch.zeros_like(ra))))

        # ---- losses ----
        # Region-aware silhouette: each mesh should cover its BINARY assigned
        # region (winner-takes-all per pixel based on w_a vs w_b inside the
        # combined mask). Fractional targets would push the silhouette to a
        # value the rasterizer can't produce.
        L_sil_a = (sa - region_a_t).abs().mean()
        L_sil_b = (sb - region_b_t).abs().mean()
        # Sanity: union still has to roughly match mask (helps in ambiguous
        # regions where w_a ~ w_b ~ 0.5).
        inter = (sil_union * mask_t).sum()
        union = (sil_union + mask_t - sil_union * mask_t).sum()
        L_sil_union = 1.0 - inter / (union + 1e-6)

        depth_valid = (corr_t > 0.5) & (depth_t > 0.05) & (depth_t < 5.0)
        rendered_visible = (sa > 0.5) | (sb > 0.5)
        depth_eval = depth_valid & rendered_visible
        if int(depth_eval.sum().item()) > 100:
            L_depth = (depth_pick[depth_eval] - depth_t[depth_eval]).abs().mean()
        else:
            L_depth = torch.tensor(0.0, device=device)

        # non-overlap: penalise pixels where both meshes are visible
        L_overlap = (sa * sb).mean()

        # rgb texture: use the union-z-buffer rgb_pick (per-pixel nearest mesh)
        # evaluated inside the combined mask. This provides cross-mesh
        # feedback: if mesh_b drifts into mesh_a's region, the z-buffer picks
        # whichever is in front and the RGB mismatch pushes the wandering
        # mesh back into its own area.
        mask_bool = mask_t > 0.5
        rgb_eval = mask_bool & rendered_visible
        if int(rgb_eval.sum().item()) > 100:
            L_rgb = (rgb_pick[rgb_eval] - rgb_t_real[rgb_eval]).abs().mean()
        else:
            L_rgb = torch.tensor(0.0, device=device)

        # scale anchors (strong: each mesh stays at its single-frame scale)
        L_scale_a = (model.log_s_a - model.log_s_a_anchor).pow(2)
        L_scale_b = (model.log_s_b - model.log_s_b_anchor).pow(2)

        # Table contact: union of both mesh's vertices, the LOWEST handful
        # (≈ the landing leg tips / cone bottom) should be near zero signed
        # distance from the table plane. Positive = above, negative =
        # penetrating. Two separate terms:
        #   L_contact:    mean of lowest-K signed dist squared -> sits ON plane
        #   L_penetrate:  mean of negative signed dists squared -> never goes
        #                 below plane (hard hinge), much stronger weight
        T_a_cur = model.T_a()
        T_b_cur = model.T_b()
        v_a_world = (verts_a_t @ T_a_cur[:3, :3].T) + T_a_cur[:3, 3]
        v_b_world = (verts_b_t @ T_b_cur[:3, :3].T) + T_b_cur[:3, 3]
        sd_a = v_a_world @ n_table_t + d_table_t
        sd_b = v_b_world @ n_table_t + d_table_t
        sd_all = torch.cat([sd_a, sd_b])
        # take lowest ~50 verts (the actual contact tips), not 1% (too broad
        # for objects with thin legs/antennae)
        kth = max(30, min(80, sd_all.numel() // 500))
        bot_k, _ = sd_all.topk(kth, largest=False)
        L_contact = bot_k.mean().pow(2)
        # Penetration: any vertex below the plane is bad. clamp(max=0) keeps
        # negative values (below plane); .pow(2).mean() makes it a strong
        # smooth penalty pulling them back up.
        L_penetrate = sd_all.clamp(max=0.0).pow(2).mean()

        # Stack alignment (rim-based): the physical contact plane between
        # the two parts is what should share a common vertical axis, not
        # each mesh's vertex mean. When the upper mesh has thin/long
        # protrusions (e.g. an antenna), its vertex mean drifts away from
        # its actual body center, and mean-based stacking systematically
        # offsets the visible body from the lower mesh's centerline.
        # Fix: align the **upper mesh's bottom rim center** with the
        # **lower mesh's top rim center** (horizontal components only).
        # Rim center = mean of the K vertices most extreme along
        # `n_table_t` on the contact side.
        rim_ratio = 0.10   # fraction of vertices considered as the rim
        n_a = v_a_world.shape[0]; n_b = v_b_world.shape[0]
        proj_a = v_a_world @ n_table_t   # larger = higher (toward camera up)
        proj_b = v_b_world @ n_table_t
        k_a = max(30, int(n_a * rim_ratio))
        k_b = max(30, int(n_b * rim_ratio))
        if upper_is_a:
            # upper=a: bottom rim of a (smallest proj), top rim of b (largest proj)
            _, idx_upper_bot = proj_a.topk(k_a, largest=False)
            _, idx_lower_top = proj_b.topk(k_b, largest=True)
            upper_bot_center = v_a_world[idx_upper_bot].mean(dim=0)
            lower_top_center = v_b_world[idx_lower_top].mean(dim=0)
        else:
            _, idx_upper_bot = proj_b.topk(k_b, largest=False)
            _, idx_lower_top = proj_a.topk(k_a, largest=True)
            upper_bot_center = v_b_world[idx_upper_bot].mean(dim=0)
            lower_top_center = v_a_world[idx_lower_top].mean(dim=0)
        # horizontal components only (project out n_table direction)
        ub_dot = (upper_bot_center * n_table_t).sum()
        lt_dot = (lower_top_center * n_table_t).sum()
        ub_h = upper_bot_center - ub_dot * n_table_t
        lt_h = lower_top_center - lt_dot * n_table_t
        L_stack = (ub_h - lt_h).pow(2).sum()

        # upright prior (each mesh's local up → table normal)
        q_a_unit = model.q_a / (model.q_a.norm() + 1e-9)
        q_b_unit = model.q_b / (model.q_b.norm() + 1e-9)
        R_a_cur = quat_to_rot(q_a_unit)
        R_b_cur = quat_to_rot(q_b_unit)
        up_a = R_a_cur @ model.local_up
        up_a = up_a / (up_a.norm() + 1e-9)
        up_b = R_b_cur @ model.local_up
        up_b = up_b / (up_b.norm() + 1e-9)
        L_upright_a = (1.0 - torch.clamp((up_a * model.upright_normal).sum(), -1.0, 1.0)).pow(2)
        L_upright_b = (1.0 - torch.clamp((up_b * model.upright_normal).sum(), -1.0, 1.0)).pow(2)

        # rotation soft anchor (allow ~few deg drift around the seed rotation)
        L_rot_a = 1.0 - torch.abs((q_a_unit * model.q_a_anchor).sum())
        L_rot_b = 1.0 - torch.abs((q_b_unit * model.q_b_anchor).sum())

        # cosine schedule for depth weight
        p = it / max(n_iter - 1, 1)
        refine_w = 1.0 - 0.5 * (1.0 + math.cos(math.pi * p))
        w_depth = 1.0 + 3.0 * refine_w

        loss = (float(args.w_sil_region) * (L_sil_a + L_sil_b)
                + float(args.w_sil_union) * L_sil_union
                + w_depth * L_depth
                + float(args.w_overlap) * L_overlap
                + 0.5 * L_rgb
                + float(args.w_scale_anchor) * (L_scale_a + L_scale_b)
                + float(args.w_upright) * (L_upright_a + L_upright_b)
                + float(args.w_rot_anchor) * L_rot_a
                + float(args.w_rot_anchor) * L_rot_b
                + float(args.w_contact) * L_contact
                + float(args.w_penetrate) * L_penetrate
                + float(args.w_stack) * L_stack)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Force the mesh's local-up axis to stay exactly aligned with
        # the camera-frame table normal after every step. This eliminates
        # the tilt that depth/silhouette/rgb losses tend to introduce as a
        # cheap way to lower their individual residuals. Yaw is preserved
        # (smallest-angle correction).
        if int(args.force_upright_each_step) != 0:
            with torch.no_grad():
                project_q_to_upright(model, "a", local_up, upright_normal, device)
                project_q_to_upright(model, "b", local_up, upright_normal, device)

        if it % 25 == 0 or it == n_iter - 1:
            with torch.no_grad():
                s_a_v = float(model.log_s_a.exp().item())
                s_b_v = float(model.log_s_b.exp().item())
            pene_mm = math.sqrt(L_penetrate.item()) * 1000
            stack_mm = math.sqrt(L_stack.item()) * 1000
            print(f"  it={it:3d}  L={loss.item():.4f}  "
                  f"silA={L_sil_a.item():.3f} silB={L_sil_b.item():.3f} "
                  f"silU={L_sil_union.item():.3f}  "
                  f"dep={L_depth.item()*1000:5.1f}mm  "
                  f"rgb={L_rgb.item():.3f}  "
                  f"cont={math.sqrt(L_contact.item())*1000:5.1f}mm "
                  f"pen={pene_mm:4.1f}mm  "
                  f"stack={stack_mm:5.1f}mm  "
                  f"s_a={s_a_v:.4f} s_b={s_b_v:.4f}")

    dt = time.perf_counter() - t0
    print(f"[done phase A] {dt:.1f}s")

    # --- Phase B: stack refinement ---------------------------------------------
    # Hard-snap the upper mesh's horizontal position to the lower mesh's
    # (component perpendicular to n_table) and refine the upper mesh's
    # rotation, vertical position and scale only. This is the explicit form
    # of: align bottom first, snap the horizontal centerline, fix rotation.
    n_stack_iter = int(args.stack_refine_iters)
    if n_stack_iter > 0:
        # which mesh is upper / lower (set during step 2 above)
        if upper_is_a:
            upper_t, lower_t = model.t_a, model.t_b
            upper_q, upper_ls = model.q_a, model.log_s_a
            upper_verts_local = verts_a_t; lower_verts_local = verts_b_t
            T_upper_fn = model.T_a; T_lower_fn = model.T_b
        else:
            upper_t, lower_t = model.t_b, model.t_a
            upper_q, upper_ls = model.q_b, model.log_s_b
            upper_verts_local = verts_b_t; lower_verts_local = verts_a_t
            T_upper_fn = model.T_b; T_lower_fn = model.T_a
        # Rim-based stacking (Phase B): align the upper mesh's BOTTOM RIM
        # centroid with the lower mesh's TOP RIM centroid in image-x. This is
        # the physical contact plane between the two parts. Anchoring to
        # rim centers (rather than vertex means) prevents thin protrusions
        # like antennae from biasing the alignment.
        fx_t = torch.tensor(float(K[0, 0]), dtype=torch.float32, device=device)
        cx_t = torch.tensor(float(K[0, 2]), dtype=torch.float32, device=device)
        rim_ratio_pb = 0.10

        def _rim_center(verts_local, T_mat, is_bottom):
            v_world = (verts_local @ T_mat[:3, :3].T) + T_mat[:3, 3]
            proj = v_world @ n_table_t
            n_v = proj.numel()
            k = max(30, int(n_v * rim_ratio_pb))
            _, idx = proj.topk(k, largest=not is_bottom)
            return v_world[idx].mean(dim=0)

        with torch.no_grad():
            lower_top = _rim_center(lower_verts_local, T_lower_fn(), is_bottom=False)
            upper_bot = _rim_center(upper_verts_local, T_upper_fn(), is_bottom=True)
            target_img_x = lower_top[0] / lower_top[2] * fx_t + cx_t
            current_img_x = upper_bot[0] / upper_bot[2] * fx_t + cx_t
            delta_x = (target_img_x - current_img_x) * upper_bot[2] / fx_t
            upper_t[0] = (upper_t[0] + delta_x).detach()
            print(f"[phase B init] snap delta_x = {delta_x.item()*1000:+.1f}mm  "
                  f"(rim-center image-x align)")
        # refine: upper q, upper t_y (will be re-snapped each step to vertical
        # only), upper log_s. Lower mesh and t.xz of upper are frozen.
        opt2 = torch.optim.Adam([
            {"params": [upper_q], "lr": float(args.lr) * 0.3},
            {"params": [upper_t], "lr": float(args.lr)},
            {"params": [upper_ls], "lr": float(args.lr) * 0.05},
        ])
        print(f"[phase B] stack refinement: {n_stack_iter} iters, "
              f"upper=part_{upper_name}, horizontal snapped to "
              f"part_{lower_name}")
        t1 = time.perf_counter()
        for it in range(n_stack_iter):
            opt2.zero_grad(set_to_none=True)
            out = renderer.render(model.T_a(), model.T_b())
            sa, sb = out["sil_a"], out["sil_b"]
            za, zb = out["depth_a"], out["depth_b"]
            ra, rb = out["rgb_a"], out["rgb_b"]
            sil_union = torch.maximum(sa, sb)
            za_v = (sa > 0.5); zb_v = (sb > 0.5)
            both = za_v & zb_v
            depth_pick = torch.where(both, torch.minimum(za, zb),
                          torch.where(za_v & ~zb_v, za,
                          torch.where(zb_v & ~za_v, zb, torch.zeros_like(za))))
            rgb_pick = torch.where(both.unsqueeze(-1),
                       torch.where((za < zb).unsqueeze(-1), ra, rb),
                       torch.where(za_v.unsqueeze(-1) & ~zb_v.unsqueeze(-1), ra,
                       torch.where(zb_v.unsqueeze(-1) & ~za_v.unsqueeze(-1), rb,
                                   torch.zeros_like(ra))))
            inter_u = (sil_union * mask_t).sum()
            union_u = (sil_union + mask_t - sil_union * mask_t).sum()
            L_sil_union2 = 1.0 - inter_u / (union_u + 1e-6)
            L_sil_a2 = (sa - region_a_t).abs().mean()
            L_sil_b2 = (sb - region_b_t).abs().mean()
            depth_valid = (corr_t > 0.5) & (depth_t > 0.05) & (depth_t < 5.0)
            rendered_visible = (sa > 0.5) | (sb > 0.5)
            depth_eval = depth_valid & rendered_visible
            L_depth2 = ((depth_pick[depth_eval] - depth_t[depth_eval]).abs().mean()
                       if int(depth_eval.sum().item()) > 100
                       else torch.tensor(0.0, device=device))
            mask_b2 = mask_t > 0.5
            rgb_eval2 = mask_b2 & rendered_visible
            L_rgb2 = ((rgb_pick[rgb_eval2] - rgb_t_real[rgb_eval2]).abs().mean()
                     if int(rgb_eval2.sum().item()) > 100
                     else torch.tensor(0.0, device=device))
            loss2 = (3.0 * (L_sil_a2 + L_sil_b2)
                     + 1.0 * L_sil_union2
                     + 2.0 * L_depth2
                     + 0.5 * L_rgb2)
            loss2.backward()
            torch.nn.utils.clip_grad_norm_([upper_q, upper_t, upper_ls], 1.0)
            opt2.step()
            # Re-snap upper's BOTTOM RIM image-x to lower's TOP RIM image-x
            # after each step. This keeps the physical contact plane of the
            # two parts on the same vertical column in the image, regardless
            # of thin protrusions that would otherwise bias a vertex-mean-
            # based snap. Depth (t.z) and vertical (t.y) remain free.
            with torch.no_grad():
                lower_top = _rim_center(lower_verts_local, T_lower_fn(), is_bottom=False)
                upper_bot = _rim_center(upper_verts_local, T_upper_fn(), is_bottom=True)
                target_img_x = lower_top[0] / lower_top[2] * fx_t + cx_t
                current_img_x = upper_bot[0] / upper_bot[2] * fx_t + cx_t
                delta_x = (target_img_x - current_img_x) * upper_bot[2] / fx_t
                upper_t[0] = (upper_t[0] + delta_x).detach()
                project_q_to_upright(model, "a" if upper_is_a else "b",
                                      local_up, upright_normal, device)
            if it % 20 == 0 or it == n_stack_iter - 1:
                print(f"  refine it={it:3d}  L={loss2.item():.4f}  "
                      f"silU={L_sil_union2.item():.3f}  "
                      f"dep={L_depth2.item()*1000:5.1f}mm  "
                      f"rgb={L_rgb2.item():.3f}")
        print(f"[done phase B] {time.perf_counter() - t1:.1f}s")

    # --- save poses
    T_a_final = model.T_a_np()
    T_b_final = model.T_b_np()
    for fid, T in zip(part_fids, [T_a_final, T_b_final]):
        R, t, s, det_R = decompose_similarity(T)
        np.savez(
            out_dir / f"pose_part_{fid}_in_C{combined_fid}.npz",
            T=T, R=R, t=t, s=np.array(s), K=K, det_R=np.array(det_R),
            combined_fid=combined_fid, part_fid=fid,
        )
    print(f"[saved] pose_part_{part_fids[0]}_in_C{combined_fid}.npz, "
          f"pose_part_{part_fids[1]}_in_C{combined_fid}.npz")

    # --- 4-panel visualization: REAL | JOINT (with outlines) | A only | B only
    with torch.no_grad():
        out = renderer.render(model.T_a(), model.T_b())
        sa = (out["sil_a"] > 0.5).cpu().numpy()
        sb = (out["sil_b"] > 0.5).cpu().numpy()
        ra = (out["rgb_a"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rb = (out["rgb_b"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        za_np = out["depth_a"].cpu().numpy()
        zb_np = out["depth_b"].cpu().numpy()
        t_a_final = model.t_a.detach().cpu().numpy()
        t_b_final = model.t_b.detach().cpu().numpy()
    panel = make_four_panel_vis(rgb, sa, sb, ra, rb, sil_mask, combined_fid, part_fids)
    Image.fromarray(panel).save(out_dir / f"side_joint_{combined_fid}.png")
    print(f"[saved] side_joint_{combined_fid}.png  (4-panel)")

    # --- depth comparison panels (combined, per-mesh) ----------------------
    # Build a union rendered depth: per pixel min(za, zb) where visible, else 0
    union_depth = np.zeros_like(za_np)
    a_vis = sa & np.isfinite(za_np) & (za_np > 0)
    b_vis = sb & np.isfinite(zb_np) & (zb_np > 0)
    union_depth[a_vis & ~b_vis] = za_np[a_vis & ~b_vis]
    union_depth[b_vis & ~a_vis] = zb_np[b_vis & ~a_vis]
    both = a_vis & b_vis
    union_depth[both] = np.minimum(za_np[both], zb_np[both])
    depth_panel_u = make_depth_compare_vis(depth, union_depth, sil_mask, combined_fid)
    Image.fromarray(depth_panel_u).save(out_dir / f"depth_compare_{combined_fid}.png")
    print(f"[saved] depth_compare_{combined_fid}.png  "
          f"(union: REAL | RENDERED | DIFF)")
    # Per-mesh depth panels: each rendered depth vs real, masked to that mesh's silhouette
    for label, sil_m, z_m in [(part_fids[0], sa, za_np), (part_fids[1], sb, zb_np)]:
        per_mask = sil_mask & sil_m
        if int(per_mask.sum()) < 30:
            continue
        rd_panel = make_depth_compare_vis(depth, z_m, per_mask, f"{label}")
        Image.fromarray(rd_panel).save(out_dir / f"depth_compare_{combined_fid}_part{label}.png")
    # Per-mesh translation z report
    real_z_in_mask = depth[sil_mask & np.isfinite(depth) & (depth > 0.05)]
    if len(real_z_in_mask) > 0:
        med_real_z = float(np.median(real_z_in_mask))
    else:
        med_real_z = 0.0
    a_real = sa & np.isfinite(depth) & (depth > 0.05)
    b_real = sb & np.isfinite(depth) & (depth > 0.05)
    med_real_a = float(np.median(depth[a_real])) if int(a_real.sum()) > 0 else 0.0
    med_real_b = float(np.median(depth[b_real])) if int(b_real.sum()) > 0 else 0.0
    med_rend_a = float(np.median(za_np[sa & np.isfinite(za_np) & (za_np > 0)])) if int(sa.sum()) > 0 else 0.0
    med_rend_b = float(np.median(zb_np[sb & np.isfinite(zb_np) & (zb_np > 0)])) if int(sb.sum()) > 0 else 0.0
    print(f"[depth] mask-median real_z={med_real_z*1000:.0f}mm")
    print(f"[depth] part_{part_fids[0]}  t.z={t_a_final[2]*1000:.0f}mm  "
          f"rendered median={med_rend_a*1000:.0f}mm  "
          f"real in this region={med_real_a*1000:.0f}mm  "
          f"diff (rend-real)={ (med_rend_a-med_real_a)*1000:+.0f}mm")
    print(f"[depth] part_{part_fids[1]}  t.z={t_b_final[2]*1000:.0f}mm  "
          f"rendered median={med_rend_b*1000:.0f}mm  "
          f"real in this region={med_real_b*1000:.0f}mm  "
          f"diff (rend-real)={ (med_rend_b-med_real_b)*1000:+.0f}mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--combined_fid", required=True,
                    help="combined frame id (e.g. '01', '34')")
    ap.add_argument("--part_mesh_fids", required=True,
                    help="comma-separated 2 part mesh fids (e.g. '0,1')")
    ap.add_argument("--combined_pose_npz", default=None,
                    help="single-rigid alignment of mesh_{combined_fid}.glb, "
                         "used as R/s seed for both parts")
    ap.add_argument("--part_pose_npzs", default=None,
                    help="comma-separated 2 pose_{part}.npz from single-frame "
                         "alignments, used as scale anchors")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", default=250, type=int)
    ap.add_argument("--lr", default=0.005, type=float)
    ap.add_argument("--w_sil_region", default=3.0, type=float,
                    help="weight on region-aware silhouette loss (per-mesh)")
    ap.add_argument("--w_sil_union", default=1.0, type=float,
                    help="weight on union-mask silhouette loss (sanity)")
    ap.add_argument("--w_overlap", default=2.0, type=float,
                    help="non-overlap penalty weight")
    ap.add_argument("--w_scale_anchor", default=50.0, type=float,
                    help="strong scale anchor weight: each mesh stays near "
                         "its single-frame metric scale")
    ap.add_argument("--w_rot_anchor", default=0.5, type=float,
                    help="rotation soft anchor pulling q toward initial "
                         "R_combined. Kept gentle so the yaw screening "
                         "winner is preserved while preventing drift.")
    ap.add_argument("--w_contact", default=20.0, type=float,
                    help="table-contact penalty: lowest K mesh vert signed "
                         "distances to table plane should be ~0")
    ap.add_argument("--w_penetrate", default=200.0, type=float,
                    help="strong hinge penalty on verts below the table "
                         "plane (signed dist < 0). Pulls penetrating verts "
                         "back up so the mesh sits ON the plane, not under.")
    ap.add_argument("--force_upright_each_step", default=1, type=int,
                    help="if non-zero, re-project each mesh's q so that its "
                         "local-up axis stays exactly aligned with the table "
                         "normal after every Adam step. Removes tilt that "
                         "depth/sil/rgb losses introduce as side effects.")
    ap.add_argument("--stack_refine_iters", default=80, type=int,
                    help="Phase B refinement iterations. After the joint "
                         "Adam (Phase A) finishes, the upper mesh's "
                         "horizontal position is snapped to the lower "
                         "mesh's (perpendicular to n_table) and only the "
                         "upper mesh's rotation, vertical position and "
                         "scale are refined. Set 0 to disable.")
    ap.add_argument("--w_stack", default=50.0, type=float,
                    help="stacking constraint weight: pulls the two meshes' "
                         "vertical axes onto a common centerline (their "
                         "horizontal positions perpendicular to n_table "
                         "should match). Encodes the prior that one part "
                         "sits on top of the other in the combined object.")
    ap.add_argument("--w_upright", default=20.0, type=float,
                    help="upright prior weight per mesh. Higher value keeps "
                         "the mesh's local up axis aligned with table "
                         "normal during main optimization, preserving the "
                         "tilt-correction applied after yaw screening.")
    ap.add_argument("--yaw_screen_iters", default=30, type=int,
                    help="Adam iters per yaw candidate during screening "
                         "(0 disables screening)")
    ap.add_argument("--yaw_screen_candidates", default="0,90,180,270",
                    help="comma-separated yaw candidate angles in degrees")
    ap.add_argument("--soft_tau", default=4.0, type=float,
                    help="softmax temperature on Mahalanobis-sq distances")
    ap.add_argument("--soft_sample_cap", default=4000, type=int,
                    help="max RGB samples per part for Gaussian fit")
    ap.add_argument("--upright_axis", default="y")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
