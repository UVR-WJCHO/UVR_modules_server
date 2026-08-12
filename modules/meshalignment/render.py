"""Differentiable rasterisation of a textured mesh under a Sim(3) pose.

One `MeshRenderer` owns the GPU-resident geometry/texture of a single mesh and
can render it at any resolution: full resolution for the final fit, a reduced
one for cheap hypothesis screening. All outputs are in image convention
(row 0 = top), matching the depth and mask arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh

_CTX: dict = {}


def _context(device: str):
    """One rasteriser context per device — creating them is expensive."""
    if device not in _CTX:
        _CTX[device] = dr.RasterizeCudaContext(device=device)
    return _CTX[device]


def projection_matrix(K: np.ndarray, H: int, W: int, near=0.05, far=10.0,
                      device="cuda") -> torch.Tensor:
    """OpenCV intrinsics -> OpenGL clip-space projection."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / W
    P[0, 2] = -(1.0 - 2.0 * cx / W)
    P[1, 1] = -2.0 * fy / H
    P[1, 2] = 1.0 - 2.0 * cy / H
    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = 1.0
    return P


def scale_intrinsics(K: np.ndarray, factor: float) -> np.ndarray:
    """K for an image resized by `factor` (pixel-centre convention)."""
    S = np.asarray(K, dtype=np.float64).copy()
    S[0, 0] *= factor
    S[1, 1] *= factor
    S[0, 2] = (S[0, 2] + 0.5) * factor - 0.5
    S[1, 2] = (S[1, 2] + 0.5) * factor - 0.5
    return S


@dataclass
class Render:
    """Rendered buffers. `sil` is antialiased (differentiable w.r.t. pose);
    `depth` is camera z in metres, 0 where nothing was hit."""
    sil: torch.Tensor      # HxW
    depth: torch.Tensor    # HxW
    rgb: torch.Tensor      # HxWx3, [0,1]

    def visible(self) -> torch.Tensor:
        return self.sil > 0.5


class MeshRenderer:
    def __init__(self, mesh: trimesh.Trimesh, K: np.ndarray, H: int, W: int,
                 device: str = "cuda"):
        uv = getattr(mesh.visual, "uv", None)
        tex_obj = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
        if uv is None or tex_obj is None:
            raise ValueError("mesh needs UVs and a baseColorTexture to render RGB")
        self.device = device
        self.H, self.W = int(H), int(W)
        self.verts = torch.tensor(np.asarray(mesh.vertices, np.float32), device=device)
        faces = np.asarray(mesh.faces, np.int32)
        self.faces = torch.tensor(faces, dtype=torch.int32, device=device)
        # A pose with det<0 mirrors the mesh; nvdiffrast would then cull the
        # side we actually see, so keep a reversed-winding copy for that case.
        self.faces_rev = torch.tensor(np.ascontiguousarray(faces[:, ::-1]),
                                      dtype=torch.int32, device=device)
        self.uv = torch.tensor(np.asarray(uv, np.float32), device=device)
        tex = np.asarray(tex_obj).astype(np.float32) / 255.0
        if tex.ndim == 2:
            tex = np.stack([tex] * 3, axis=-1)
        self.tex = torch.tensor(np.ascontiguousarray(tex[::-1, :, :3]),
                                dtype=torch.float32, device=device)
        self.P = projection_matrix(K, self.H, self.W, device=device)
        self.ctx = _context(device)

    def at(self, K: np.ndarray, factor: float) -> "MeshRenderer":
        """A sibling renderer for the same mesh at `factor` x resolution."""
        clone = object.__new__(MeshRenderer)
        clone.__dict__.update(self.__dict__)
        clone.H = max(int(round(self.H * factor)), 8)
        clone.W = max(int(round(self.W * factor)), 8)
        clone.P = projection_matrix(scale_intrinsics(K, factor), clone.H, clone.W,
                                    device=self.device)
        return clone

    def render(self, T: torch.Tensor) -> Render:
        """Render under a 4x4 Sim(3) `T` (torch, camera frame). Differentiable
        w.r.t. `T`."""
        det = float(torch.linalg.det(T[:3, :3]).detach())
        faces = self.faces_rev if det < 0 else self.faces
        v_cam = self.verts @ T[:3, :3].T + T[:3, 3]
        v_clip = (torch.cat([v_cam, torch.ones_like(v_cam[:, :1])], 1)
                  @ self.P.T).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(self.ctx, v_clip, faces, resolution=[self.H, self.W])
        rast_c = rast.contiguous()
        sil = dr.antialias((rast[..., 3:4] > 0).float(), rast, v_clip, faces)
        depth, _ = dr.interpolate(v_cam[:, 2:3].unsqueeze(0).contiguous(), rast_c, faces)
        uv, _ = dr.interpolate(self.uv.unsqueeze(0).contiguous(), rast_c, faces)
        rgb = dr.texture(self.tex[None], uv.contiguous(), filter_mode="linear")
        # nvdiffrast writes row 0 at the bottom of NDC; flip to image order.
        return Render(
            sil=torch.flip(sil.squeeze(0).squeeze(-1), dims=[0]),
            depth=torch.flip(depth.squeeze(0).squeeze(-1), dims=[0]),
            rgb=torch.flip(rgb.squeeze(0), dims=[0]),
        )

    def render_np(self, T_np: np.ndarray) -> Render:
        with torch.no_grad():
            return self.render(torch.tensor(np.asarray(T_np, np.float32), device=self.device))


def fuse(renders: list[Render]) -> Render:
    """Per-pixel nearest-visible z-buffer over several renders.

    Gradients flow to whichever render wins a pixel, so the fused image stays
    differentiable w.r.t. every pose involved.
    """
    if len(renders) == 1:
        return renders[0]
    sil = torch.stack([r.sil for r in renders], 0)
    depth = torch.stack([r.depth for r in renders], 0)
    rgb = torch.stack([r.rgb for r in renders], 0)
    vis = sil > 0.5
    idx = torch.where(vis, depth, torch.full_like(depth, 1e6)).argmin(0)
    any_vis = vis.any(0)
    H, W = idx.shape
    return Render(
        sil=sil.max(0).values,
        depth=torch.where(any_vis, torch.gather(depth, 0, idx[None]).squeeze(0),
                          torch.zeros_like(depth[0])),
        rgb=torch.gather(rgb, 0, idx[None, ..., None].expand(1, H, W, 3)).squeeze(0),
    )
