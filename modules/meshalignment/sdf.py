"""Signed distance field of a mesh, for testing whether parts share volume.

Two assembled parts may legitimately be *nested* — an engine cluster sits
inside the hollow of a skirt — while still never sharing material. Anything
that reasons in image space cannot tell those apart: along a view ray the
engine lies between the skirt's near and far surfaces in both the correct
assembly and a broken one. Distinguishing them needs the actual solid, so we
carry a coarse occupancy-derived distance field per mesh and query it in 3D.

Distances are in mesh-local units; multiply by the part's metric scale to get
metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.ndimage import distance_transform_edt, label


@dataclass
class MeshSDF:
    grid: torch.Tensor     # 1x1xDxHxW, signed distance in local units (+ outside)
    origin: torch.Tensor   # local coordinate of voxel (0,0,0) centre
    pitch: float
    size: torch.Tensor     # (D, H, W) as float, for normalising to [-1, 1]

    @staticmethod
    def build(mesh: trimesh.Trimesh, resolution: int = 48, pad_voxels: int = 4,
              device: str = "cuda") -> "MeshSDF":
        extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
        pitch = extent / max(int(resolution), 8)
        vox = mesh.voxelized(pitch=pitch)
        shell = np.pad(np.asarray(vox.matrix, dtype=bool), pad_voxels,
                       mode="constant", constant_values=False)

        # "Inside" means enclosed by material, found by flooding the empty
        # space inward from the padded border: whatever the flood cannot reach
        # is sealed off. Filling every void instead — which is what a plain
        # fill does — would turn an open shell into a solid block, and then a
        # part correctly nested in another's hollow, an engine cluster inside a
        # skirt, would read as deep interpenetration.
        free = ~shell
        labels, n = label(free)
        outer = set(np.unique(np.concatenate([
            labels[0].ravel(), labels[-1].ravel(),
            labels[:, 0].ravel(), labels[:, -1].ravel(),
            labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])))
        outer.discard(0)
        reachable = np.isin(labels, list(outer)) if outer else np.zeros_like(free)
        occ = shell | (free & ~reachable)

        # EDT reports >= 1 voxel on both sides of the boundary, so the raw
        # difference straddles the surface by a whole voxel; shifting each side
        # half a voxel puts the zero crossing on the surface itself.
        signed = distance_transform_edt(~occ) - distance_transform_edt(occ)
        signed = np.where(signed < 0, signed + 0.5, signed - 0.5)
        sdf = signed.astype(np.float32) * pitch
        # Local coordinate of matrix cell (0, 0, 0), shifted by the padding.
        cell0 = np.asarray(vox.indices_to_points(np.zeros((1, 3), dtype=np.int64))[0],
                           dtype=np.float64)
        origin = cell0 - pad_voxels * pitch
        D, H, W = sdf.shape
        return MeshSDF(
            grid=torch.tensor(sdf, device=device)[None, None],
            origin=torch.tensor(origin, dtype=torch.float32, device=device),
            pitch=pitch,
            size=torch.tensor([D, H, W], dtype=torch.float32, device=device),
        )

    def query(self, points_local: torch.Tensor) -> torch.Tensor:
        """Signed distance at Nx3 points in the mesh's own frame.

        Differentiable w.r.t. the points, so gradients reach whatever pose
        produced them. Points outside the padded grid clamp to the border,
        which is safely positive — a part far away is never penalised.
        """
        idx = (points_local - self.origin) / self.pitch          # voxel coords (i,j,k)
        norm = 2.0 * idx / (self.size - 1.0) - 1.0               # -> [-1, 1]
        # grid_sample addresses the last axis first.
        g = norm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
        return F.grid_sample(self.grid, g, mode="bilinear",
                             padding_mode="border", align_corners=True).view(-1)


def signed_distance(points_world: torch.Tensor, M_inv: torch.Tensor,
                    origin: torch.Tensor, sdf_other: MeshSDF,
                    dist_scale: torch.Tensor) -> torch.Tensor:
    """Signed distance from `points_world` to the other part, in metres.

    Negative inside its material, positive outside. `M_inv` and `origin` invert
    the other part's placement, bringing our points into the frame its distance
    field is expressed in.

    `dist_scale` converts that field's local units to metres. When the other
    part is scaled differently across and along its axis there is no single
    such factor — distance stretches by direction — so the across-axis scale is
    used, the one that governs the mating surfaces we care about. The two
    differ by only as much as the part is out of proportion.
    """
    local = (points_world - origin) @ M_inv.T
    return sdf_other.query(local) * dist_scale


def interpenetration(points_world: torch.Tensor, M_inv: torch.Tensor,
                     origin: torch.Tensor, sdf_other: MeshSDF,
                     dist_scale: torch.Tensor) -> torch.Tensor:
    """How far `points_world` reach inside the other part, in metres (>= 0)."""
    return torch.relu(-signed_distance(points_world, M_inv, origin, sdf_other, dist_scale))
