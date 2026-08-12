"""Stage 2 — place Stage 1's parts into one capture of the assembly.

Each part's size and attitude are already known from Stage 1, where it was
photographed alone. The assembly capture adds where the parts sit relative to
one another, and shows that only partly, because assembled parts hide each
other. What follows is shaped by that, and by one thing the picture cannot show
at all: a real assembly *mates*. The parts seat against each other along a
joint, and getting that joint right matters more than matching pixels.

**The data term is occlusion-aware.** The parts are composited through a shared
z-buffer before being compared, so every pixel is judged against the part
actually visible there rather than one hidden behind another.

**Placement order follows visibility.** Parts go in one at a time, most visible
first, each searched against the region the placed ones leave unexplained.

**Everything is expressed about each part's own central axis.** That axis is
what a turned part is built around, what a joint is coaxial with, and what
separates the reorientation the capture can resolve from the one it cannot:
tipping a part away from its Stage 1 attitude is bounded and shows in the
outline, while turning it about its own axis is neither. Scale splits along it
too — a reconstruction can come out the right height and the wrong girth, and
only letting those move separately can seat a joint without spoiling a profile.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from . import geom
from .frames import Frame, _paths
from .render import MeshRenderer, Render, fuse
from .score import FitMetrics, Observation, fit_loss, score
from .sdf import MeshSDF, signed_distance
from .solo import place, trimesh_sample

# Prior strengths, relative to the data terms in `score.fit_loss` (silhouette
# O(1), depth O(0.01)). The join outranks the picture: the terms that seat the
# parts against each other dominate, and the measured size is a starting point
# that may be trimmed to close a joint rather than a value to be defended.
W_SCALE_ANCHOR = 5.0       # (log s - log s_measured)^2, a leash not a lock
W_PENETRATION = 3000.0     # mean squared shared depth, metres^2
W_CONTACT = 2000.0         # mean squared gap left at the interface, metres^2
# Coaxiality is a real constraint but a cheap one to satisfy, and it competes
# with the only evidence of where the assembly actually is. Pushed hard it wins
# and drags the whole stack off the measured depth, so it is scaled to nudge
# rather than to dictate: enough to close a 23 deg misalignment, not enough to
# move the parts off the observation to do it.
W_COAXIAL_ANGLE = 30.0     # 1 - |a_i . a_j| between mating units
W_COAXIAL_OFFSET = 2000.0  # squared lateral offset between their axes, metres^2
# Sized so a millimetre of joint error costs what a millimetre of depth error
# costs (W_DEPTH * e against W_RIM * e^2, matched at 5 mm). Without that the
# joint terms simply outbid the only measurement that sees the assembly tipping
# toward the camera — a motion the mating circles cannot feel, since sliding
# both parts around their shared seam leaves centres, radii and normals intact.
W_RIM = 600.0              # squared mismatch of the two mating circles, metres^2
W_PROFILE = 300.0          # squared radius mismatch at the joint, metres^2
MAX_GIRTH_FIX = 0.25       # largest across-axis correction Stage 1 may ask for
W_TILT_LIMIT = 20.0        # squared radians past the allowed tipping
W_TWIST_HOLD = 50.0        # squared radians of turn about an unresolvable axis
MAX_TILT_DEG = 30.0        # how far an assembled unit may tip from its solo pose
YAW_HOLD_ASYMMETRY = 0.05  # below this a unit counts as a surface of revolution


@dataclass
class Member:
    """One mesh inside a unit, at a fixed pose in the unit's frame."""
    fid: str
    internal: np.ndarray
    renderer: MeshRenderer = field(repr=False)
    sdf: Optional[MeshSDF] = field(default=None, repr=False)


@dataclass
class Unit:
    """A rigid body to be placed: one part, or several with fixed relatives."""
    name: str
    members: List[Member]
    scale: float
    scale_radial: float
    R_solo: np.ndarray
    points: np.ndarray = field(repr=False)   # unit frame
    axis: np.ndarray = field(repr=False)     # central axis, unit frame
    rims: List[Tuple[np.ndarray, float]] = field(default_factory=list, repr=False)
    asymmetry: float = 1.0
    axis_spread: float = 0.0
    yaw_hold_thresh: float = YAW_HOLD_ASYMMETRY

    @property
    def yaw_held(self) -> bool:
        """True when the outline is a surface of revolution, so it cannot say
        how far the part is turned about its own axis.

        Only the shape is weighed here. A symmetric shape may still carry a
        texture that fixes the turn, so this is a choice about which evidence
        to trust, exposed as a threshold rather than assumed."""
        return self.asymmetry < self.yaw_hold_thresh

    def axis_world(self, R: np.ndarray) -> np.ndarray:
        return geom.normalize(np.asarray(R) @ self.axis)

    def render(self, T: torch.Tensor, renderers=None) -> Render:
        rs = renderers if renderers is not None else [m.renderer for m in self.members]
        return fuse([r.render(T @ torch.tensor(m.internal, dtype=T.dtype,
                                               device=T.device))
                     for r, m in zip(rs, self.members)])


@dataclass
class UnitReport:
    name: str
    T: np.ndarray
    scale_radial: float
    scale_axial: float
    scale_ref: float
    visible_px: int
    alone_px: int
    occluded_frac: float
    inside_mask_frac: float
    tilt_deg: float
    twist_deg: float
    yaw_held: bool
    asymmetry: float
    hypothesis: str

    def line(self) -> str:
        yaw = (f"turn held ({self.twist_deg:+6.1f}deg)" if self.yaw_held
               else f"turn {self.twist_deg:+7.1f}deg")
        return (f"unit {self.name:>5}: visible {self.visible_px:6d}/{self.alone_px:6d} "
                f"(occluded {self.occluded_frac * 100:4.1f}%)  "
                f"on-object {self.inside_mask_frac * 100:5.1f}%  "
                f"tilt {self.tilt_deg:4.1f}deg  {yaw}  "
                f"scale across x{self.scale_radial / self.scale_ref:.3f} "
                f"along x{self.scale_axial / self.scale_ref:.3f}")


@dataclass
class JointResult:
    poses: Dict[str, np.ndarray]          # per member fid
    unit_poses: Dict[str, np.ndarray]
    metrics: FitMetrics
    units: List[UnitReport]
    penetration_mm: float
    joint_gap_mm: float
    closest_gap_mm: float
    coaxial_deg: float
    coaxial_offset_mm: float
    rim_gap_mm: float
    rim_radius_diff_mm: float
    rim_plane_deg: float
    order: List[str]
    seconds: float


def load_unit(data_dir, fids: Sequence[str], seed_dir, device: str, K, H: int, W: int,
              init_dir=None, init_cid: Optional[str] = None, n_sample: int = 6000,
              sdf_resolution: int = 48,
              yaw_hold_thresh: float = YAW_HOLD_ASYMMETRY) -> Unit:
    """Build a unit from one or more parts.

    A single part takes its size and attitude from Stage 1. Several take their
    relative placement from an earlier assembly solve — that solve's first-named
    part becomes the unit's frame — so a group already fitted together carries
    forward as one rigid object instead of being re-solved against a capture
    that shows less of it.
    """
    data_dir = Path(data_dir)
    if len(fids) == 1:
        internals = [np.eye(4)]
        z = np.load(Path(seed_dir) / f"pose_{fids[0]}.npz", allow_pickle=True)
        scale, R_solo = float(z["s"]), np.asarray(z["R"], dtype=np.float64)
        # Stage 1 saw this part whole and reported how its rendered width
        # compared with the measured one. A reconstruction that came out thin
        # is measurable there, where nothing occludes it — far better than
        # inferring it later from a seam, where shrinking the *other* part
        # closes the step just as well and just as wrongly.
        #
        # If Stage 1 already acted on that and reshaped the mesh, the width it
        # goes on to report is what is left over, and correcting for it again
        # would be correcting twice. Worse, a second correction here cannot be
        # baked into the mesh, so it survives as a scale that differs across
        # and along the axis — which no rigid-transform format can carry.
        girth = float(z["girth_factor"]) if "girth_factor" in z.files else 1.0
        if abs(girth - 1.0) > 1e-6:
            scale_radial = scale                    # already the right shape
        else:
            wr = float(z["width_ratio"]) if "width_ratio" in z.files else 1.0
            fix = 1.0 / wr if 0.5 < wr < 2.0 else 1.0
            scale_radial = scale * float(np.clip(fix, 1 - MAX_GIRTH_FIX,
                                                 1 + MAX_GIRTH_FIX))
    else:
        if init_dir is None or init_cid is None:
            raise ValueError(f"unit {'+'.join(fids)} needs --init_dir and --init_cid "
                             f"naming the earlier solve that fixed its internals")
        Ts = []
        for f in fids:
            p = Path(init_dir) / f"pose_{f}_in_C{init_cid}.npz"
            if not p.exists():
                raise FileNotFoundError(f"unit member {f}: {p} missing")
            Ts.append(np.asarray(np.load(p, allow_pickle=True)["T"], dtype=np.float64))
        anchor_inv = np.linalg.inv(Ts[0])
        internals = [anchor_inv @ T for T in Ts]
        R_solo, _, scale = geom.decompose(Ts[0])
        scale_radial = scale

    members, pts = [], []
    for f, internal in zip(fids, internals):
        p = _paths(data_dir, f)
        if p is None:
            raise FileNotFoundError(f"part {f}: no mesh under {data_dir}")
        mesh = trimesh.load(p["mesh"], force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        members.append(Member(
            fid=f, internal=np.asarray(internal, dtype=np.float32),
            renderer=MeshRenderer(mesh, K, H, W, device=device),
            sdf=MeshSDF.build(mesh, resolution=sdf_resolution, device=device)))
        pts.append(geom.apply(trimesh_sample(mesh, n_sample, seed=7), internal))
    points = np.concatenate(pts, axis=0)
    axis, spread = geom.central_axis(points, R_solo=R_solo)
    _, asym = geom.symmetry_axis(points)
    return Unit(name="+".join(fids), members=members, scale=scale,
                scale_radial=scale_radial, R_solo=R_solo,
                points=points, axis=axis, rims=geom.rim_circles(points, axis),
                asymmetry=asym, axis_spread=spread,
                yaw_hold_thresh=yaw_hold_thresh)


def orientation_candidates(unit: Unit, max_tilt_deg: float, n_shells: int,
                           n_axes: int, n_yaw: int) -> List[Tuple[str, np.ndarray]]:
    """Attitudes to try: a bounded tipping combined with a free turn.

    The two get different budgets because the capture constrains them
    differently. Tipping a part away from how it stood in its own capture is
    small and shows in the outline. Turning it about its own axis is neither —
    an assembly can be clocked any way, and a part whose outline is a surface of
    revolution barely changes as it turns — so the turn is swept fully, or left
    where Stage 1 had it when nothing can resolve it.
    """
    a_ref = unit.axis_world(unit.R_solo)
    tmp = np.array([1.0, 0, 0]) if abs(a_ref[0]) < 0.9 else np.array([0, 1.0, 0])
    u = geom.normalize(np.cross(a_ref, tmp))
    v = np.cross(a_ref, u)

    yaws = ([0.0] if unit.yaw_held
            else list(np.linspace(0.0, 2 * np.pi, int(n_yaw), endpoint=False)))
    tilts = [(0.0, 0.0)]
    for k in range(1, int(n_shells) + 1):
        ang = math.radians(max_tilt_deg * k / n_shells)
        for j in range(int(n_axes)):
            tilts.append((ang, 2 * np.pi * j / n_axes))

    out = []
    for yi, th in enumerate(yaws):
        R_yaw = geom.rot_about_axis(a_ref, float(th)) @ unit.R_solo
        for ang, phi in tilts:
            if ang == 0.0:
                out.append((f"y{yi:02d}", R_yaw))
                continue
            axis = math.cos(phi) * u + math.sin(phi) * v
            out.append((f"y{yi:02d}t{int(round(math.degrees(ang))):02d}",
                        geom.rot_about_axis(axis, ang) @ R_yaw))
    return out


def _unexplained(frame: Frame, renders: Sequence[Render]) -> np.ndarray:
    """Measured object pixels no placed unit accounts for yet."""
    if not renders:
        return frame.corr.copy()
    covered = torch.stack([r.visible() for r in renders], 0).any(0)
    if covered.shape != frame.corr.shape:
        covered = torch.nn.functional.interpolate(
            covered[None, None].float(), size=frame.corr.shape, mode="nearest")[0, 0]
    return frame.corr & ~(covered.cpu().numpy() > 0.5)


def greedy_order(units: Sequence[Unit], frame: Frame, obs_small: Observation,
                 small: Sequence[List[MeshRenderer]], rng: np.random.Generator,
                 max_tilt_deg: float, n_shells: int, n_axes: int, n_yaw: int,
                 verbose: bool = True):
    order: List[int] = []
    Ts: List[np.ndarray] = []
    names: List[str] = []
    placed: List[Render] = []
    device = obs_small.mask.device

    for step in range(len(units)):
        residual = _unexplained(frame, placed)
        if residual.sum() < 200:
            residual = frame.corr
        z_ref = float(np.median(frame.depth[residual]))
        ys, xs = np.where(residual)
        centroid_px = (float(xs.mean()), float(ys.mean()))

        best = None
        for i, unit in enumerate(units):
            if i in order:
                continue
            for name, R in orientation_candidates(unit, max_tilt_deg, n_shells,
                                                  n_axes, n_yaw):
                T = place(unit.points, R, frame, z_ref, centroid_px, rng,
                          scale=unit.scale, corr=residual)
                if T is None:
                    continue
                T_t = torch.tensor(T, dtype=torch.float32, device=device)
                with torch.no_grad():
                    r = unit.render(T_t, small[i])
                    sc = score(fuse(placed + [r]), obs_small).score
                if best is None or sc > best[0]:
                    best = (sc, i, T, name, r)
        if best is None:
            raise RuntimeError("no placeable orientation remained")
        sc, i, T, name, r = best
        order.append(i); Ts.append(T); names.append(name); placed.append(r)
        if verbose:
            print(f"    place #{step + 1}: unit {units[i].name} [{name}] "
                  f"union score {sc:+.3f}")
    return order, Ts, names


def _mating(units, Ts, pts, log_srs, contact_frac: float = 0.02,
            reach_m: float = 0.02):
    """Interpenetration and the gap left at the join, from one signed distance.

    The negative side is material two units share and gets pushed apart; the
    smallest positive values are the points that ought to be touching and get
    pulled closed. Only the nearest `contact_frac` of samples counts toward the
    join, because parts meet over an interface rather than their whole surface,
    and the pull saturates past `reach_m`, so units nowhere near each other are
    left alone instead of dragged together.
    """
    if len(units) < 2:
        return None, None, float("nan")
    pen_t, gap_t, count, min_gap = None, None, 0, float("inf")
    for i in range(len(units)):
        world = pts[i] @ Ts[i][:3, :3].T + Ts[i][:3, 3]
        for j, uj in enumerate(units):
            if i == j:
                continue
            for m in uj.members:
                if m.sdf is None:
                    continue
                T_m = Ts[j] @ torch.tensor(m.internal, dtype=Ts[j].dtype,
                                           device=Ts[j].device)
                M_inv = torch.linalg.inv(T_m[:3, :3])
                d = signed_distance(world, M_inv, T_m[:3, 3], m.sdf, log_srs[j].exp())
                pen = torch.relu(-d).pow(2).mean()
                k = max(int(len(d) * contact_frac), 20)
                nearest = torch.topk(torch.relu(d), k, largest=False).values
                gap = torch.clamp(nearest, max=reach_m).pow(2).mean()
                pen_t = pen if pen_t is None else pen_t + pen
                gap_t = gap if gap_t is None else gap_t + gap
                min_gap = min(min_gap, float(nearest.min().detach()))
                count += 1
    n = max(count, 1)
    return pen_t / n, gap_t / n, min_gap * 1000.0


def _coaxial(units, qs, Ts, pts):
    """Angle and lateral offset between the units' central axes.

    Not an assumption about the object but a consequence of the joint: two
    circular interfaces can only seat flush if the axes they are turned about
    coincide.
    """
    if len(units) < 2:
        return None, None
    axes, cens = [], []
    for k, (u, q, T) in enumerate(zip(units, qs, Ts)):
        a = geom.quat_to_rot_torch(q) @ torch.tensor(
            u.axis, dtype=torch.float32, device=q.device)
        axes.append(a / (a.norm() + 1e-9))
        cens.append((pts[k] @ T[:3, :3].T + T[:3, 3]).mean(0))
    ang_t, off_t, count = None, None, 0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            ang = 1.0 - torch.abs((axes[i] * axes[j]).sum())
            d = cens[j] - cens[i]
            off = (d - (d * axes[i]).sum() * axes[i]).pow(2).sum()
            ang_t = ang if ang_t is None else ang_t + ang
            off_t = off if off_t is None else off_t + off
            count += 1
    return ang_t / max(count, 1), off_t / max(count, 1)


def choose_rim_pair(units, Ts_np):
    """Which two ends actually meet.

    Nearness alone is not enough to say: a part that has slid inside its
    neighbour can put the wrong ends closest together, and the pairing that
    results is one no assembly could have. Faces that mate look at each other,
    so their outward normals oppose — which rules out two tops or two bottoms
    however close they happen to lie — and among the pairings that remain the
    nearest is taken.
    """
    normals, centres = [], []
    for u, T in zip(units, Ts_np):
        a = geom.normalize(np.asarray(T)[:3, :3] @ u.axis)
        normals.append([-a, a])                       # [bottom, top] face outward
        centres.append([geom.apply(u.rims[e][0][None], T)[0] for e in (0, 1)])
    best = None
    for ei in (0, 1):
        for ej in (0, 1):
            if float(np.dot(normals[0][ei], normals[1][ej])) > -0.2:
                continue                              # not facing one another
            d = float(np.linalg.norm(centres[0][ei] - centres[1][ej]))
            if best is None or d < best[0]:
                best = (d, ei, ej)
    if best is None:      # degenerate placement: fall back to nearest
        best = min(((float(np.linalg.norm(centres[0][i] - centres[1][j])), i, j)
                    for i in (0, 1) for j in (0, 1)), key=lambda x: x[0])
    return best[1], best[2], best[0]


def _rim_match(units, Ts, pair):
    """How far the two mating circles are from being a single circle.

    Being one circle takes three things, not two: the centres must coincide,
    the radii must agree, and the planes must be parallel. Leave the last out
    and the joint hinges — the rims meet at a point and the parts splay apart
    from there, which is what a small angle at the seam turns into several
    millimetres by the far end of a part.

    The tilt is charged as the distance it throws the rim edge off, `r sin(t)`,
    so all three read in metres and one weight covers the lot.
    """
    if len(units) != 2 or pair is None:
        return None, float("nan"), float("nan"), float("nan")
    cs, rs, ns = [], [], []
    for u, T, e in zip(units, Ts, pair):
        c_loc, rad = u.rims[e]
        a_loc = torch.tensor(u.axis, dtype=T.dtype, device=T.device)
        perp = torch.tensor(
            geom.normalize(np.cross(u.axis, [1.0, 0, 0] if abs(u.axis[0]) < 0.9
                                    else [0, 1.0, 0])),
            dtype=T.dtype, device=T.device)
        n = T[:3, :3] @ a_loc
        ns.append(n / (n.norm() + 1e-9))
        cs.append(torch.tensor(c_loc, dtype=T.dtype, device=T.device) @ T[:3, :3].T
                  + T[:3, 3])
        rs.append(rad * (T[:3, :3] @ perp).norm())
    cos = torch.clamp(torch.abs((ns[0] * ns[1]).sum()), -1.0, 1.0)
    sin2 = torch.clamp(1.0 - cos * cos, min=0.0)
    lever = 0.5 * (rs[0] + rs[1])
    dc = (cs[0] - cs[1]).pow(2).sum()
    dr = (rs[0] - rs[1]).pow(2)
    dn = lever.pow(2) * sin2
    return (dc + dr + dn,
            float(dc.detach()) ** 0.5 * 1000,
            float((rs[0] - rs[1]).detach()) * 1000,
            math.degrees(math.acos(min(1.0, max(-1.0, float(cos.detach()))))))


def _profile_step(units, Ts, pts, band_frac: float = 0.12, q: float = 0.9):
    """How far the two units' outer surfaces disagree in radius where they meet.

    Touching is not the same as being flush. Two parts can seat against each
    other along the joint and still leave a step in the profile, because the
    section one ends on is not the section the other begins with — and a step
    is what shows in the picture as a notch at the seam. Requiring the outer
    radius to agree in a band either side of the joint closes it, and the
    across-axis scale is what can give.

    The joint is located, not assumed: it is where the two point sets come
    closest along their shared axis.
    """
    if len(units) != 2:
        return None, float("nan")
    a = Ts[0][:3, :3] @ torch.tensor(units[0].axis, dtype=Ts[0].dtype,
                                     device=Ts[0].device)
    a = a / (a.norm() + 1e-9)
    W = [p @ T[:3, :3].T + T[:3, 3] for p, T in zip(pts, Ts)]
    h = [w @ a for w in W]
    # the seam sits between the two spans, at whichever end they face
    lo = [torch.quantile(x, 0.02) for x in h]
    hi = [torch.quantile(x, 0.98) for x in h]
    if float(h[0].median()) < float(h[1].median()):
        seam = 0.5 * (hi[0] + lo[1])
    else:
        seam = 0.5 * (hi[1] + lo[0])
    span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    band = band_frac * span

    radii = []
    for w, hh in zip(W, h):
        near = (hh - seam).abs() < band
        if int(near.sum()) < 50:
            return None, float("nan")
        d = w[near] - (torch.outer((w[near] @ a), a) + 0.0)
        r = d.norm(dim=1)
        k = max(int(len(r) * (1.0 - q)), 20)
        radii.append(torch.topk(r, k, largest=True).values.mean())
    diff = radii[0] - radii[1]
    return diff.pow(2), float(diff.detach()) * 1000.0


def refine_joint(units: Sequence[Unit], T_init: Sequence[np.ndarray],
                 obs: Observation, iters: int, lr: float, device: str,
                 w_penetration: float, w_contact: float, w_scale: float,
                 w_coaxial: float, w_profile: float, w_scale_radial: float,
                 w_rim: float, lock_scale: bool, max_tilt_deg: float,
                 verbose: bool = True) -> List[np.ndarray]:
    qs, ts, log_srs, log_sas, axes_t = [], [], [], [], []
    for u, T in zip(units, T_init):
        R, t, s = geom.decompose(T)
        qs.append(torch.tensor(geom.quat_from_rot(R), dtype=torch.float32,
                               device=device, requires_grad=True))
        ts.append(torch.tensor(t, dtype=torch.float32, device=device,
                               requires_grad=True))
        log_srs.append(torch.tensor(math.log(max(s * u.scale_radial / max(u.scale, 1e-9),
                                                 1e-9)), dtype=torch.float32,
                                    device=device, requires_grad=not lock_scale))
        log_sas.append(torch.tensor(math.log(max(s, 1e-9)), dtype=torch.float32,
                                    device=device, requires_grad=not lock_scale))
        axes_t.append(torch.tensor(u.axis, dtype=torch.float32, device=device))

    anchors = [torch.tensor(math.log(max(u.scale, 1e-9)), device=device) for u in units]
    anchors_r = [torch.tensor(math.log(max(u.scale_radial, 1e-9)), device=device)
                 for u in units]
    R_refs = [torch.tensor(u.R_solo, dtype=torch.float32, device=device) for u in units]
    ax_ref = [torch.tensor(u.axis_world(u.R_solo), dtype=torch.float32, device=device)
              for u in units]
    pts = [torch.tensor(u.points, dtype=torch.float32, device=device) for u in units]
    tilt_limit = math.radians(max_tilt_deg)
    rim_pair = None
    if len(units) == 2 and all(u.rims for u in units):
        ei, ej, d0 = choose_rim_pair(units, T_init)
        rim_pair = (ei, ej)
        if verbose:
            print(f"    mating circles: {units[0].name} "
                  f"{'top' if ei else 'bottom'} <-> {units[1].name} "
                  f"{'top' if ej else 'bottom'}  (start {d0 * 1000:.1f}mm apart, "
                  f"radii {units[0].rims[ei][1] * units[0].scale * 1000:.1f} / "
                  f"{units[1].rims[ej][1] * units[1].scale * 1000:.1f}mm)")

    groups = [{"params": ts, "lr": lr}, {"params": qs, "lr": lr * 0.5}]
    if not lock_scale:
        groups.append({"params": log_srs + log_sas, "lr": lr * 0.1})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1))

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        Ts = [geom.sim3_aniso_torch(q, t, sr, sa, a)
              for q, t, sr, sa, a in zip(qs, ts, log_srs, log_sas, axes_t)]
        loss = fit_loss(fuse([u.render(T) for u, T in zip(units, Ts)]), obs)
        data = float(loss.detach())

        # The across-axis size is held loosely on purpose: a reconstruction
        # that came out thin is exactly what leaves a step at the seam, and
        # letting the girth move is the only way to close it without changing
        # the height that was measured correctly.
        for sr, sa, ar, aa in zip(log_srs, log_sas, anchors_r, anchors):
            loss = loss + w_scale_radial * (sr - ar).pow(2) + w_scale * (sa - aa).pow(2)

        # Bound the tipping, which the capture resolves; leave the turn about
        # the part's own axis free, or pinned where Stage 1 left it when
        # nothing can resolve it.
        for q, R_ref, a_ref, unit in zip(qs, R_refs, ax_ref, units):
            v = geom.rot_log_torch(geom.quat_to_rot_torch(q) @ R_ref.T)
            twist = (v * a_ref).sum()
            tilt = (v - twist * a_ref).norm()
            loss = loss + W_TILT_LIMIT * torch.relu(tilt - tilt_limit).pow(2)
            if unit.yaw_held:
                loss = loss + W_TWIST_HOLD * twist.pow(2)

        pen, gap, min_gap_mm = _mating(units, Ts, pts, log_srs)
        if pen is not None:
            loss = loss + w_penetration * pen + w_contact * gap
        cang, coff = _coaxial(units, qs, Ts, pts)
        if cang is not None:
            loss = loss + w_coaxial * (W_COAXIAL_ANGLE * cang + W_COAXIAL_OFFSET * coff)
        prof, step_mm = _profile_step(units, Ts, pts)
        if prof is not None and w_profile > 0:
            loss = loss + w_profile * prof
        rim, rim_mm, rad_mm, rim_deg = _rim_match(units, Ts, rim_pair)
        if rim is not None:
            loss = loss + w_rim * rim
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qs + ts + log_srs + log_sas, 1.0)
        opt.step()
        sched.step()
        if verbose and (it % 60 == 0 or it == iters - 1):
            pm = 0.0 if pen is None else math.sqrt(max(float(pen.detach()), 0)) * 1000
            gm = 0.0 if gap is None else math.sqrt(max(float(gap.detach()), 0)) * 1000
            ca = 0.0 if cang is None else math.degrees(math.acos(
                min(1.0, max(0.0, 1.0 - float(cang.detach())))))
            om = 0.0 if coff is None else math.sqrt(max(float(coff.detach()), 0)) * 1000
            print(f"    it={it:3d}  data={data:.4f}  overlap={pm:5.1f}mm  "
                  f"gap={gm:5.1f}mm  coaxial={ca:4.1f}deg/{om:5.1f}mm  "
                  f"rims {rim_mm:5.1f}mm apart, radii {rad_mm:+5.1f}mm, "
                  f"planes {rim_deg:4.2f}deg")

    with torch.no_grad():
        return [geom.sim3_aniso_torch(q, t, sr, sa, a).cpu().numpy().astype(np.float64)
                for q, t, sr, sa, a in zip(qs, ts, log_srs, log_sas, axes_t)]


@torch.no_grad()
def resolve_turn_by_texture(units, Ts_np, obs, device, pair=None, coarse: int = 72,
                            fine_deg: float = 5.0, fine_step: float = 0.5,
                            verbose: bool = True):
    """Settle a held turn against the photograph, once the fit is done.

    Only units whose outline is a surface of revolution are turned, and only
    about the axis running through the mating circle's own centre. Both
    restrictions are what make this safe rather than merely cheap: such a part
    keeps its outline and its depth as it turns, and a circle turned about its
    own centre is the same circle, so the joint that was just solved is left
    exactly where it was. Texture is then the only thing left that can move,
    which is precisely the evidence the fit could not use.

    Scoring is the same objective as everywhere else rather than texture alone.
    For a true surface of revolution the other terms are constant and it comes
    down to texture by itself; where the shape does say something after all,
    they push back instead of being ignored.
    """
    Ts = [np.asarray(T, dtype=np.float64).copy() for T in Ts_np]
    turns = [0.0] * len(units)
    for i, u in enumerate(units):
        if not u.yaw_held:
            if verbose:
                print(f"    unit {u.name}: shape fixes the turn (asym "
                      f"{u.asymmetry:.3f}) — left alone")
            continue
        a = geom.normalize(Ts[i][:3, :3] @ u.axis)
        end = pair[i] if pair is not None else 1
        pivot = geom.apply(u.rims[end][0][None], Ts[i])[0]

        def at(theta_deg):
            A = np.eye(4)
            A[:3, :3] = geom.rot_about_axis(a, math.radians(theta_deg))
            A[:3, 3] = pivot - A[:3, :3] @ pivot
            cand = list(Ts)
            cand[i] = A @ Ts[i]
            rs = [uu.render(torch.tensor(np.asarray(T, np.float32), device=device))
                  for uu, T in zip(units, cand)]
            return score(fuse(rs), obs), A

        base, _ = at(0.0)
        grid = [(at(t)[0].score, t) for t in np.linspace(0, 360, coarse, endpoint=False)]
        best = max(grid)
        fine = [(at(t)[0].score, t) for t in
                np.arange(best[1] - fine_deg, best[1] + fine_deg + 1e-9, fine_step)]
        best = max(fine + [best])
        if best[0] > base.score + 1e-4:
            m, A = at(best[1])
            Ts[i] = A @ Ts[i]
            turns[i] = float(best[1])
            if verbose:
                print(f"    unit {u.name}: turn {best[1]:+7.1f}deg  "
                      f"rgb {base.rgb_err:.3f} -> {m.rgb_err:.3f}  "
                      f"iou {base.iou:.3f} -> {m.iou:.3f}")
        elif verbose:
            print(f"    unit {u.name}: no turn improves it — kept")
    return Ts, turns


@torch.no_grad()
def _reports(units, Ts_np, obs, names):
    device = obs.mask.device
    Ts = [torch.tensor(T, dtype=torch.float32, device=device) for T in Ts_np]
    renders = [u.render(T) for u, T in zip(units, Ts)]
    metrics = score(fuse(renders), obs)

    vis = torch.stack([r.visible() for r in renders], 0)
    depth = torch.stack([r.depth for r in renders], 0)
    front = torch.where(vis, depth, torch.full_like(depth, 1e6)).argmin(0)
    mask_b = obs.mask > 0.5

    out, log_srs, qs = [], [], []
    for i, u in enumerate(units):
        alone, shown = vis[i], vis[i] & (front == i)
        n_alone, n_shown = int(alone.sum()), int(shown.sum())
        M = np.asarray(Ts_np[i])[:3, :3]
        a_hat = geom.normalize(u.axis)
        s_a = float(np.linalg.norm(M @ a_hat))
        perp_local = geom.normalize(np.cross(a_hat, [1.0, 0, 0]
                                             if abs(a_hat[0]) < 0.9 else [0, 1.0, 0]))
        s_r = float(np.linalg.norm(M @ perp_local))
        R_est = M / max(s_r, 1e-12)
        U, _, Vt = np.linalg.svd(R_est)
        R_orth = U @ Vt
        tilt, twist = geom.tilt_twist(R_orth, u.R_solo, u.axis)
        out.append(UnitReport(
            name=u.name, T=Ts_np[i], scale_radial=s_r, scale_axial=s_a,
            scale_ref=u.scale, visible_px=n_shown, alone_px=n_alone,
            occluded_frac=1.0 - n_shown / max(n_alone, 1),
            inside_mask_frac=int((shown & mask_b).sum()) / max(n_shown, 1),
            tilt_deg=tilt, twist_deg=twist, yaw_held=u.yaw_held,
            asymmetry=u.asymmetry, hypothesis=names[i]))
        log_srs.append(torch.tensor(math.log(max(s_r, 1e-9)), device=device))
        qs.append(torch.tensor(geom.quat_from_rot(R_orth), dtype=torch.float32,
                               device=device))

    pts = [torch.tensor(u.points, dtype=torch.float32, device=device) for u in units]
    pen, gap, min_gap_mm = _mating(units, Ts, pts, log_srs)
    cang, coff = _coaxial(units, qs, Ts, pts)
    pair = choose_rim_pair(units, Ts_np)[:2] if (
        len(units) == 2 and all(u.rims for u in units)) else None
    _, rim_mm, rad_mm, rim_deg = _rim_match(units, Ts, pair)
    return (out, rim_mm, rad_mm, rim_deg,
            0.0 if pen is None else math.sqrt(max(float(pen), 0.0)) * 1000,
            0.0 if gap is None else math.sqrt(max(float(gap), 0.0)) * 1000,
            min_gap_mm,
            0.0 if cang is None else math.degrees(math.acos(
                min(1.0, max(0.0, 1.0 - float(cang))))),
            0.0 if coff is None else math.sqrt(max(float(coff), 0.0)) * 1000,
            metrics)


def align_assembly(frame: Frame, units: List[Unit], *, device: str = "cuda",
                   screen_scale: float = 0.25, iters: int = 300, lr: float = 0.01,
                   w_penetration: float = W_PENETRATION, w_contact: float = W_CONTACT,
                   w_scale: float = W_SCALE_ANCHOR, w_coaxial: float = 0.1,
                   w_profile: float = 0.0, w_scale_radial: float = 3.0,
                   w_rim: float = W_RIM,
                   lock_scale: bool = False, max_tilt_deg: float = MAX_TILT_DEG,
                   n_shells: int = 2, n_axes: int = 8, n_yaw: int = 24,
                   turn_by_texture: bool = True,
                   seed: int = 7, verbose: bool = True) -> JointResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    obs = Observation.build(frame.mask, frame.depth, frame.corr, frame.rgb, device)
    obs_small = obs.downscale(screen_scale)
    small = [[m.renderer.at(frame.K, screen_scale) for m in u.members] for u in units]

    if verbose:
        for u in units:
            held = ("turn HELD at Stage 1 (surface of revolution)" if u.yaw_held
                    else "turn searched")
            print(f"  unit {u.name}: s={u.scale:.4f} (girth x{u.scale_radial / u.scale:.3f} "
                  f"from Stage 1 width)  axis={np.round(u.axis, 3).tolist()}  "
                  f"asym={u.asymmetry:.3f}  {held}")
        print(f"  tilt budget {max_tilt_deg:g}deg from the Stage 1 attitude")

    order, Ts_ord, names_ord = greedy_order(
        units, frame, obs_small, small, rng, max_tilt_deg, n_shells, n_axes,
        n_yaw, verbose)
    ordered = [units[i] for i in order]
    T_ref = refine_joint(ordered, Ts_ord, obs, iters, lr, device, w_penetration,
                         w_contact, w_scale, w_coaxial, w_profile, w_scale_radial,
                         w_rim, lock_scale, max_tilt_deg, verbose)
    T_by_name = {ordered[k].name: T_ref[k] for k in range(len(order))}
    name_by_unit = {ordered[k].name: names_ord[k] for k in range(len(order))}

    Ts_np = [T_by_name[u.name] for u in units]
    if turn_by_texture:
        if verbose:
            print("  settling each turn about its own axis against the texture:")
        pair = (choose_rim_pair(units, Ts_np)[:2]
                if len(units) == 2 and all(u.rims for u in units) else None)
        Ts_np, _ = resolve_turn_by_texture(units, Ts_np, obs, device, pair=pair,
                                           verbose=verbose)
        T_by_name = {u.name: T for u, T in zip(units, Ts_np)}
    reports, rim_mm, rad_mm, rim_deg, pen_mm, gap_mm, min_gap_mm, cax_deg, cax_mm, metrics = _reports(
        units, Ts_np, obs, [name_by_unit[u.name] for u in units])
    poses = {m.fid: T @ m.internal.astype(np.float64)
             for u, T in zip(units, Ts_np) for m in u.members}
    return JointResult(poses=poses, unit_poses=T_by_name, metrics=metrics,
                       units=reports, penetration_mm=pen_mm, joint_gap_mm=gap_mm,
                       closest_gap_mm=min_gap_mm, coaxial_deg=cax_deg,
                       coaxial_offset_mm=cax_mm, rim_gap_mm=rim_mm,
                       rim_radius_diff_mm=rad_mm, rim_plane_deg=rim_deg,
                       order=[units[i].name for i in order],
                       seconds=time.perf_counter() - t0)
