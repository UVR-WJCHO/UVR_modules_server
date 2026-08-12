"""Result images. Nothing here feeds back into the fit."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .frames import Frame
from .render import MeshRenderer
from .score import FitMetrics

MASK_COLOUR = (255, 64, 64)      # observed foreground outline
RENDER_COLOUR = (64, 255, 64)    # rendered outline


def _outline(img: np.ndarray, mask: np.ndarray, colour, thickness=2) -> None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, colour, thickness)


def _crop(img: np.ndarray, mask: np.ndarray, pad: int = 40) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return img
    H, W = mask.shape
    return img[max(int(ys.min()) - pad, 0):min(int(ys.max()) + pad, H),
               max(int(xs.min()) - pad, 0):min(int(xs.max()) + pad, W)]


def save_compare(path: Path, frame: Frame, renderer: MeshRenderer,
                 T: np.ndarray, metrics: FitMetrics, title: str = "") -> None:
    """REAL | RENDERED | OVERLAY, cropped to the object."""
    r = renderer.render_np(T)
    sil = r.visible().cpu().numpy()
    rgb_r = (r.rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    both = frame.mask | sil
    real = _crop(frame.rgb.copy(), both)

    rendered = (frame.rgb.astype(np.float32) * 0.25).astype(np.uint8)
    rendered[sil] = rgb_r[sil]
    rendered = _crop(rendered, both)

    overlay = frame.rgb.copy()
    overlay[sil] = (0.45 * overlay[sil] + 0.55 * rgb_r[sil]).astype(np.uint8)
    _outline(overlay, frame.mask, MASK_COLOUR)
    _outline(overlay, sil, RENDER_COLOUR)
    overlay = _crop(overlay, both)

    cv2.putText(real, "REAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(rendered, "RENDERED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)
    cv2.putText(overlay, "red=observed  green=rendered", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(overlay, metrics.line(), (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if title:
        cv2.putText(overlay, title, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 255), 1)

    h = min(real.shape[0], rendered.shape[0], overlay.shape[0])
    panel = np.concatenate([real[:h], rendered[:h], overlay[:h]], axis=1)
    Image.fromarray(panel).save(path)


# distinct outline colour per part, in placement order
PART_COLOURS = [(255, 64, 64), (255, 220, 60), (80, 180, 255), (140, 255, 140),
                (255, 130, 240)]


def save_assembly(path: Path, frame: Frame, units, poses, metrics) -> None:
    """REAL | ASSEMBLED | per-part contributions, cropped to the object.

    The assembled panel shows what the parts jointly look like through a shared
    z-buffer, which is what the fit is actually judged on. Each part panel dims
    everything except the pixels where that part ends up in front, so a part
    that the assembly hides shows up as nearly empty — the honest picture of how
    much this capture could constrain it.
    """
    import torch
    from .render import fuse

    import numpy as _np
    dev = units[0].members[0].renderer.verts.device
    renders = [u.render(torch.tensor(_np.asarray(T, _np.float32), device=dev))
               for u, T in zip(units, poses)]
    united = fuse(renders)
    sil_u = united.visible().cpu().numpy()
    rgb_u = (united.rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    vis = torch.stack([r.visible() for r in renders], 0)
    depth = torch.stack([r.depth for r in renders], 0)
    front = torch.where(vis, depth, torch.full_like(depth, 1e6)).argmin(0).cpu().numpy()

    both = frame.mask | sil_u
    panels = [_crop(frame.rgb.copy(), both)]
    cv2.putText(panels[0], "REAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)

    assembled = frame.rgb.copy()
    assembled[sil_u] = rgb_u[sil_u]
    for i, _u in enumerate(units):
        _outline(assembled, sil_u & (front == i), PART_COLOURS[i % len(PART_COLOURS)])
    assembled = _crop(assembled, both)
    cv2.putText(assembled, "ASSEMBLED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 200, 100), 2)
    cv2.putText(assembled, metrics.line(), (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    panels.append(assembled)

    for i, u in enumerate(units):
        shown = sil_u & (front == i)
        img = (frame.rgb.astype(np.float32) * 0.25).astype(np.uint8)
        img[shown] = rgb_u[shown]
        _outline(img, shown, PART_COLOURS[i % len(PART_COLOURS)])
        img = _crop(img, both)
        cv2.putText(img, f"unit {u.name}  {int(shown.sum())}px visible", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    PART_COLOURS[i % len(PART_COLOURS)], 2)
        panels.append(img)

    h = min(x.shape[0] for x in panels)
    Image.fromarray(np.concatenate([x[:h] for x in panels], axis=1)).save(path)


def save_orbit_views(path: Path, frame: Frame, units, poses, rim_pair=None,
                     angles=(0, 45, 90, 135)) -> None:
    """The assembly alone, turned about its own axis, on a plain ground.

    One capture shows one side. A joint that reads as flush from there can be
    splayed in the direction the camera never saw, and nothing in the fit would
    say so, because nothing was measured there. Turning the solved assembly and
    looking is the only check on the half of it the data could not constrain.

    The mating circles are drawn where they now sit, one colour each, so a
    joint that has hinged open shows as two ellipses instead of one.
    """
    import torch
    from . import geom
    from .render import fuse

    dev = units[0].members[0].renderer.verts.device
    H, W = frame.shape
    K = frame.K
    axis = geom.normalize(np.asarray(poses[0])[:3, :3] @ units[0].axis)
    centre = np.concatenate([geom.apply(u.points, T)
                             for u, T in zip(units, poses)]).mean(axis=0)

    def project(p):
        return (int(K[0, 0] * p[0] / p[2] + K[0, 2]),
                int(K[1, 1] * p[1] / p[2] + K[1, 2]))

    panels = []
    for deg in angles:
        V = np.eye(4)
        V[:3, :3] = geom.rot_about_axis(axis, np.radians(deg))
        V[:3, 3] = centre - V[:3, :3] @ centre
        rs = [u.render(torch.tensor(np.asarray(V @ T, np.float32), device=dev))
              for u, T in zip(units, poses)]
        united = fuse(rs)
        sil = united.visible().cpu().numpy()
        rgb = (united.rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        vis = torch.stack([r.visible() for r in rs], 0)
        dep = torch.stack([r.depth for r in rs], 0)
        front = torch.where(vis, dep, torch.full_like(dep, 1e6)).argmin(0).cpu().numpy()

        img = np.full((H, W, 3), 38, np.uint8)
        img[sil] = rgb[sil]
        for i in range(len(units)):
            _outline(img, sil & (front == i), PART_COLOURS[i % len(PART_COLOURS)])
        if rim_pair is not None:
            for i, (u, T, e) in enumerate(zip(units, poses, rim_pair)):
                c_loc, rad = u.rims[e]
                M = (V @ T)[:3, :3]
                cw = geom.apply(c_loc[None], V @ T)[0]
                aw = geom.normalize(M @ u.axis)
                perp = geom.normalize(np.cross(u.axis, [1.0, 0, 0]
                                               if abs(u.axis[0]) < 0.9 else [0, 1.0, 0]))
                sc = float(np.linalg.norm(M @ perp))
                uu = geom.normalize(np.cross(aw, [1.0, 0, 0]
                                             if abs(aw[0]) < 0.9 else [0, 1.0, 0]))
                vv = np.cross(aw, uu)
                pts = [project(cw + rad * sc * (np.cos(t) * uu + np.sin(t) * vv))
                       for t in np.linspace(0, 2 * np.pi, 72)
                       if (cw + rad * sc * (np.cos(t) * uu + np.sin(t) * vv))[2] > 0.02]
                for k in range(len(pts) - 1):
                    cv2.line(img, pts[k], pts[k + 1],
                             (0, 255, 255) if i == 0 else (255, 0, 255), 2)
        ys, xs = np.where(sil)
        crop = img[max(int(ys.min()) - 40, 0):min(int(ys.max()) + 40, H),
                   max(int(xs.min()) - 60, 0):min(int(xs.max()) + 60, W)].copy()
        cv2.putText(crop, f"{deg} deg" + (" (as captured)" if deg == 0 else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(crop)
    h = min(p.shape[0] for p in panels)
    Image.fromarray(np.concatenate([p[:h] for p in panels], axis=1)).save(path)
