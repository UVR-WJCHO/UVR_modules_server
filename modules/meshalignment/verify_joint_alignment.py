"""Multi-view verification of joint two-part alignment results.

Generates three independent checks for a given combined frame:

1. **Top-down view**: renders the two aligned part meshes from a synthetic
   camera placed directly above the object (looking along -n_table). If the
   stacking constraint produced a true vertical assembly, both parts should
   appear concentric (their silhouettes overlap) in the top-down image.

2. **Side view**: renders from a synthetic camera 90 degrees around the
   table normal from the real camera. Makes the object's tilt directly
   visible -- if the alignment correctly represents a slightly-tilted
   object, the rendered side view should look upright/tilted accordingly.

3. **Combined-mesh comparison**: renders the single-frame combined mesh
   (`mesh_01.glb` / `mesh_34.glb` from its own single alignment) in the
   same original view and side-by-side with our union of part meshes.
   Convergence between these two independent solutions cross-validates
   our part decomposition.

Inputs (auto-found):
    --data_dir         frame data (rgb, depth, mask, intrinsics, meshes)
    --combined_fid     '01' or '34'
    --part_fids        '0,1' or '3,4'
    --joint_dir        directory with pose_part_<f>_in_C<id>.npz
    --combined_pose    pose_<id>.npz from single-alignment (for comparison)
    --output_dir       where the verification PNGs go

Outputs:
    verify_topdown_<id>.png       (REAL view | TOP-DOWN view of union)
    verify_sideview_<id>.png      (REAL view | SIDE view of union)
    verify_combined_<id>.png      (our union | combined-mesh single | DIFF)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image

_BASE = Path("/mnt/d/metaobj") if Path("/mnt/d/metaobj").exists() else Path("D:/metaobj")
sys.path.insert(0, str(_BASE))

from auto_align_mesh_rgbd_scale_locked import (
    NVTexturedRenderer, load_saved_K, make_mask_from_provided,
    estimate_table_normal_from_depth, AlignConfig,
)


# ----------------------------- view setup -----------------------------------

def build_view_matrix(
    object_center: np.ndarray,
    n_table: np.ndarray,
    fwd_dir: np.ndarray,
    distance: float,
    image_down_ref: np.ndarray,
) -> np.ndarray:
    """Synthetic camera at `object_center - distance * fwd_dir`, looking in
    direction `fwd_dir`. `image_down_ref` is projected onto the plane
    perpendicular to `fwd_dir` to provide a stable +y axis (image-down).
    Returns a 4x4 transform that maps original-camera-frame points into the
    synthetic camera frame."""
    fwd = fwd_dir / (np.linalg.norm(fwd_dir) + 1e-9)
    cam_pos = object_center - distance * fwd
    # image-down: pick a direction perpendicular to fwd, closest to image_down_ref
    ref = image_down_ref - image_down_ref.dot(fwd) * fwd
    ref_n = np.linalg.norm(ref)
    if ref_n < 1e-6:
        # fall back to a default
        ref = np.array([0.0, 1.0, 0.0]) - np.array([0.0, 1.0, 0.0]).dot(fwd) * fwd
        ref_n = np.linalg.norm(ref)
    y_synth = ref / ref_n
    x_synth = np.cross(y_synth, fwd)
    x_synth = x_synth / (np.linalg.norm(x_synth) + 1e-9)
    R = np.stack([x_synth, y_synth, fwd], axis=0)
    t = -R @ cam_pos
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def render_mesh_union(
    renderers: list,
    poses_T: list,
    view_M: np.ndarray,
    device: str,
) -> tuple:
    """Render multiple meshes under a synthetic view and produce a z-buffered
    union RGB + silhouette + depth."""
    H, W = renderers[0].H, renderers[0].W
    union_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    union_depth = np.full((H, W), np.inf, dtype=np.float32)
    for r, T in zip(renderers, poses_T):
        new_T = view_M @ T
        with torch.no_grad():
            sil_t, depth_t, rgb_t = r.render_T(new_T)
        sil = (sil_t > 0.5).cpu().numpy()
        depth = depth_t.cpu().numpy()
        rgb_img = (rgb_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        nearer = sil & (depth > 0.05) & (depth < union_depth)
        union_rgb[nearer] = rgb_img[nearer]
        union_depth[nearer] = depth[nearer]
    union_sil = union_depth < np.inf
    union_depth[~union_sil] = 0.0
    return union_rgb, union_sil, union_depth


# ----------------------------- viz helpers ----------------------------------

def label(img, text, y=28, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2)
    return img


def crop_to_content(img: np.ndarray, sil: np.ndarray, pad: int = 30) -> np.ndarray:
    if not sil.any():
        return img
    ys, xs = np.where(sil)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, img.shape[1])
    return img[y0:y1, x0:x1]


# ----------------------------- main ------------------------------------------

def run(args):
    data_dir = Path(args.data_dir)
    joint_dir = Path(args.joint_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    fid = args.combined_fid
    part_fids = [p.strip() for p in args.part_fids.split(",")]

    rgb = np.array(Image.open(data_dir / f"rgb_{fid}.png").convert("RGB"))
    masked = np.array(Image.open(data_dir / f"rgb_masked_{fid}.png"))
    depth = np.load(data_dir / f"depth_{fid}.npy").astype(np.float32)
    K = load_saved_K(data_dir, fid)
    H, W = depth.shape
    sil_mask = make_mask_from_provided(masked)

    cfg = AlignConfig()
    n_table, _ = estimate_table_normal_from_depth(
        depth, sil_mask, K,
        iters=cfg.table_ransac_iters,
        thresh_m=cfg.table_plane_thresh_m, seed=0,
    )
    if n_table is None:
        n_table = np.array([0.0, -1.0, 0.0])
    print(f"[setup] n_table = {n_table.round(3).tolist()}")

    # Load part meshes and poses
    meshes = []
    poses_T = []
    for pf in part_fids:
        m = trimesh.load(data_dir / f"mesh_{pf}.glb", force="mesh")
        d = np.load(joint_dir / f"pose_part_{pf}_in_C{fid}.npz", allow_pickle=True)
        T = np.asarray(d["T"], dtype=np.float64)
        meshes.append(m); poses_T.append(T)
        s = float(d["s"])
        t_vec = np.asarray(d["t"]).reshape(3)
        print(f"[load] part_{pf}  s={s:.4f}  t={t_vec.round(3).tolist()}  "
              f"t.z={t_vec[2]*1000:.0f}mm")
    renderers = [NVTexturedRenderer(m, K, H, W, device=device) for m in meshes]

    # Object center in camera frame: mean of mesh union vertices
    all_verts_cam = []
    for m, T in zip(meshes, poses_T):
        v = np.asarray(m.vertices, dtype=np.float64)
        v_cam = (T[:3, :3] @ v.T).T + T[:3, 3]
        all_verts_cam.append(v_cam)
    union_verts_cam = np.concatenate(all_verts_cam, axis=0)
    object_center = union_verts_cam.mean(axis=0)
    print(f"[setup] object_center (camera frame) = {object_center.round(3).tolist()}m")

    # ====================== View 1: REAL CAMERA (sanity) ====================
    real_view_M = np.eye(4)
    rgb_orig, sil_orig, depth_orig = render_mesh_union(
        renderers, poses_T, real_view_M, device)
    # Overlay onto real
    real_overlay = rgb.copy()
    real_overlay[sil_orig] = rgb_orig[sil_orig]

    # ====================== View 2: TOP-DOWN ================================
    # Camera placed at object_center + d * n_table (above the object), looking
    # in -n_table direction (down toward table).
    d_above = 0.45  # distance above object center
    # image-down ref = projection of original camera +z (carrying viewer's
    # forward) onto the table plane -> consistent orientation across frames
    cam_z = np.array([0.0, 0.0, 1.0])
    cam_z_horiz = cam_z - cam_z.dot(n_table / np.linalg.norm(n_table)) * (n_table / np.linalg.norm(n_table))
    topdown_M = build_view_matrix(
        object_center=object_center,
        n_table=n_table,
        fwd_dir=-n_table,
        distance=d_above,
        image_down_ref=cam_z_horiz,
    )
    rgb_td, sil_td, depth_td = render_mesh_union(
        renderers, poses_T, topdown_M, device)

    # ====================== View 3: SIDE VIEW ===============================
    # Pick a horizontal direction (perpendicular to n_table) and put camera
    # there at fixed distance. Used to verify tilt and stacking from the side.
    # Horizontal "forward" = cross(n_table, camera_right_axis_in_world)
    cam_x = np.array([1.0, 0.0, 0.0])
    side_fwd = np.cross(n_table, cam_x)
    side_fwd = side_fwd / (np.linalg.norm(side_fwd) + 1e-9)
    # if degenerate, pick another
    if np.linalg.norm(side_fwd) < 1e-3:
        cam_x = np.array([0.0, 0.0, 1.0])
        side_fwd = np.cross(n_table, cam_x)
        side_fwd = side_fwd / (np.linalg.norm(side_fwd) + 1e-9)
    d_side = 0.55
    side_M = build_view_matrix(
        object_center=object_center,
        n_table=n_table,
        fwd_dir=-side_fwd,        # face the object
        distance=d_side,
        image_down_ref=-n_table,   # image-down = world-down (-n_table)
    )
    rgb_side, sil_side, depth_side = render_mesh_union(
        renderers, poses_T, side_M, device)

    # ====================== View 4: COMBINED MESH ===========================
    combined_mesh_path = data_dir / f"mesh_{fid}.glb"
    has_combined = (args.combined_pose
                    and Path(args.combined_pose).exists()
                    and combined_mesh_path.exists())
    if has_combined:
        m_c = trimesh.load(combined_mesh_path, force="mesh")
        d_c = np.load(args.combined_pose, allow_pickle=True)
        T_c = np.asarray(d_c["T"], dtype=np.float64) if "T" in d_c.files \
              else np.eye(4)
        if "T" not in d_c.files:
            R_c = np.asarray(d_c["R"]); t_c = np.asarray(d_c["t"]).reshape(3)
            s_c = float(d_c["s"])
            T_c = np.eye(4)
            T_c[:3, :3] = s_c * R_c
            T_c[:3, 3] = t_c
        r_c = NVTexturedRenderer(m_c, K, H, W, device=device)
        with torch.no_grad():
            sil_ct, depth_ct, rgb_ct = r_c.render_T(T_c)
        sil_c = (sil_ct > 0.5).cpu().numpy()
        depth_c = depth_ct.cpu().numpy()
        rgb_c = (rgb_ct.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        combined_overlay = rgb.copy()
        combined_overlay[sil_c] = rgb_c[sil_c]

    # ====================== SAVE PANELS =====================================

    # Panel: REAL view sanity + top-down + side
    # First crop everything to the union mask region for the original view,
    # else to the silhouette region for synthetic views.
    h_panel = max(rgb.shape[0], rgb_td.shape[0], rgb_side.shape[0])
    def to_panel(img, sil, title):
        c = crop_to_content(img, sil, pad=30)
        if c.shape[0] != h_panel:
            scale = h_panel / c.shape[0]
            new_w = max(1, int(c.shape[1] * scale))
            c = cv2.resize(c, (new_w, h_panel))
        label(c, title)
        return c

    # === verify_topdown ===
    real_p = to_panel(real_overlay, sil_orig | sil_mask, "REAL view (overlay)")
    td_p = to_panel(rgb_td, sil_td, "TOP-DOWN (along n_table)")
    # outline of each individual mesh on top-down for clarity
    td_individual = np.zeros_like(rgb_td)
    colors = [(255, 80, 80), (80, 255, 255)]  # red=lower, yellow=upper
    # We also collect each mesh's silhouette centroid (image-space) so we can
    # mark it and measure the horizontal axis offset.
    centroids_pix = []
    sils_each = []
    for r, T, col in zip(renderers, poses_T, colors):
        new_T = topdown_M @ T
        with torch.no_grad():
            sil_t, _, _ = r.render_T(new_T)
        s_arr = (sil_t > 0.5).cpu().numpy()
        sils_each.append(s_arr)
        edge = cv2.dilate(s_arr.astype(np.uint8), np.ones((3, 3), np.uint8)) - s_arr.astype(np.uint8)
        td_individual[s_arr] = (np.array(col) * 0.4).astype(np.uint8)
        td_individual[edge > 0] = col
        if s_arr.any():
            ys, xs = np.where(s_arr)
            centroids_pix.append((float(xs.mean()), float(ys.mean())))
        else:
            centroids_pix.append(None)

    # Compute the *metric* horizontal offset between the two meshes' vertical
    # axes in the original camera frame. Each mesh's center (mean of vertices
    # transformed by its pose) projected onto the plane perpendicular to
    # n_table gives that mesh's horizontal location on the table. The
    # distance between the two horizontal points is the axis offset.
    centers_cam = []
    for m, T in zip(meshes, poses_T):
        v = np.asarray(m.vertices, dtype=np.float64)
        v_cam = (T[:3, :3] @ v.T).T + T[:3, 3]
        centers_cam.append(v_cam.mean(axis=0))
    n_unit = n_table / (np.linalg.norm(n_table) + 1e-9)
    diff = centers_cam[0] - centers_cam[1]
    diff_horiz = diff - diff.dot(n_unit) * n_unit
    horiz_offset_mm = float(np.linalg.norm(diff_horiz)) * 1000.0
    vert_offset_mm = float(abs(diff.dot(n_unit))) * 1000.0
    print(f"[topdown] horizontal axis offset = {horiz_offset_mm:.1f} mm  "
          f"(vertical separation = {vert_offset_mm:.1f} mm)")

    # Make a clean axis-check panel: silhouettes + centroid markers + distance
    axis_panel = np.zeros_like(rgb_td)
    # union background = very dark grey
    axis_panel[sil_td] = (28, 28, 28)
    # silhouettes with alpha-like fill
    for s_arr, col in zip(sils_each, colors):
        axis_panel[s_arr] = (np.array(col) * 0.35).astype(np.uint8)
    # outlines
    for s_arr, col in zip(sils_each, colors):
        ed = cv2.dilate(s_arr.astype(np.uint8), np.ones((3, 3), np.uint8)) - s_arr.astype(np.uint8)
        axis_panel[ed > 0] = col
    # draw the line between centroids and a cross at each
    if all(c is not None for c in centroids_pix):
        (xa, ya), (xb, yb) = centroids_pix
        cv2.line(axis_panel, (int(xa), int(ya)), (int(xb), int(yb)),
                 (255, 255, 255), 1, cv2.LINE_AA)
        for (x, y), col in zip(centroids_pix, colors):
            # outer black halo
            cv2.drawMarker(axis_panel, (int(x), int(y)), (0, 0, 0),
                           cv2.MARKER_CROSS, 16, 3, cv2.LINE_AA)
            cv2.drawMarker(axis_panel, (int(x), int(y)), col,
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(axis_panel, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
    # crop to silhouette area and add labels
    axis_panel_c = crop_to_content(axis_panel, sil_td, pad=30)
    if axis_panel_c.shape[0] != h_panel:
        scale = h_panel / axis_panel_c.shape[0]
        new_w = max(1, int(axis_panel_c.shape[1] * scale))
        axis_panel_c = cv2.resize(axis_panel_c, (new_w, h_panel))
    label(axis_panel_c, "VERTICAL-AXIS CHECK (top-down)", y=24, scale=0.55)
    label(axis_panel_c, f"horizontal axis offset = {horiz_offset_mm:.1f} mm",
          y=axis_panel_c.shape[0] - 50, scale=0.5,
          color=(255, 255, 255))
    label(axis_panel_c, f"vertical separation = {vert_offset_mm:.1f} mm",
          y=axis_panel_c.shape[0] - 28, scale=0.5,
          color=(200, 200, 200))
    label(axis_panel_c, "+ = mesh centroid  (line = axis displacement)",
          y=axis_panel_c.shape[0] - 8, scale=0.4,
          color=(180, 180, 180))

    td_outlines = to_panel(td_individual, sil_td,
                            "TOP-DOWN outlines (red=lower yellow=upper)")
    pan1 = np.concatenate([real_p, td_p, td_outlines, axis_panel_c], axis=1)
    Image.fromarray(pan1).save(out_dir / f"verify_topdown_{fid}.png")
    print(f"[saved] verify_topdown_{fid}.png  "
          f"(axis offset {horiz_offset_mm:.1f}mm)")

    # === verify_sideview ===
    side_p = to_panel(rgb_side, sil_side, "SIDE view (~perpendicular)")
    # outlined side
    side_individual = np.zeros_like(rgb_side)
    for r, T, col in zip(renderers, poses_T, colors):
        new_T = side_M @ T
        with torch.no_grad():
            sil_t, _, _ = r.render_T(new_T)
        s_arr = (sil_t > 0.5).cpu().numpy()
        edge = cv2.dilate(s_arr.astype(np.uint8), np.ones((3, 3), np.uint8)) - s_arr.astype(np.uint8)
        side_individual[s_arr] = (np.array(col) * 0.4).astype(np.uint8)
        side_individual[edge > 0] = col
    side_outlines = to_panel(side_individual, sil_side, "SIDE outlines (red=lower yellow=upper)")
    pan2 = np.concatenate([real_p, side_p, side_outlines], axis=1)
    Image.fromarray(pan2).save(out_dir / f"verify_sideview_{fid}.png")
    print(f"[saved] verify_sideview_{fid}.png")

    # === verify_combined ===
    if has_combined:
        # IoU between our union silhouette and combined-mesh silhouette
        u = sil_orig | sil_c
        i = sil_orig & sil_c
        iou = (i.sum() / max(u.sum(), 1)) * 100
        # depth diff at intersection
        both_d = sil_orig & sil_c & (depth_orig > 0) & (depth_c > 0)
        if int(both_d.sum()) > 50:
            d_med = float(np.median(np.abs(depth_orig[both_d] - depth_c[both_d]))) * 1000
        else:
            d_med = 0.0
        real_oc = to_panel(rgb, sil_mask, "REAL")
        ours_oc = to_panel(real_overlay, sil_orig | sil_mask, "OUR union")
        comb_oc = to_panel(combined_overlay, sil_c | sil_mask, "COMBINED mesh single")
        # diff panel: show pixels covered by exactly one of the two
        diff_panel = rgb.copy()
        only_ours = sil_orig & ~sil_c
        only_comb = sil_c & ~sil_orig
        diff_panel[only_ours] = (255, 80, 80)
        diff_panel[only_comb] = (80, 255, 255)
        diff_oc = to_panel(diff_panel, sil_orig | sil_c | sil_mask,
                           f"DIFF  IoU={iou:.1f}%  |dz|med={d_med:.0f}mm")
        label(diff_oc, "red=only ours  yellow=only combined", y=50, scale=0.45)
        pan3 = np.concatenate([real_oc, ours_oc, comb_oc, diff_oc], axis=1)
        Image.fromarray(pan3).save(out_dir / f"verify_combined_{fid}.png")
        print(f"[saved] verify_combined_{fid}.png  "
              f"(IoU(ours, combined) = {iou:.1f}%, depth diff median {d_med:.0f}mm)")
    else:
        print("[skip] combined mesh comparison (need --combined_pose + mesh_<fid>.glb)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--combined_fid", required=True)
    ap.add_argument("--part_fids", required=True,
                    help="comma separated, e.g. '0,1' or '3,4'")
    ap.add_argument("--joint_dir", required=True,
                    help="dir with pose_part_<f>_in_C<id>.npz")
    ap.add_argument("--combined_pose", default=None,
                    help="single-mesh combined pose npz (e.g. pose_01.npz)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
