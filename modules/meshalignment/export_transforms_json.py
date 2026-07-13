"""Export alignment poses to the transforms.json format used by the user.

Format (per part):
  {
    "name": "<identifier>",
    "translation": [tx, ty, tz],
    "rotation_euler_degrees": [rx, ry, rz],   # 'xyz' extrinsic, degrees
    "rotation_quaternion": [w, x, y, z],
    "scale": [sx, sy, sz]                      # uniform: [s, s, s]
  }

Reads every pose_{fid}.npz in --pose_dir and writes a single transforms.json.

Usage:
  python export_transforms_json.py --pose_dir D:/metaobj/results_2606_v3
  python export_transforms_json.py \
      --pose_dir D:/metaobj/results_2606_v3 \
      --output  D:/metaobj/results_2606_v3/transforms.json \
      --name_prefix stage_ --name_suffix _start
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


# Coordinate-frame change matrix: OpenCV camera -> Blender world
# OpenCV: +X right, +Y down, +Z forward
# Blender: +X right, +Y forward, +Z up
# mapping: x_b = x_o,  y_b = z_o,  z_b = -y_o  (rotation by -90 deg around X)
M_CV_TO_BLENDER = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def part_dict_from_pose(
    name: str,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
    target_frame: str = "opencv",
    euler_order: str = "xyz",
    include_quat: bool = True,
) -> dict:
    """Convert a single similarity pose (R, t, s) into one transforms.json
    part entry. `R/t` are the OpenCV-camera pose of the mesh
    (mesh point in camera = s * R @ p_mesh + t).

    target_frame='blender' changes the basis of the camera frame to Blender
    world (camera at origin), matching the metaobj_wrapper transforms.json.
    Rotation conjugates under the basis change: R' = M R M^T; t' = M t.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    s = float(s)
    if target_frame == "blender":
        R = M_CV_TO_BLENDER @ R @ M_CV_TO_BLENDER.T
        t = M_CV_TO_BLENDER @ t
    t = t.reshape(3)

    rot = Rotation.from_matrix(R)
    euler = rot.as_euler(euler_order, degrees=True)
    part = {
        "name": name,
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "rotation_euler_degrees": [float(euler[0]), float(euler[1]),
                                   float(euler[2])],
    }
    if include_quat:
        qxyzw = rot.as_quat()  # scipy order (x, y, z, w)
        part["rotation_quaternion"] = [float(qxyzw[3]), float(qxyzw[0]),
                                       float(qxyzw[1]), float(qxyzw[2])]
    part["scale"] = [s, s, s]
    return part


def fid_sort_key(fid: str):
    """Sort by leading numeric component first, then by full string.
    Order: '0' < '01' < '0_check' < '3' < '34' < '4' < '4-0' < '4-1'."""
    m = re.match(r"^(\d+)", fid)
    head = int(m.group(1)) if m else 1_000_000
    return (head, fid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", required=True, type=str,
                    help="folder containing pose_{fid}.npz files")
    ap.add_argument("--output", default=None, type=str,
                    help="output transforms.json path "
                         "(default: <pose_dir>/transforms.json)")
    ap.add_argument("--name_prefix", default="", type=str,
                    help="prefix prepended to each fid in 'name' field "
                         "(e.g. 'stage_' produces 'stage_0')")
    ap.add_argument("--name_suffix", default="", type=str,
                    help="suffix appended to each fid (e.g. '_start')")
    ap.add_argument("--name_map", default=None, type=str,
                    help="optional comma-separated mapping like "
                         "'0:stage_0_start,3:stage_3_start,4:stage_4_start'. "
                         "overrides prefix/suffix for listed fids; "
                         "unlisted fids fall back to prefix+fid+suffix.")
    ap.add_argument("--frame_ids", default=None, type=str,
                    help="comma-separated subset of fids to export "
                         "(default: all pose_*.npz files)")
    ap.add_argument("--include_metrics", action="store_true",
                    help="also include f1, depth_inlier, depth_median_mm "
                         "in each part for debugging")
    ap.add_argument("--euler_order", default="xyz", type=str,
                    help="scipy Euler order (default 'xyz' = X,Y,Z extrinsic)")
    ap.add_argument("--target_frame", default="opencv",
                    choices=["opencv", "blender"],
                    help="output coordinate frame. 'opencv' keeps poses in "
                         "the PV camera frame (X right, Y down, Z forward). "
                         "'blender' converts to Blender world (X right, "
                         "Y forward, Z up) with the camera at origin "
                         "(matches metaobj_wrapper transforms.json).")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    out_path = (Path(args.output) if args.output
                else pose_dir / "transforms.json")

    name_map = {}
    if args.name_map:
        for pair in args.name_map.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            k, v = pair.split(":", 1)
            name_map[k.strip()] = v.strip()

    # Discover all pose files
    pat = re.compile(r"^pose_(.+)\.npz$")
    fids = sorted(
        (pat.match(p.name).group(1) for p in pose_dir.glob("pose_*.npz")
         if pat.match(p.name)),
        key=fid_sort_key,
    )
    if args.frame_ids:
        wanted = {x.strip() for x in args.frame_ids.split(",") if x.strip()}
        fids = [f for f in fids if f in wanted]

    if not fids:
        raise RuntimeError(f"no pose_*.npz files under {pose_dir}")

    parts = []
    for fid in fids:
        p = pose_dir / f"pose_{fid}.npz"
        d = np.load(p, allow_pickle=True)
        R = np.asarray(d["R"], dtype=np.float64)
        t = np.asarray(d["t"], dtype=np.float64).reshape(3)
        s = float(d["s"])

        if fid in name_map:
            name = name_map[fid]
        else:
            name = f"{args.name_prefix}{fid}{args.name_suffix}"

        part = part_dict_from_pose(
            name, R, t, s,
            target_frame=args.target_frame,
            euler_order=args.euler_order,
            include_quat=True,
        )
        if args.include_metrics:
            part["metrics"] = {
                "f1": float(d["f1"]),
                "depth_inlier": float(d["depth_inlier"]),
                "depth_median_mm": float(d["depth_median_abs_m"]) * 1000.0,
                "det_R": float(d["det_R"]),
            }
        parts.append(part)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"parts": parts}, indent=2,
                                    ensure_ascii=False))
    print(f"wrote {len(parts)} parts to {out_path}")
    for p in parts:
        print(f"  {p['name']:>20}  t={[round(x,4) for x in p['translation']]}  "
              f"euler={[round(x,2) for x in p['rotation_euler_degrees']]}  "
              f"s={p['scale'][0]:.4f}")


if __name__ == "__main__":
    main()
