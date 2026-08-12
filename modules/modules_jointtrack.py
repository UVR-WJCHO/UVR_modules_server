#!/usr/bin/env python
"""Fit part meshes to their captures, assemble them, and write what
metaobj_wrapper/combine_rocket_glb.py needs.

    python modules_jointtrack.py \
        --data_dir  /path/to/frames \
        --parts     2,3,01 \
        --assembly  23:2,3 \
        --assembly  012:01,2 \
        --order     01,2,3 \
        --output_dir /path/to/out

Two stages, then the handover.

**Fit** each part to the capture it was reconstructed from. Every orientation
hypothesis is rendered and scored with texture included, because a part that is
nearly a surface of revolution gives the outline nothing to say about its turn.
Girth is corrected here too, where the part is alone and fully in view — the
only place a reconstruction that came out thin can be measured without
guessing. Out come a pose, a metric scale, and a reshaped mesh.

**Assemble** those parts into each capture of them joined. The parts are
composited through a shared z-buffer so each pixel is judged against whatever
is actually in front, and they go in one at a time, most visible first. The
joint is a circle-to-circle registration: parts meet on circular faces, so
seating, coaxiality and the step at the seam are one constraint rather than
three that pull against each other.

**Hand over** the corrected meshes as mesh_0.glb, mesh_1.glb, … and a
transforms.json placing them, in Blender's axes. Assemblies solved against
different captures are chained through a part they share.

A *unit* is one part, or several already solved together, named by joining
their ids with `+`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshalignment import geom
from meshalignment.assemble import (NotASimilarity, chain_solves, stage_meshes,
                                    write_transforms)
from meshalignment.calibrate import calibrate_frame
from meshalignment.frames import _paths, load_frame, mask_touches_border
from meshalignment.joint import align_assembly, choose_rim_pair, load_unit
from meshalignment.render import MeshRenderer
from meshalignment.solo import align_frame
from meshalignment.viz import save_assembly, save_compare, save_orbit_views


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------

def fit_parts(data_dir: Path, fids: List[str], out_dir: Path, mesh_dir: Path,
              device: str, girth_rounds: int, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fid in fids:
        print(f"\n--- part {fid} ---")
        frame = load_frame(data_dir, fid)
        cropped = [k for k, v in mask_touches_border(frame.mask).items() if v]
        if cropped:
            print(f"  NOTE: reaches the {', '.join(cropped)} border — cropped "
                  f"there, so the outline says nothing about size on that side")
        res = align_frame(frame, device=device, verbose=verbose)
        print(f"  fit  {res.hypothesis}  {res.metrics.line()}")

        mesh, T, metrics, girth = frame.mesh, res.T, res.metrics, 1.0
        if girth_rounds > 0:
            mesh, cal = calibrate_frame(frame, res.T, device=device,
                                        rounds=girth_rounds)
            T, metrics, girth = cal.T, cal.metrics_after, cal.factor
            print(f"  girth x{girth:.3f}  (width {cal.width_before:.3f} -> "
                  f"{cal.width_after:.3f})")
        d = mesh_dir / f"part_{fid}"
        d.mkdir(parents=True, exist_ok=True)
        mesh.export(d / "mesh.glb")
        src = _paths(data_dir, fid)
        for key, name in [("rgb", "rgb.png"), ("masked", "rgb_masked.png"),
                          ("depth", "depth.npy"), ("K", "intrinsic.npy")]:
            link = d / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src[key].resolve())

        R, t, s = geom.decompose(T)
        np.savez(out_dir / f"pose_{fid}.npz", T=T, R=R, t=t, s=np.array(s),
                 K=frame.K, fid=fid, girth_factor=np.array(girth),
                 **{k: np.array(v) for k, v in metrics.as_dict().items()})
        H, W = frame.shape
        save_compare(out_dir / f"compare_{fid}.png", frame,
                     MeshRenderer(mesh, frame.K, H, W, device=device), T, metrics,
                     title=f"{res.hypothesis}  girth x{girth:.3f}")
        summary[fid] = {"scale": s, "girth_factor": girth,
                        "cropped_borders": cropped, **metrics.as_dict()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------
# stage 2
# --------------------------------------------------------------------------

def fit_assembly(data_dir: Path, mesh_dir: Path, seed_dir: Path, cid: str,
                 unit_ids: List[str], out_dir: Path, device: str,
                 init_dir=None, init_cid=None,
                 lock_scale: bool = True) -> Dict[str, np.ndarray]:
    """The capture of the assembly comes from the original data; the meshes
    placed into it come from stage 1, reshaped. Only the parts were fitted
    there, so the assembly's own frame was never staged alongside them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- assembly {cid} <- units {unit_ids} ---")
    frame = load_frame(data_dir, cid)
    cropped = [k for k, v in mask_touches_border(frame.mask).items() if v]
    if cropped:
        print(f"  NOTE: reaches the {', '.join(cropped)} border")
    H, W = frame.shape
    units = [load_unit(mesh_dir, u.split("+"), seed_dir, device, frame.K, H, W,
                       init_dir=init_dir, init_cid=init_cid) for u in unit_ids]
    res = align_assembly(frame, units, device=device, lock_scale=lock_scale)

    print(f"\n  UNION {res.metrics.line()}")
    print(f"  JOINT  overlap {res.penetration_mm:.1f}mm   "
          f"gap {res.joint_gap_mm:.1f}mm")
    print(f"  RIMS   centres {res.rim_gap_mm:.1f}mm apart   "
          f"radii differ {res.rim_radius_diff_mm:+.1f}mm   "
          f"planes {res.rim_plane_deg:.2f}deg apart")
    for r in res.units:
        print("  " + r.line())
        if r.occluded_frac > 0.85:
            print(f"       WARNING: unit {r.name} is almost entirely hidden "
                  f"here; its pose rests on priors, not on this capture")

    Ts = [res.unit_poses[u.name] for u in units]
    for u, T in zip(units, Ts):
        for m in u.members:
            Tm = res.poses[m.fid]
            R, t, s = geom.decompose(Tm)
            np.savez(out_dir / f"pose_{m.fid}_in_C{cid}.npz", T=Tm, R=R, t=t,
                     s=np.array(s), K=frame.K, part_fid=m.fid, combined_fid=cid,
                     unit=u.name)
    save_assembly(out_dir / f"side_{cid}.png", frame, units, Ts, res.metrics)
    pair = (choose_rim_pair(units, Ts)[:2]
            if len(units) == 2 and all(u.rims for u in units) else None)
    save_orbit_views(out_dir / f"views_{cid}.png", frame, units, Ts, pair)
    (out_dir / "result.json").write_text(json.dumps({
        "combined_fid": cid, "units": [u.name for u in units],
        "order": res.order, "union": res.metrics.as_dict(),
        "penetration_mm": res.penetration_mm, "joint_gap_mm": res.joint_gap_mm,
        "rim_gap_mm": res.rim_gap_mm, "rim_plane_deg": res.rim_plane_deg,
        "rim_radius_diff_mm": res.rim_radius_diff_mm,
        "per_unit": [{"name": r.name, "occluded_frac": r.occluded_frac,
                      "inside_mask_frac": r.inside_mask_frac,
                      "tilt_deg": r.tilt_deg, "twist_deg": r.twist_deg,
                      "yaw_held": r.yaw_held} for r in res.units],
    }, indent=2, default=str))
    return dict(res.unit_poses)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="captures: <dir>/part_<fid>/{rgb,rgb_masked,depth,"
                         "intrinsic,mesh}.* or <dir>/{rgb,...}_<fid>.*")
    ap.add_argument("--parts", required=True,
                    help="frame ids to fit in stage 1, e.g. '2,3,01'")
    ap.add_argument("--assembly", action="append", required=True,
                    metavar="CID:UNITS",
                    help="an assembly capture and the units in it, e.g. "
                         "'23:2,3' or '012:01,2'. Repeat for each assembly. "
                         "Join a unit's parts with '+'.")
    ap.add_argument("--order", default=None,
                    help="unit order for mesh_0.glb, mesh_1.glb, … "
                         "(default: first appearance)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--girth_rounds", type=int, default=4,
                    help="passes correcting each mesh's width; 0 leaves the "
                         "reconstruction as it is")
    ap.add_argument("--init_dir", default=None,
                    help="earlier solve holding a multi-part unit's internals")
    ap.add_argument("--init_cid", default=None)
    ap.add_argument("--target_frame", default="blender",
                    choices=["blender", "opencv"])
    ap.add_argument("--euler_order", default="xyz")
    ap.add_argument("--name_template", default="{unit}",
                    help="part name in transforms.json; {unit} = unit id")
    ap.add_argument("--skip_fit", action="store_true",
                    help="reuse stage 1 output already in --output_dir")
    args = ap.parse_args()

    out = Path(args.output_dir)
    seed_dir, mesh_dir = out / "stage1", out / "stage1" / "meshes"
    fids = [f.strip() for f in args.parts.split(",") if f.strip()]

    assemblies = []
    for spec in args.assembly:
        if ":" not in spec:
            ap.error(f"--assembly {spec!r} must look like 'CID:UNIT,UNIT'")
        cid, units = spec.split(":", 1)
        assemblies.append((cid.strip(),
                           [u.strip() for u in units.split(",") if u.strip()]))

    print("=" * 70 + "\n[1/3] FIT EACH PART TO ITS OWN CAPTURE\n" + "=" * 70)
    if args.skip_fit:
        print("  reusing", seed_dir)
    else:
        fit_parts(Path(args.data_dir), fids, seed_dir, mesh_dir, args.device,
                  args.girth_rounds)

    print("\n" + "=" * 70 + "\n[2/3] ASSEMBLE\n" + "=" * 70)
    solves = [fit_assembly(Path(args.data_dir), mesh_dir, seed_dir, cid, units,
                           out / f"C{cid}", args.device, args.init_dir,
                           args.init_cid)
              for cid, units in assemblies]

    print("\n" + "=" * 70 + "\n[3/3] HAND OVER TO metaobj_wrapper\n" + "=" * 70)
    merged = chain_solves(solves)
    order = ([u.strip() for u in args.order.split(",") if u.strip()]
             if args.order else list(merged))
    missing = [u for u in order if u not in merged]
    if missing:
        ap.error(f"--order names units that were never solved: {missing}")

    inputs = out / "inputs"
    stage_meshes(mesh_dir, order, inputs / "glbs")
    try:
        parts = write_transforms(
            inputs / "transforms.json",
            [(args.name_template.format(unit=u), merged[u]) for u in order],
            target_frame=args.target_frame, euler_order=args.euler_order)
    except NotASimilarity as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    for i, (u, p) in enumerate(zip(order, parts)):
        print(f"  mesh_{i}.glb  <- unit {u:>4}   "
              f"t={[round(v, 4) for v in p['translation']]}  "
              f"s={p['scale'][0]:.5f}")
    print(f"\n  {inputs}/glbs/mesh_0..{len(order) - 1}.glb")
    print(f"  {inputs}/transforms.json  ({len(parts)} parts, "
          f"{args.target_frame} axes)")
    print(f"\n  combine_rocket_glb.py expects exactly 5 parts; this run wrote "
          f"{len(parts)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
