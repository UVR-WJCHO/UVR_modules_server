"""Integrated joint two-part alignment pipeline.

Runs the full chain for a combined frame in one command:

  1. ALIGN   -- joint_two_part_align.run(): fit two single-part meshes
                simultaneously to the combined frame's RGB+depth+mask.
                Produces pose_part_<p>_in_C<cid>.npz per part plus the
                4-panel / depth-compare diagnostics.
  2. VERIFY  -- verify_joint_alignment.run(): render the aligned union from
                top-down / side / combined-mesh viewpoints. Produces
                verify_topdown_<cid>.png, verify_sideview_<cid>.png,
                verify_combined_<cid>.png.
  3. EXPORT  -- write transforms.json in the metaobj_wrapper format (one
                entry per part: translation / rotation_euler_degrees /
                rotation_quaternion / scale). Byte-for-byte the same format
                export_transforms_json.py produces, via the shared
                part_dict_from_pose() helper.

The three stages already exist as standalone scripts; this file only
orchestrates them (building the argument namespaces and chaining outputs)
so a single invocation does alignment -> images -> transforms.json.

Usage
-----
    python run_joint_pipeline.py \
        --data_dir      D:/metaobj/data/2606_samples \
        --combined_fid  01 \
        --part_mesh_fids 0,1 \
        --seed_dir      D:/metaobj/results_2606_v16 \
        --output_dir    D:/metaobj/results_2606_verify_v8_01 \
        --device        cuda

`--seed_dir` is a convenience: it auto-locates the combined-frame and
per-part seed poses as <seed_dir>/pose_<cid>.npz and
<seed_dir>/pose_<part>.npz. Override any of them explicitly with
--combined_pose_npz / --part_pose_npzs if they live elsewhere.

Skip flags let you re-run a single stage:
    --skip_align   reuse existing pose_part_*_in_C<cid>.npz
    --skip_verify  don't render the verification views
    --skip_export  don't (re)write transforms.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Dependency modules live alongside this file under modules/meshalignment/.
# Load them by relative file path (no sys.path manipulation). They are
# registered in sys.modules under their bare names so the modules' internal
# `from auto_align_mesh_rgbd_scale_locked import ...` resolves from the cache.
_MESH_DIR = Path(__file__).resolve().parent / "meshalignment"


def _load_local(mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, _MESH_DIR / f"{mod_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# auto_align_* first: the others import it by bare name.
_load_local("auto_align_mesh_rgbd_scale_locked")
align_run = _load_local("joint_two_part_align").run
verify_run = _load_local("verify_joint_alignment").run
part_dict_from_pose = _load_local("export_transforms_json").part_dict_from_pose


# ----------------------------- stage 1: align -------------------------------

def stage_align(args, out_dir: Path) -> None:
    ns = SimpleNamespace(
        data_dir=args.data_dir,
        combined_fid=args.combined_fid,
        part_mesh_fids=args.part_mesh_fids,
        combined_pose_npz=args.combined_pose_npz,
        part_pose_npzs=args.part_pose_npzs,
        output_dir=str(out_dir),
        device=args.device,
        # optimisation hyper-parameters (defaults mirror joint_two_part_align)
        iters=args.iters,
        lr=args.lr,
        w_sil_region=args.w_sil_region,
        w_sil_union=args.w_sil_union,
        w_overlap=args.w_overlap,
        w_scale_anchor=args.w_scale_anchor,
        w_rot_anchor=args.w_rot_anchor,
        w_contact=args.w_contact,
        w_penetrate=args.w_penetrate,
        force_upright_each_step=args.force_upright_each_step,
        stack_refine_iters=args.stack_refine_iters,
        w_stack=args.w_stack,
        w_upright=args.w_upright,
        yaw_screen_iters=args.yaw_screen_iters,
        yaw_screen_candidates=args.yaw_screen_candidates,
        soft_tau=args.soft_tau,
        soft_sample_cap=args.soft_sample_cap,
        upright_axis=args.upright_axis,
    )
    print("\n" + "=" * 70 + "\n[STAGE 1/3] JOINT ALIGNMENT\n" + "=" * 70)
    align_run(ns)


# ----------------------------- stage 2: verify ------------------------------

def stage_verify(args, out_dir: Path) -> None:
    ns = SimpleNamespace(
        data_dir=args.data_dir,
        combined_fid=args.combined_fid,
        part_fids=args.part_mesh_fids,
        joint_dir=str(out_dir),
        combined_pose=args.combined_pose_npz,
        output_dir=str(out_dir),
        device=args.device,
    )
    print("\n" + "=" * 70 + "\n[STAGE 2/3] MULTI-VIEW VERIFICATION\n" + "=" * 70)
    verify_run(ns)


# ----------------------------- stage 3: export ------------------------------

def stage_export(args, out_dir: Path, part_fids: list[str]) -> Path:
    print("\n" + "=" * 70 + "\n[STAGE 3/3] EXPORT transforms.json\n" + "=" * 70)
    cid = args.combined_fid
    parts = []
    for pf in part_fids:
        npz_path = out_dir / f"pose_part_{pf}_in_C{cid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"missing alignment pose {npz_path} -- run without "
                f"--skip_align first")
        d = np.load(npz_path, allow_pickle=True)
        R = np.asarray(d["R"], dtype=np.float64)
        t = np.asarray(d["t"], dtype=np.float64).reshape(3)
        s = float(d["s"])
        name = args.name_template.format(fid=pf, cid=cid)
        part = part_dict_from_pose(
            name, R, t, s,
            target_frame=args.target_frame,
            euler_order=args.euler_order,
            include_quat=True,
        )
        parts.append(part)

    out_path = out_dir / args.transforms_name
    out_path.write_text(json.dumps({"parts": parts}, indent=2,
                                   ensure_ascii=False))
    print(f"[export] wrote {len(parts)} parts to {out_path} "
          f"(frame={args.target_frame})")
    for p in parts:
        print(f"  {p['name']:>16}  t={[round(x, 4) for x in p['translation']]}  "
              f"euler={[round(x, 2) for x in p['rotation_euler_degrees']]}  "
              f"s={p['scale'][0]:.4f}")
    return out_path


# ----------------------------- driver ---------------------------------------

def resolve_seeds(args) -> None:
    """Fill combined_pose_npz / part_pose_npzs from --seed_dir when not
    given explicitly."""
    part_fids = [p.strip() for p in args.part_mesh_fids.split(",")]
    if args.seed_dir:
        seed = Path(args.seed_dir)
        if args.combined_pose_npz is None:
            cand = seed / f"pose_{args.combined_fid}.npz"
            if cand.exists():
                args.combined_pose_npz = str(cand)
        if args.part_pose_npzs is None:
            cands = [seed / f"pose_{pf}.npz" for pf in part_fids]
            if all(c.exists() for c in cands):
                args.part_pose_npzs = ",".join(str(c) for c in cands)
    print(f"[seeds] combined_pose = {args.combined_pose_npz}")
    print(f"[seeds] part_poses    = {args.part_pose_npzs}")


def main():
    ap = argparse.ArgumentParser(
        description="Joint two-part alignment -> verification images -> "
                    "transforms.json, in one command.")
    # --- core io ---
    ap.add_argument("--data_dir", required=True,
                    help="frame data (rgb_/rgb_masked_/depth_/intrinsic_/mesh_)")
    ap.add_argument("--combined_fid", required=True,
                    help="combined frame id, e.g. '01' or '34'")
    ap.add_argument("--part_mesh_fids", required=True,
                    help="exactly two comma-separated part ids, e.g. '0,1'")
    ap.add_argument("--output_dir", required=True,
                    help="all pipeline artifacts land here")
    ap.add_argument("--device", default="cuda")
    # --- seeds ---
    ap.add_argument("--seed_dir", default=None,
                    help="dir with single-frame pose_<id>.npz used to auto-fill "
                         "--combined_pose_npz and --part_pose_npzs")
    ap.add_argument("--combined_pose_npz", default=None,
                    help="single-rigid pose of mesh_<cid>.glb (R/s seed + "
                         "verify comparison). Overrides --seed_dir lookup.")
    ap.add_argument("--part_pose_npzs", default=None,
                    help="comma-separated 2 pose_<part>.npz (scale anchors + "
                         "yaw seeds). Overrides --seed_dir lookup.")
    # --- stage toggles ---
    ap.add_argument("--skip_align", action="store_true")
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--skip_export", action="store_true")
    # --- transforms.json export options ---
    ap.add_argument("--transforms_name", default="transforms.json")
    ap.add_argument("--name_template", default="stage_{fid}_start",
                    help="part name template; {fid}=part id, {cid}=combined id")
    ap.add_argument("--target_frame", default="blender",
                    choices=["opencv", "blender"],
                    help="'blender' matches the metaobj_wrapper transforms.json")
    ap.add_argument("--euler_order", default="xyz")
    # --- alignment hyper-parameters (mirror joint_two_part_align defaults) ---
    ap.add_argument("--iters", default=250, type=int)
    ap.add_argument("--lr", default=0.005, type=float)
    ap.add_argument("--w_sil_region", default=3.0, type=float)
    ap.add_argument("--w_sil_union", default=1.0, type=float)
    ap.add_argument("--w_overlap", default=2.0, type=float)
    ap.add_argument("--w_scale_anchor", default=50.0, type=float)
    ap.add_argument("--w_rot_anchor", default=0.5, type=float)
    ap.add_argument("--w_contact", default=20.0, type=float)
    ap.add_argument("--w_penetrate", default=200.0, type=float)
    ap.add_argument("--force_upright_each_step", default=1, type=int)
    ap.add_argument("--stack_refine_iters", default=80, type=int)
    ap.add_argument("--w_stack", default=50.0, type=float)
    ap.add_argument("--w_upright", default=20.0, type=float)
    ap.add_argument("--yaw_screen_iters", default=30, type=int)
    ap.add_argument("--yaw_screen_candidates", default="0,90,180,270")
    ap.add_argument("--soft_tau", default=4.0, type=float)
    ap.add_argument("--soft_sample_cap", default=4000, type=int)
    ap.add_argument("--upright_axis", default="y")
    args = ap.parse_args()

    part_fids = [p.strip() for p in args.part_mesh_fids.split(",")]
    if len(part_fids) != 2:
        ap.error("--part_mesh_fids must be exactly two ids, e.g. '0,1'")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolve_seeds(args)

    if not args.skip_align:
        stage_align(args, out_dir)
    else:
        print("[skip] alignment (reusing existing pose_part_*.npz)")

    if not args.skip_verify:
        stage_verify(args, out_dir)
    else:
        print("[skip] verification views")

    transforms_path = None
    if not args.skip_export:
        transforms_path = stage_export(args, out_dir, part_fids)
    else:
        print("[skip] transforms.json export")

    # --- summary ---
    print("\n" + "=" * 70 + "\n[PIPELINE DONE] artifacts in " + str(out_dir)
          + "\n" + "=" * 70)
    cid = args.combined_fid
    expected = [
        f"pose_part_{part_fids[0]}_in_C{cid}.npz",
        f"pose_part_{part_fids[1]}_in_C{cid}.npz",
        f"side_joint_{cid}.png",
        f"depth_compare_{cid}.png",
        f"verify_topdown_{cid}.png",
        f"verify_sideview_{cid}.png",
        f"verify_combined_{cid}.png",
    ]
    if transforms_path is not None:
        expected.append(transforms_path.name)
    for name in expected:
        p = out_dir / name
        flag = "ok " if p.exists() else "MISS"
        print(f"  [{flag}] {name}")


if __name__ == "__main__":
    main()