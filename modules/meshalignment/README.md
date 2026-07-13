# metaobj_alignment

Joint two-part mesh alignment to a combined RGB-D frame, with multi-view
verification and export to a `transforms.json` pose file.

Given one captured frame (RGB + depth + mask + intrinsics) of a two-part
object and the two part meshes, the pipeline fits **both meshes
simultaneously** so their union covers the object, renders verification
views, and writes the per-part poses in the `transforms.json` format.

## Pipeline

`run_joint_pipeline.py` runs three stages in one command:

| Stage | Module | Output |
|-------|--------|--------|
| 1. Align  | `joint_two_part_align.py` | `pose_part_<p>_in_C<cid>.npz` ×2, `side_joint_<cid>.png`, `depth_compare_<cid>.png`, `soft_assign_<cid>.png` |
| 2. Verify | `verify_joint_alignment.py` | `verify_topdown_<cid>.png`, `verify_sideview_<cid>.png`, `verify_combined_<cid>.png` |
| 3. Export | `export_transforms_json.py` (`part_dict_from_pose`) | `transforms.json` |

`auto_align_mesh_rgbd_scale_locked.py` is the shared core (renderer, mask /
intrinsics / table-normal helpers, similarity-transform utilities) that the
alignment and verification stages build on.

```
run_joint_pipeline.py
├── joint_two_part_align.py ──┐
├── verify_joint_alignment.py ┼──► auto_align_mesh_rgbd_scale_locked.py
└── export_transforms_json.py    (no local deps)
```

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate                     # Linux/mac

# Install a CUDA-matched torch first, then nvdiffrast (see requirements.txt):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -r requirements.txt
```

A CUDA GPU is required (nvdiffrast rasterization). Reference environment:
torch 2.11.0+cu128, nvdiffrast 0.4.0, NVIDIA RTX A6000.

## Usage

```bash
python run_joint_pipeline.py \
  --data_dir       path/to/frame_data \
  --combined_fid   01 \
  --part_mesh_fids 0,1 \
  --seed_dir       path/to/single_frame_poses \
  --output_dir     path/to/results_01 \
  --device         cuda
```

`transforms.json` is written to `--output_dir` (alongside the verification
images). Re-run a single stage with `--skip_align` / `--skip_verify` /
`--skip_export`.

### Expected input files (`--data_dir`)

Per frame id `<fid>` (both the combined id and each part id):

- `rgb_<fid>.png` — colour image
- `rgb_masked_<fid>.png` — foreground mask source
- `depth_<fid>.npy` — metric depth (float32, metres)
- `intrinsic_<fid>.npy` — camera intrinsics
- `mesh_<fid>.glb` — mesh (part meshes, and `mesh_<cid>.glb` for the
  combined-mesh comparison in stage 2)

`--seed_dir` supplies single-frame seed poses `pose_<id>.npz` (combined-frame
pose = R/s seed + verify comparison; per-part poses = scale anchors + yaw
seeds). Override explicitly with `--combined_pose_npz` / `--part_pose_npzs`.

## transforms.json format

One entry per part, in Blender world frame (camera at origin) by default:

```json
{
  "parts": [
    {
      "name": "stage_0_start",
      "translation": [x, y, z],
      "rotation_euler_degrees": [rx, ry, rz],
      "rotation_quaternion": [w, x, y, z],
      "scale": [s, s, s]
    }
  ]
}
```

Use `--target_frame opencv` to keep poses in the camera frame,
`--name_template` to change part names (default `stage_{fid}_start`).

## Note on paths

The stage modules add the project directory to `sys.path` so the sibling
imports resolve. Run commands from the repository root (or keep all five
`.py` files in the same directory).
