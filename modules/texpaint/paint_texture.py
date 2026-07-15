"""
Standalone texture-repaint stage (Strategy A).

TRELLIS geometry  +  single reference image  --->  Hunyuan3D-Paint (2.1) textured GLB

This module is fully decoupled from the main project:
  - it imports nothing from ../modules,
  - it only consumes files that the main pipeline already writes
    (output/<timestamp>/mesh.glb  and  output/<timestamp>/rgb_masked.png),
  - it must run inside the separate `hunyuan3dpaint` conda env (NOT `trellis`),
    because Hunyuan3D-2.1 has conflicting dependencies.

Usage (from the hunyuan3dpaint env):
    python paint_texture.py \
        --repo   /home/uvrlab/projects/Hunyuan3D-2.1 \
        --mesh   ../output/20260701_120000/mesh.glb \
        --image  ../output/20260701_120000/rgb_masked.png \
        --output ../output/20260701_120000/mesh_painted.glb

The Hunyuan3D-2.1 repo path is passed with --repo (or HUNYUAN3D_REPO env var).
"""

import os
import sys
import types
import argparse
import time


def _abs(p):
    return os.path.abspath(os.path.expanduser(p))


def _inject_bpy_stub():
    """Hunyuan3D-Paint's mesh_utils does a top-level `import bpy`, only used for
    the optional final OBJ->GLB conversion. No bpy wheel exists for this
    python/platform, so we stub the module to let the import succeed and do the
    GLB conversion ourselves with trimesh (see paint()). save_glb=False ensures
    the stub's attributes are never actually touched at runtime.
    """
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = types.ModuleType("bpy")


def build_pipeline(repo_root, max_num_view, resolution):
    """Import + construct the Hunyuan3D-Paint pipeline.

    Hunyuan3D-2.1's texture code assumes the current working directory is the
    repo root and that `hy3dpaint/` is importable, so we set both up here and
    make the checkpoint/config paths absolute to survive the chdir.
    """
    repo_root = _abs(repo_root)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Hunyuan3D-2.1 repo not found: {repo_root}")

    # textureGenPipeline.py lives under hy3dpaint/
    sys.path.insert(0, os.path.join(repo_root, "hy3dpaint"))
    # relative ckpt/cfg paths inside the pipeline are resolved against cwd
    os.chdir(repo_root)

    _inject_bpy_stub()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = os.path.join(repo_root, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(repo_root, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(repo_root, "hy3dpaint/hunyuanpaintpbr")

    pipeline = Hunyuan3DPaintPipeline(conf)
    return pipeline


def _obj_to_glb(obj_path, glb_path):
    """OBJ (+ PBR maps) -> GLB via trimesh (replaces the bpy path), embedding
    full PBR. Hunyuan3D-Paint writes albedo as <base>.jpg and separate
    <base>_metallic.jpg / <base>_roughness.jpg. glTF packs metal+rough into one
    metallicRoughnessTexture: G=roughness, B=metallic (R=occlusion, unused->255).
    """
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    mesh = trimesh.load(obj_path, process=False, force="mesh")
    base = os.path.splitext(obj_path)[0]

    albedo = Image.open(base + ".jpg").convert("RGB")
    mr_path = base + "_metallic.jpg"
    ro_path = base + "_roughness.jpg"

    if os.path.isfile(mr_path) and os.path.isfile(ro_path):
        rough = Image.open(ro_path).convert("L")
        metal = Image.open(mr_path).convert("L").resize(rough.size)
        mr = Image.merge("RGB", (Image.new("L", rough.size, 255), rough, metal))
        material = PBRMaterial(
            baseColorTexture=albedo,
            metallicRoughnessTexture=mr,
            metallicFactor=1.0,
            roughnessFactor=1.0,
        )
    else:
        material = PBRMaterial(baseColorTexture=albedo)

    mesh.visual = TextureVisuals(uv=mesh.visual.uv, material=material)
    mesh.export(glb_path)
    return glb_path


def paint(pipeline, mesh_path, image_path, output_path):
    mesh_path = _abs(mesh_path)
    image_path = _abs(image_path)
    output_path = _abs(output_path)

    for p, name in [(mesh_path, "mesh"), (image_path, "image")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # the pipeline works in OBJ space; we handle GLB ourselves
    obj_out = os.path.splitext(output_path)[0] + ".obj"

    t0 = time.time()
    result_obj = pipeline(
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=obj_out,
        save_glb=False,
    )
    print(f"[texpaint] paint done in {time.time() - t0:.1f}s -> {result_obj}")

    if output_path.lower().endswith(".glb"):
        _obj_to_glb(result_obj, output_path)
        print(f"[texpaint] exported GLB -> {output_path}")
        return output_path
    return result_obj


def main():
    ap = argparse.ArgumentParser(description="Hunyuan3D-Paint texture stage (Strategy A)")
    ap.add_argument("--repo", default=os.environ.get("HUNYUAN3D_REPO"),
                    help="Path to the Hunyuan3D-2.1 clone (or set HUNYUAN3D_REPO)")
    ap.add_argument("--mesh", required=True, help="Untextured/geometry mesh from TRELLIS (.glb/.obj)")
    ap.add_argument("--image", required=True, help="Single reference image (e.g. rgb_masked.png)")
    ap.add_argument("--output", required=True, help="Output textured mesh path (.glb)")
    ap.add_argument("--views", type=int, default=6, help="max_num_view (6-9). Lower if OOM.")
    ap.add_argument("--resolution", type=int, default=512, help="multiview resolution (512 or 768)")
    args = ap.parse_args()

    if not args.repo:
        ap.error("--repo (or HUNYUAN3D_REPO) is required")

    pipeline = build_pipeline(args.repo, args.views, args.resolution)
    paint(pipeline, args.mesh, args.image, args.output)


if __name__ == "__main__":
    main()
