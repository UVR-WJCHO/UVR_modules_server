# texpaint — Strategy A texture stage (decoupled)

Improves texture quality by re-texturing TRELLIS geometry with **Hunyuan3D-Paint (2.1)**
using a single reference image. Completely separate from the main pipeline — no existing
file under `modules/` or `main_meshrecon.py` is modified.

## Data flow

```
main pipeline (trellis env)                texpaint (hunyuan3dpaint env)
--------------------------------           ------------------------------
capture image ─► TRELLIS ─► mesh.glb ─┐
              └► rgb_masked.png ───────┼─►  paint_texture.py ─► mesh_painted.glb
                                       │      (Hunyuan3D-Paint)
```

`paint_texture.py` only reads `mesh.glb` + `rgb_masked.png` that the main pipeline already
saves under `output/<timestamp>/`. Hunyuan3D-Paint re-textures from **geometry + reference
image**, so the existing baked texture on `mesh.glb` is irrelevant — only its shape is used.

## Feasibility (RTX 3090/4090, 24GB) — VERIFIED

- **Measured: 46.3s** for the paint step on an RTX 3090 (default `max_num_view=6,
  resolution=512`), producing a **2048×2048** texture. Fits in 24GB.
- Total budget: TRELLIS geometry ~15–25s + paint ~46s ≈ ~65s on a 3090; a 4090/A100 brings
  the whole pipeline comfortably under a minute.
- If OOM: lower `--views` to 4, keep `--resolution 512`.

## Note on PBR maps in the GLB

Hunyuan3D-Paint outputs full PBR (`.jpg` albedo + `_metallic.jpg` + `_roughness.jpg` next to
the `.obj`). The GLB written here uses trimesh, which embeds **albedo only**. If HoloLens needs
metallic/roughness inside the GLB, add a proper PBR GLB exporter (pygltflib) in `_obj_to_glb`.

## Why a separate conda env

Hunyuan3D-2.1 pins different `diffusers`/`torch` versions and ships two custom CUDA
extensions (`custom_rasterizer`, `DifferentiableRenderer`). Installing it into the `trellis`
env risks breaking geometry generation. Keep them isolated; the two stages hand off via files.

## Setup

```bash
bash setup_env.sh          # clones Hunyuan3D-2.1, builds env `hunyuan3dpaint`, compiles CUDA ops
```
The `tencent/Hunyuan3D-2.1` repo is public (non-gated) — no license click needed. Weights
(~20GB) auto-download to `~/.cache/huggingface` on first run.

## Run

```bash
conda activate hunyuan3dpaint
export CUDA_HOME=/usr/local/cuda-12.1
export HUNYUAN3D_REPO=$HOME/projects/extra/Hunyuan3D-2.1
python paint_texture.py \
    --mesh   ../output/<ts>/mesh_01.glb \
    --image  ../output/<ts>/rgb_masked_01.png \
    --output ../output/<ts>/mesh_painted_01.glb
```

Output `mesh_painted.glb` carries full PBR: `baseColorTexture` + `metallicRoughnessTexture`
(glTF packs roughness in G, metallic in B). View in a PBR-capable viewer (Blender,
gltf-viewer.donmccurdy.com) — simple viewers show albedo only.

## Main pipeline integration (flag-gated)

`main_meshrecon.py` keeps its original behaviour by default. The new pipeline is opt-in via
env var — no code edits needed to switch:

```bash
# old behaviour (TRELLIS gaussian-baked texture):
python main_meshrecon.py

# new behaviour (TRELLIS geometry + Hunyuan3D-Paint PBR):
UVR_USE_TEXPAINT=1 python main_meshrecon.py      # also needs flag_recon_mesh=True
```

When `UVR_USE_TEXPAINT=1`, after TRELLIS exports `mesh.glb` the main process offloads TRELLIS
to CPU (frees VRAM), runs `paint_texture.py` in the `hunyuan3dpaint` env via subprocess to make
`mesh_painted.glb`, then reloads TRELLIS. Knobs: `UVR_TEXPAINT_VIEWS` (drop to 4 if the paint
step OOMs — seg/hotrack models also hold VRAM), `UVR_TEXPAINT_RESOLUTION` (512/768). Bridge code
is `modules/modules_texpaint.py`.
