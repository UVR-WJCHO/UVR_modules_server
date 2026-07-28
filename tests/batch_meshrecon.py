"""Reconstruct mesh.glb for every output/20260715_* capture (mesh recon only).

Replays exactly what main_meshrecon.py feeds MeshReconstructor for each capture,
so the produced mesh.glb matches the live pipeline. flag_texpaint is not
involved here — this is only the TRELLIS gaussian-baked mesh.

Input fidelity: main runs `meshrecon.run(Image.fromarray(masked_color))` and
saves that same array with `cv2.imwrite(..., masked_color)`. cv2 write/read
round-trips the raw array, so loading with cv2.imread (NOT PIL, which would
reinterpret channel order) and wrapping in Image.fromarray reproduces main's
exact input.

Run in an env that has the TRELLIS deps (metaobj or trellis both work):
    conda run -n metaobj python batch_meshrecon.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
CAPTURE_PATTERN = "20260715_*"

# Make packages under modules/ importable as top-level packages.
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from modules_mesh import MeshReconstructor  # noqa: E402


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main() -> int:
    # Process folders that already have mesh.glb first (known to bake fine); a
    # SIGABRT from a bad capture can't be caught, so put untested ones last to
    # avoid starving the rest of their previews.
    capture_dirs = sorted(
        (path for path in OUTPUT_ROOT.glob(CAPTURE_PATTERN) if path.is_dir()),
        key=lambda p: (not (p / "mesh.glb").is_file(), p.name),
    )
    if not capture_dirs:
        print(f"[Batch] No capture folders found: {OUTPUT_ROOT / CAPTURE_PATTERN}")
        return 1

    print(f"[Batch] Found {len(capture_dirs)} capture folders")
    print("[Batch] Initializing MeshReconstructor...")
    meshrecon = MeshReconstructor()

    succeeded = 0
    failed = 0
    skipped = 0

    for capture_dir in capture_dirs:
        image_path = capture_dir / "rgb_masked.png"
        output_path = capture_dir / "mesh.glb"
        preview_path = capture_dir / "mesh_preview.png"
        mesh_glb = None

        if output_path.is_file() and preview_path.is_file():
            print(f"[Batch] Skip {capture_dir.name} (mesh.glb + preview exist)")
            skipped += 1
            continue

        if not image_path.is_file():
            print(f"[Batch][Error] Missing input: {image_path}")
            failed += 1
            continue

        print(f"[Batch] Reconstructing {capture_dir.name}")
        try:
            # IMREAD_UNCHANGED keeps the alpha channel. cv2 write/read round-trips
            # the raw array, so this reproduces main's meshrecon input exactly:
            #   - RGBA (mask in alpha) -> TRELLIS uses the mask, skips rembg (no slab)
            #   - RGB (older captures without alpha) -> falls back to rembg path
            arr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"failed to read {image_path}")
            if arr.ndim == 3 and arr.shape[2] == 4:
                masked_image = Image.fromarray(arr, "RGBA")
            else:
                masked_image = Image.fromarray(arr)

            mesh_glb, mesh_preview = meshrecon.run(masked_image, return_preview=True)
            mesh_glb.export(str(output_path))
            # preview is RGB; cv2 writes BGR
            cv2.imwrite(str(preview_path), cv2.cvtColor(mesh_preview, cv2.COLOR_RGB2BGR))
            succeeded += 1
            print(f"[Batch] Saved: {output_path.name} + {preview_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[Batch][Error] {capture_dir.name}: {exc}")
        finally:
            if mesh_glb is not None:
                del mesh_glb
            clear_gpu_memory()

    del meshrecon
    clear_gpu_memory()

    print(
        f"[Batch] Done: total={len(capture_dirs)}, succeeded={succeeded}, "
        f"skipped={skipped}, failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Ensure the env's ninja binary (next to this python) is on PATH, so torch's
    # JIT C++ extension build works even when run via full-path python (unactivated).
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    raise SystemExit(main())
