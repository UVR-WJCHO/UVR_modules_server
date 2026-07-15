"""
Thin bridge from the main pipeline to the standalone Hunyuan3D-Paint texture
stage. The paint model lives in a SEPARATE conda env (`hunyuan3dpaint`) with
dependencies that conflict with the main/TRELLIS env, so we invoke it as a
subprocess (modules/texpaint/paint_texture.py) rather than importing it in-process.

Only stdlib is used here, so importing this module is cheap and never touches
the GPU. Configurable via env vars (all have working defaults):
  UVR_TEXPAINT_PYTHON      python of the hunyuan3dpaint env
  UVR_TEXPAINT_SCRIPT      path to paint_texture.py
  HUNYUAN3D_REPO           Hunyuan3D-2.1 clone
  CUDA_HOME                CUDA toolkit used to build the extensions
  UVR_TEXPAINT_VIEWS       max_num_view (lower to 4 if the paint step OOMs)
  UVR_TEXPAINT_RESOLUTION  multiview resolution (512 or 768)
"""

import os
import subprocess

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # modules/


class TexPaintRunner:
    def __init__(self):
        self.python = os.environ.get(
            "UVR_TEXPAINT_PYTHON",
            "/home/uvrlab/anaconda3/envs/hunyuan3dpaint/bin/python",
        )
        self.script = os.environ.get(
            "UVR_TEXPAINT_SCRIPT",
            os.path.join(_MODULE_DIR, "texpaint", "paint_texture.py"),
        )
        self.repo = os.environ.get(
            "HUNYUAN3D_REPO", "/home/uvrlab/projects/extra/Hunyuan3D-2.1"
        )
        self.cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.1")
        self.views = os.environ.get("UVR_TEXPAINT_VIEWS", "6")
        self.resolution = os.environ.get("UVR_TEXPAINT_RESOLUTION", "512")

        for path, name in [(self.python, "python"), (self.script, "script")]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"TexPaintRunner {name} not found: {path}. "
                    "Run texpaint/setup_env.sh or set the UVR_TEXPAINT_* env vars."
                )

    def run(self, mesh_path, image_path, output_path):
        """mesh (.glb geometry) + reference image -> textured PBR .glb.
        Blocks until the subprocess finishes; raises on non-zero exit.
        """
        env = os.environ.copy()
        env["CUDA_HOME"] = self.cuda_home
        env["PATH"] = os.path.join(self.cuda_home, "bin") + os.pathsep + env.get("PATH", "")

        cmd = [
            self.python, self.script,
            "--repo", self.repo,
            "--mesh", mesh_path,
            "--image", image_path,
            "--output", output_path,
            "--views", str(self.views),
            "--resolution", str(self.resolution),
        ]
        print(f"[TexPaint] launching: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=True)
        return output_path
