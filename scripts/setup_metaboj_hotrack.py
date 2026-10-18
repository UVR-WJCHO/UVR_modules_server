"""Install the complete server pipeline into one Conda environment.

Run on the Linux/NVIDIA execution host. This installs packages and builds CUDA
extensions; it never downloads checkpoints, constructs models or starts tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "metaboj_hotrack"
CONSTRAINTS = ROOT / "constraints-metaboj_hotrack.txt"
NATIVE = ROOT / "requirements-metaboj_hotrack-native.txt"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("Install the full metaboj_hotrack runtime on Linux x86_64/NVIDIA.")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise SystemExit("Anaconda/Miniconda must be installed first.")
    if not shutil.which("git"):
        raise SystemExit("git is required to install the pinned extension sources.")
    cuda_home_raw = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_home_raw) / "bin/nvcc") if cuda_home_raw else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise SystemExit("Install the CUDA 12.8 toolkit and set CUDA_HOME (nvcc is required).")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    if not re.search(r"release\s+12\.8(?:\D|$)", version):
        raise SystemExit("CUDA toolkit 12.8 is required by the pinned PyTorch/CUDA environment.")

    env = os.environ.copy()
    env["CUDA_HOME"] = str(Path(cuda_home_raw).resolve() if cuda_home_raw else Path(nvcc).resolve().parents[1])
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env.get("PATH", "")
    env["FORCE_CUDA"] = "1"
    env["SAM2_BUILD_CUDA"] = "1"
    env["SAM2_BUILD_ALLOW_ERRORS"] = "0"
    env.setdefault("MAX_JOBS", "4")

    environments = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))
    exists = any(Path(prefix).name == ENV_NAME for prefix in environments["envs"])
    action = "update" if exists else "create"
    command = [conda, "env", action, "--name", ENV_NAME, "--file", str(ROOT / "environment-metaboj_hotrack.yml")]
    if not exists:
        command.append("--yes")
    run(command, env)

    # Every phase targets exactly the same environment. Source builds must see
    # its installed torch rather than a separate pip build-isolation environment.
    pip = [conda, "run", "--no-capture-output", "-n", ENV_NAME, "python", "-m", "pip"]
    # All native runtime dependencies are declared in the base manifest. Force
    # fresh builds so a rerun cannot retain extensions built for an older torch.
    run(pip + ["install", "--no-build-isolation", "--no-deps", "--no-cache-dir", "--force-reinstall",
               "--constraint", str(CONSTRAINTS), "--requirement", str(NATIVE)], env)
    run(pip + ["install", "--no-build-isolation", "--no-deps", "--editable", str(ROOT / "modules/segmentor/sam2_realtime")], env)
    # rembg depends on the headless distribution; restore the pinned GUI files
    # after dependency installation because main_metaobjrecon uses cv2.imshow.
    run(pip + ["install", "--force-reinstall", "--no-deps", "opencv-python==4.12.0.88"], env)
    run(pip + ["check"], env)
    print(f"Installed packages in {ENV_NAME}. Checkpoints and runtime tests were not run.")
    print(f"conda activate {ENV_NAME}")
    print("python comm_hub.py --port 37001")
    print("python main_metaobjrecon.py  # another terminal in the same environment")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Installation stopped (exit {exc.returncode}); no model or inference test was started.", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
