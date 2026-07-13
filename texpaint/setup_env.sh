#!/usr/bin/env bash
# One-time setup for the standalone Hunyuan3D-Paint texture stage.
# Creates an ISOLATED conda env (does NOT touch the existing `trellis` env).
# These are the exact steps that were verified working on this machine
# (RTX 3090/4090 24GB, CUDA 12.1 toolkit at /usr/local/cuda-12.1).
set -e

REPO_DIR="${HUNYUAN3D_REPO:-$HOME/projects/extra/Hunyuan3D-2.1}"
ENV_NAME="hunyuan3dpaint"
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.1}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/8] Clone Hunyuan3D-2.1 -> $REPO_DIR"
mkdir -p "$(dirname "$REPO_DIR")"
[ -d "$REPO_DIR/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 "$REPO_DIR"

echo "[2/8] Create conda env '$ENV_NAME' (python 3.10)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.10 -y || true
PY="$(conda info --base)/envs/$ENV_NAME/bin/python"
PIP="$PY -m pip"

echo "[3/8] PyTorch 2.5.1 + cu121"
$PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "[4/8] Requirements MINUS bpy/deepspeed/tb_nightly (bpy has no py3.10 wheel;"
echo "      deepspeed/tb_nightly are training-only. GLB export is done via trimesh instead.)"
grep -viE "^\s*(bpy|deepspeed|tb_nightly)\b" "$REPO_DIR/requirements.txt" > /tmp/hy3d_req_nobpy.txt
$PIP install -r /tmp/hy3d_req_nobpy.txt

echo "[5/8] Pin setuptools<81 (realesrgan/basicsr still need pkg_resources)"
$PIP install "setuptools<81"

echo "[6/8] Patch basicsr for torchvision>=0.17 (functional_tensor was removed)"
BASICSR_DEG="$(conda info --base)/envs/$ENV_NAME/lib/python3.10/site-packages/basicsr/data/degradations.py"
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$BASICSR_DEG"

echo "[7/8] Compile the two custom extensions (needs CUDA_HOME + --no-build-isolation)"
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
( cd "$REPO_DIR/hy3dpaint/custom_rasterizer" && $PIP install -e . --no-build-isolation )
( cd "$REPO_DIR/hy3dpaint/DifferentiableRenderer" && \
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME" && \
    bash compile_mesh_painter.sh )

echo "[8/8] RealESRGAN checkpoint"
mkdir -p "$REPO_DIR/hy3dpaint/ckpt"
[ -f "$REPO_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || wget -c -O \
    "$REPO_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

echo
echo "Done. Hunyuan3D-2.1 weights (~20GB) auto-download from Hugging Face on first run"
echo "(the repo is public/non-gated; no license click needed)."
echo
echo "Test:"
echo "  conda activate $ENV_NAME"
echo "  export CUDA_HOME=$CUDA_HOME_DIR HUNYUAN3D_REPO=$REPO_DIR"
echo "  python $SELF_DIR/paint_texture.py \\"
echo "      --mesh <ts>/mesh.glb --image <ts>/rgb_masked.png --output <ts>/mesh_painted.glb"
