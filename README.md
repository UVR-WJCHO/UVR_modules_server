# UVR Modules Server

A HoloLens2 / webcam driven server bundling several ML pipelines:

- **Object mesh reconstruction** — hand–object segmentation (HoTrack + SAM2) → image-to-3D mesh (TRELLIS) → optional VLM-based material/affordance property estimation.
- **Hand tracking & gesture recognition** — 3D hand pose (SARTE / WiLoR) + gesture classification, streamed back to the HoloLens2 over UDP.

All importable packages live under `modules/`; each entry point adds `modules/` to `sys.path` at startup, so internal packages (`meshrecon`, `segmentor`, `hotrack`, `handtracker`, …) resolve as top-level imports.

---

## Project Structure

```
.
├── comm_hub.py                # ZeroMQ ROUTER broker — every entry point talks through this
├── main_meshrecon_comm.py     # Capture → reconstruct → align → one combined GLB
├── main_handtrack_comm.py     # Hand tracking + gesture recognition
├── main_all_hl2_receiver.py   # HL2DATA viewer (RGB / depth / overlay)
│
├── modules/
│   ├── modules_mesh.py        # MeshReconstructor        (wraps meshrecon/ TRELLIS)
│   ├── modules_segment.py     # HOSegmentor              (legacy depth-based, wraps segmentor/)
│   ├── modules_hotrack.py     # InteractiveHoTrackSegmentor (wraps hotrack/ + segmentor SAM2)
│   ├── modules_hand.py        # HandTracker_onnx (WiLoR-ONNX)
│   ├── modules_gesture.py     # GestureClassfier
│   ├── modules_obj.py         # ObjTracker               (YOLO object detection)
│   ├── modules_hl2.py         # Hl2Manager               (HoloLens2 streaming via hl2ss)
│   ├── modules_behavior.py    # BehaviorPropertyEstimator (GLB -> property JSON, no visualization)
│   │
│   ├── meshrecon/             # TRELLIS image-to-3D
│   ├── segmentor/             # SAM2 realtime + hand/object detection
│   ├── hotrack/               # online hand-object tracking
│   ├── handtracker/           # SARTE hand pose
│   ├── handtracker_wilor/     # WiLoR hand pose
│   ├── gestureclassifier/     # gesture model
│   └── behavior/              # VLM material/affordance pipeline (self-contained)
│
├── _hl2ss/                    # vendored hl2ss library (HoloLens2 sensor streaming)
├── _utils/                    # misc utilities
├── _calibration/              # HL2 depth calibration (auto-generated on connect)
├── pretrained/                # model weights — meshrecon (TRELLIS) + object (YOLO)
├── metaobj_wrapper/           # mesh/glb wrapper assets
├── output/                    # pipeline outputs (git-ignored)
│
├── requirements.txt           # legacy/bootstrap dependency snapshot
├── .env                       # API keys (OPENAI_API_KEY) — git-ignored
├── WEIGHTS.md                 # locations of all (git-ignored) model weights
└── HOTRACK_STAGE1.md          # HoTrack controls / output layout / tuning env vars
```

---

## Environment Setup

The GPU pipelines run in the **`uvr_integ`** conda environment. The currently
validated server environment is Python 3.10 with PyTorch 2.7.0+cu128.

```bash
conda activate uvr_integ
```

`requirements.txt` is an older Python-3.10/CUDA-12.1 bootstrap snapshot, not an
exact export of the current `uvr_integ` environment. Do not replace a working
`uvr_integ` environment with it blindly. A few packages still require local
installation: `SAM-2` (in-repo, `pip install -e modules/segmentor/sam2_realtime`)
and the PhysX-3D CUDA extensions (`nvdiffrast`,
`diff_gaussian_rasterization`, `diffoctreerast`).

The lightweight communication/viewer/recording tools can instead use the
separate `wiseui_commu` environment described in `_comm/README.md`; the GPU
algorithm entry points below must use `uvr_integ`.

### Model weights

All checkpoints (~16 GB) are **git-ignored** and must be placed manually.
See **[WEIGHTS.md](WEIGHTS.md)** for every path, size, and which module consumes it.

### API key (behavior pipeline only)

The behavior property estimation calls the OpenAI API. Put your key in a repo-root `.env`:

```
OPENAI_API_KEY=sk-...
```

`.env` is git-ignored and loaded automatically by `modules_behavior.py` and `modules/behavior/main.py`.

---

## Entry Points

### 1. Mesh reconstruction — HoloLens2 (`main_meshrecon_comm.py`)

Receives RGB+depth over `comm_hub`, runs interactive hand–object segmentation,
reconstructs each part on demand, then aligns the parts and returns them as one
combined GLB.

```bash
conda activate uvr_integ
python comm_hub.py --port 37001     # terminal 1
python main_meshrecon_comm.py       # terminal 2
```

Flow: HL2 frame → HoTrack segmentation → **`Space`** per unit → **`a`** per assembly
→ **`Enter`** to align and combine → `UPLOAD kw=MESH_RESULT` back to HL2.

The hl2ss-direct predecessor (`main_meshrecon.py`) is retired under `_legacy/`.

Each capture is written to its own timestamped folder:

```
output/<YYYYMMDD_HHMMSS>/
├── rgb.png
├── rgb_masked.png
├── depth.npy
├── intrinsic.npy
├── mesh.glb            # TRELLIS reconstruction
└── property.json       # only if flag_behavior = True
```

Toggles (top of the file):

| flag | default | effect |
|---|---|---|
| `flag_recon_mesh` | `True` | run TRELLIS mesh reconstruction |
| `flag_interactive_hotrack` | `True` | `True` = HoTrack (color); `False` = legacy depth-based `HOSegmentor` |
| `flag_behavior` | `False` | run behavior property estimation after each mesh |

The combined GLB goes back through `comm_hub` as a `MeshResult` — no HTTP server,
no UDP signal. See `_comm/README.md` for the transport.

### 2. Hand tracking & gesture — HoloLens2 (`main_handtrack_comm.py`)

Receives RGB+depth over `comm_hub`, estimates 3D hand pose, and returns absolute
3D joints as a `ServerResult`.

```bash
conda activate uvr_integ
python comm_hub.py --port 37001     # terminal 1
python main_handtrack_comm.py       # terminal 2
```

- Hand model: `HandTracker_onnx` (WiLoR-ONNX). Needs `onnxruntime-gpu` for the CUDA provider.
- Optional gesture recognition: `FLAG_GESTURE` (off by default).
- The UDP predecessor (`main_handtrack.py`) is retired under `_legacy/`, along with
  the SARTE (v1) and WiLoR-torch (v2) trackers.

### 3. Viewing what HL2 sends (`main_all_hl2_receiver.py`)

Subscribes to `HL2DATA` and shows RGB, aligned depth, and the overlay used to check
their registration. Read-only — it produces nothing.

```bash
python main_all_hl2_receiver.py            # --no-gui for console only
```

---

## Notes

- **Input trigger:** the mesh pipeline reads `Space` / `a` / `Enter` from stdin (terminal), so run it from a real terminal alongside the OpenCV windows.
- **GPU:** a CUDA-capable NVIDIA GPU is required for TRELLIS, SAM2, and the hand/behavior models. The current `uvr_integ` deployment uses PyTorch 2.7.0+cu128.
- **HoloLens2 streaming** uses the vendored `_hl2ss/` library; calibration data lands in `_calibration/` on first connect.
- See **[HOTRACK_STAGE1.md](HOTRACK_STAGE1.md)** for HoTrack controls, output layout, and `UVR_HOTRACK_*` tuning variables.
