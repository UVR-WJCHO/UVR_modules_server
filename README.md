# UVR Modules Server

A HoloLens2 / webcam driven server bundling several ML pipelines:

- **Object mesh reconstruction** — hand–object segmentation (HoTrack + SAM2) → image-to-3D mesh (TRELLIS) → optional VLM-based material/affordance property estimation.
- **Hand tracking & gesture recognition** — 3D hand pose (SARTE / WiLoR) + gesture classification, streamed back to the HoloLens2 over UDP.

All importable packages live under `modules/`; each entry point adds `modules/` to `sys.path` at startup, so internal packages (`meshrecon`, `segmentor`, `hotrack`, `handtracker`, …) resolve as top-level imports.

---

## Project Structure

```
.
├── main_meshrecon.py          # Mesh reconstruction — HoloLens2 input
├── main_meshrecon_webcam.py   # Mesh reconstruction — webcam input (color only)
├── main_handtrack.py          # Hand tracking + gesture recognition — HoloLens2 input
├── test_downloadMeshonHL2.py  # Utility / test script
│
├── modules/
│   ├── modules_mesh.py        # MeshReconstructor        (wraps meshrecon/ TRELLIS)
│   ├── modules_segment.py     # HOSegmentor              (legacy depth-based, wraps segmentor/)
│   ├── modules_hotrack.py     # InteractiveHoTrackSegmentor (wraps hotrack/ + segmentor SAM2)
│   ├── modules_hand.py        # HandTracker_our / HandTracker_our_wilor
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
├── requirements.txt           # snapshot of the `metaobj` conda env
├── .env                       # API keys (OPENAI_API_KEY) — git-ignored
├── WEIGHTS.md                 # locations of all (git-ignored) model weights
└── HOTRACK_STAGE1.md          # HoTrack controls / output layout / tuning env vars
```

---

## Environment Setup

Tested with the **`metaobj`** conda environment: Python 3.10, CUDA 12.1, torch 2.4.0+cu121.

```bash
conda activate metaobj
pip install -r requirements.txt
```

`requirements.txt` already pins the CUDA-12.1 torch build and the pytorch3d prebuilt wheel
(via `--extra-index-url` / `-f` lines at the top). A few packages must be installed manually
(commented at the bottom of the file): `SAM-2` (in-repo, `pip install -e modules/segmentor/sam2_realtime`)
and the PhysX-3D local source builds (`nvdiffrast`, `diff_gaussian_rasterization`, `diffoctreerast`).

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

### 1. Mesh reconstruction — HoloLens2 (`main_meshrecon.py`)

Streams RGB+depth from a HoloLens2, runs interactive hand–object segmentation, and
reconstructs a 3D mesh of the held object on demand.

```bash
python main_meshrecon.py
```

Flow: HL2 frame → HoTrack segmentation → **press `Space`** → reconstruct mesh → save outputs → (optional) property JSON → UDP signal to HL2.

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

Set the HoloLens2 IP in `host` (default `192.168.50.137`). A GLB HTTP server runs on port `8000`.

### 2. Mesh reconstruction — Webcam (`main_meshrecon_webcam.py`)

Same pipeline driven by a local webcam (color only — no depth/intrinsic, no HL2 UDP).
Uses the color-only HoTrack path.

```bash
python main_meshrecon_webcam.py
```

Set `WEBCAM_INDEX` (default `0`) and `cam_width`/`cam_height` at the top of the file.
Outputs `rgb.png`, `rgb_masked.png`, `mesh.glb` (+ `property.json`) under `output/<timestamp>/`.

### 3. Hand tracking & gesture — HoloLens2 (`main_handtrack.py`)

Streams RGB from a HoloLens2, estimates 3D hand pose, classifies gestures, and sends the
pose + gesture back over UDP.

```bash
python main_handtrack.py
```

- Hand models: `HandTracker_our` (SARTE, v1) / `HandTracker_our_wilor` (WiLoR, v2) — toggle with `Space`.
- Optional object-aware gesture gating: `FLAG_INTERACTION_DETECT`.
- Set HL2 IP in `HOST` (default `192.168.50.31`), UDP port `5005`.

---

## Notes

- **Input trigger:** the mesh pipelines read `Space` from stdin (terminal), so run them from a real terminal alongside the OpenCV windows.
- **GPU:** an NVIDIA GPU (CUDA 12.1) is required for TRELLIS, SAM2, and the hand/behavior models.
- **HoloLens2 streaming** uses the vendored `_hl2ss/` library; calibration data lands in `_calibration/` on first connect.
- See **[HOTRACK_STAGE1.md](HOTRACK_STAGE1.md)** for HoTrack controls, output layout, and `UVR_HOTRACK_*` tuning variables.
