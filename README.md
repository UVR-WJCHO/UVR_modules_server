# UVR Modules Server

A HoloLens2 / webcam driven server bundling several ML pipelines:

- **Object mesh reconstruction** — hand–object segmentation (HoTrack + SAM2) → image-to-3D mesh (TRELLIS) → optional VLM-based material/affordance property estimation.
- **Hand tracking & gesture recognition** — 3D hand pose (SARTE / WiLoR) + gesture classification, streamed back to the HoloLens2 over UDP.
- **User state (HQS)** — FOCUS / MOVING / TALKING modes and HQS panel values from HL2 sensors and the Galaxy Watch, without learned models.

All importable packages live under `modules/`; each entry point adds `modules/` to `sys.path` at startup, so internal packages (`meshrecon`, `segmentor`, `hotrack`, `handtracker`, …) resolve as top-level imports.

---

## Project Structure

```
.
├── comm_hub.py                # ZeroMQ ROUTER broker — every entry point talks through this
├── main_metaobjrecon.py       # Capture → reconstruct → align → metaobj (.glb + .json + teaser)
├── main_handtrack.py          # Hand tracking + gesture recognition
├── main_handtrack_forecast.py # Delay-compensated hand tracking (multi-horizon forecast)
├── main_all_hl2_receiver.py   # HL2DATA viewer (RGB / depth / overlay); --hqs adds user state (HQS)
├── main_devicebridge.py       # Watch (WebSocket) and phone (TCP) → comm_hub bridge
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
│   ├── modules_HQS.py         # HQS (user state) + HQSPanel (panel values)
│   ├── modules_HQS_profile.py # HQSProfile (per-user statistics of panel values and modes)
│   │
│   ├── meshrecon/             # TRELLIS image-to-3D
│   ├── meshalignment/         # mesh alignment
│   ├── segmentor/             # SAM2 realtime + hand/object detection
│   ├── hotrack/               # online hand-object tracking
│   ├── handtracker_onnx/      # WiLoR hand pose (ONNX)
│   ├── delay_nowcasting/      # delay-conditioned pose forecasting (train / eval / deploy)
│   ├── gestureclassifier/     # gesture model
│   ├── texpaint/              # texture painting
│   └── behavior/              # VLM material/affordance pipeline (self-contained)
│
├── _comm/                     # protobuf definitions + HubClient shared by every entry point
├── _hl2ss/                    # vendored hl2ss library (HoloLens2 sensor streaming)
├── _utils/                    # misc utilities
├── _calibration/              # HL2 depth calibration (auto-generated on connect)
├── pretrained/                # model weights — meshrecon (TRELLIS), object (YOLO), forecast (ONNX)
├── research_data/             # forecasting datasets and training runs (git-ignored)
├── metaobj_wrapper/           # mesh/glb wrapper assets
├── output/                    # pipeline outputs (git-ignored)
│
├── requirements-wiseui.txt    # comm_hub + every entry point except main_metaobjrecon
├── requirements-metaobj.txt   # comm_hub + main_metaobjrecon
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

To build a fresh environment, the entry points split into two projects, each with its own
dependency list traced from its imports. `comm_hub.py` is part of both.

| file | entry points |
|---|---|
| `requirements-wiseui.txt` | `comm_hub.py`, `main_all_hl2_receiver.py`, `main_devicebridge.py`, `main_adaptiveMM.py`, `main_handtrack.py`, `main_handtrack_forecast.py` |
| `requirements-metaobj.txt` | `comm_hub.py`, `main_metaobjrecon.py` |

Install the CUDA build of `torch` first, then the file; the header of each file gives the
exact commands and the packages that must be built from source.

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

### 1. Metaobj reconstruction — HoloLens2 (`main_metaobjrecon.py`)

Receives RGB+depth over `comm_hub`, runs interactive hand–object segmentation,
reconstructs each part on demand, then aligns the parts and returns them as one
combined GLB.

```bash
conda activate uvr_integ
python comm_hub.py --port 37001     # terminal 1
python main_metaobjrecon.py            # terminal 2
```

Flow: HL2 frame → HoTrack segmentation → **`Space`** per unit → **`a`** per assembly
→ **`Enter`** to align, combine, estimate per-part properties, and export the metaobj
→ `UPLOAD kw=MESH_RESULT` back to HL2.

The hl2ss-direct predecessor is retired as `_legacy/main_meshrecon.py`.

Everything for one run lands in a timestamped session folder:

```
output/<YYYYMMDD_HHMMSS>/
├── part_0/ part_1/ …   # unit captures: rgb.png, rgb_masked.png, depth.npy, intrinsic.npy, mesh.glb
├── part_01/ part_12/ … # assembly captures (no mesh)
├── stage1/ C01/ …      # alignment work
├── inputs/             # transforms.json + staged glbs fed to the combiner
└── metaobj/            # the deliverable — see modules/modules_metaobj.py
    ├── <session>.glb   # combined object; root node named after the session
    ├── <session>.json  # per-part material / affordance / 9 rangeProperties (HL2 reads this)
    ├── <session>.png   # teaser: textured render of the combined object
    ├── vlm_result.json # raw VLM output, kept for reference
    └── vlm/            # VLM input renders
```

Per-part properties come from `modules/behavior` (GPT-4o-mini over per-part X-ray renders,
needs `OPENAI_API_KEY` in the repo-root `.env`). Three of the nine properties
(electrical conductivity, thermal expansion, fracture toughness) are not produced by that
module yet and are filled from a per-material placeholder table in `modules_metaobj.py`;
the run log says so. If the VLM step fails the JSON still goes out with the combiner's
constant defaults and the failure is printed loudly.

Toggles (top of the file):

| flag | default | effect |
|---|---|---|
| `flag_recon_mesh` | `True` | run TRELLIS mesh reconstruction |
| `flag_interactive_hotrack` | `True` | `True` = HoTrack (color); `False` = legacy depth-based `HOSegmentor` |
| `flag_behavior` | `True` | run the VLM property estimation on the combined GLB at `Enter`; `False` = constant defaults, no teaser |

The combined GLB goes back through `comm_hub` as a `MeshResult` — no HTTP server,
no UDP signal. See `_comm/README.md` for the transport.

### 2. Hand tracking & gesture — HoloLens2 (`main_handtrack.py`)

Receives RGB+depth over `comm_hub`, estimates 3D hand pose, and returns absolute
3D joints as a `ServerResult`.

```bash
conda activate uvr_integ
python comm_hub.py --port 37001     # terminal 1
python main_handtrack.py            # terminal 2
```

- Hand model: `HandTracker_onnx` (WiLoR-ONNX). Needs `onnxruntime-gpu` for the CUDA provider.
- Optional gesture recognition: `FLAG_GESTURE` (off by default).
- The UDP predecessor under `_legacy/` shares this file name but is unrelated; it is
  retired along with the SARTE (v1) and WiLoR-torch (v2) trackers.

### 3. Delay-compensated hand tracking (`main_handtrack_forecast.py`)

Same input as entry point 2, but the reply is a grid of short-horizon forecasts rather
than the pose for the captured frame. The headset measures how stale the reply is when
it renders and interpolates the two horizons that bracket that delay, so the pose it
draws describes the moment it is shown.

```bash
conda activate uvr_integ
python comm_hub.py --port 37001        # terminal 1
python main_handtrack_forecast.py      # terminal 2
```

- Forecast weights: `pretrained/forecast/mixed3.onnx` (default) and `mixed2.onnx`,
  selected with `--model`.
- Sent on its own channel (`HAND_FORECAST`), separate from entry point 2's `SERVER_RESULT`.
- Training and evaluation code for the forecaster lives in `modules/delay_nowcasting/`;
  its datasets and training runs are under `research_data/` (git-ignored).

### 4. HL2 viewer and user state (`main_all_hl2_receiver.py`)

Without options the receiver subscribes to `HL2DATA` and shows RGB with the hand, gaze and
depth overlay. In this mode it is read-only and publishes nothing.

```bash
python main_all_hl2_receiver.py            # --no-gui for console only
```

With `--hqs` it also runs the user-state module (`modules/modules_HQS.py`) and publishes two
channels to the HoloLens2:

- `HL2_MODE` — `FOCUS`, `MOVING`, `TALKING` or `NONE`, once per second.
- `HL2_UI` — HQS panel values (emotion / physical / social) as JSON, sent when a value changes,
  with a per-type `highlight` flag when the value changes sharply within a few seconds.

To judge speech it requests the HL2 microphone over `HL2_CONTROL`. The wire format for the
headset side is specified in [_comm/HQS_PROTOCOL.md](_comm/HQS_PROTOCOL.md).

```bash
conda activate uvr_integ
python comm_hub.py --port 37001            # terminal 1
python main_devicebridge.py                # terminal 2   (watch and phone; see below)
python main_all_hl2_receiver.py --hqs      # terminal 3
```

| option | effect |
|---|---|
| `--start-posture standing` | the session starts standing (default `seated`); head height is relative, so the starting posture sets the reference |
| `--record <file>` | appends raw `HL2DATA` packets to `<file>` and microphone chunks to `<file>.audio`; replay with `python modules/modules_HQS.py --replay <file>` |
| `--no-voice` | does not request the microphone; `TALKING` is then never reported |
| `--general` | selects the qualifying state with the highest score instead of the fixed priority order |
| `--profile <file>` | user profile to accumulate into (default `output/hqs/profiles/default.json`); see below |

`HL2_CONTROL` sets the complete stream state on the headset, so running `--hqs` together
with `main_adaptiveMM.py` makes the two overwrite each other's requests.

**Dependence on the device bridge.** The modes are computed from HoloLens2 data alone, but
part of the panel and the phone link come from `main_devicebridge.py`:

| output | without `main_devicebridge.py` | with `main_devicebridge.py` |
|---|---|---|
| `HL2_MODE` | complete | unchanged |
| `HL2_UI` emotion | not sent | watch heart rate |
| `HL2_UI` physical | substitute from HL2 head motion (`source: "hqs"`) | watch activity |
| `HL2_UI` social | HL2 face and HL2 microphone | additionally the watch VAD |
| phone ↔ HL2 (`PHONE_TO_HL2` / `HL2_TO_PHONE`) | unavailable | available |

Run the bridge whenever the panel is expected to reflect watch data or the phone app is in
use. Without it the receiver keeps running; after 5 s without watch messages it logs a
warning and switches to the substitutes above, and it returns to the watch values once
messages resume.

**User profile.** With `--hqs` the receiver adds the current panel values, heart rate and mode
to a per-user profile once per second while `HL2DATA` is arriving, and writes it every 60 s and
on exit. One JSON file holds one person (`--profile output/hqs/profiles/A.json`). An existing
file is always loaded and extended; a missing file is created. The profile stores statistics,
not a log:

- one entry per local calendar month, keeping the current month and the previous six;
- per month, weekday and weekend × 24 local hours, each with observed seconds, seconds with the
  watch connected, seconds per mode, and `[n, sum, sum of squares]` for emotion, physical,
  social, digital and heart rate (seconds without a value, such as no watch, are not counted);
- per month, observed seconds per place.

The size is therefore bounded (a few hundred KB at most) regardless of how long it is
recorded, and any period or hour range can be merged by adding the entries. The place is the
city-level location of the server's public IP, looked up once at start from `ipinfo.io`; this
assumes the headset is on the same network as the server. If the lookup fails the place is
`unknown` and the server's time zone is used.

### 5. Device bridge — Galaxy Watch and phone (`main_devicebridge.py`)

Devices that cannot speak ZeroMQ get an adapter here instead of a change to their app.
The Wear OS `WatchSensor` app streams heart rate, activity transitions and 16 kHz PCM
over WebSocket; the bridge accepts it exactly as the original `server.py` did, runs
`webrtcvad` on the audio, and publishes `WATCH_HR`, `WATCH_ACTIVITY`, `WATCH_AUDIO` and
`WATCH_VAD`. The RoverHandoff phone app speaks length-prefixed JSON over TCP to what used
to be the HoloLens; the bridge takes that role and relays frames as `PHONE_TO_HL2` /
`HL2_TO_PHONE`. Point both apps' server IP at this PC; nothing else on them changes.
Details in [modules/devicebridge/README.md](modules/devicebridge/README.md). The watch
channels feed the HQS panel of entry point 4.

```bash
python comm_hub.py --port 37001        # terminal 1
python main_devicebridge.py            # terminal 2   (--no-raw-audio to publish VAD only)
```

---

## Notes

- **Input trigger:** the mesh pipeline reads `Space` / `a` / `Enter` from stdin (terminal), so run it from a real terminal alongside the OpenCV windows.
- **GPU:** a CUDA-capable NVIDIA GPU is required for TRELLIS, SAM2, and the hand/behavior models. The current `uvr_integ` deployment uses PyTorch 2.7.0+cu128.
- **HoloLens2 streaming** uses the vendored `_hl2ss/` library; calibration data lands in `_calibration/` on first connect.
- See **[HOTRACK_STAGE1.md](HOTRACK_STAGE1.md)** for HoTrack controls, output layout, and `UVR_HOTRACK_*` tuning variables.
