# Interactive HoTrack Stage 1

This branch wires the online interactive HoTrack stage-1 tracker into the existing `main_all.py` entry point.

## What It Does

- Receives live HL2 RGB/depth frames through the existing `Hl2Manager` loop.
- Automatically detects hand-object interactions with `segmentor/100DOH_small.pt`.
- Adds only the interacting object masks to SAM2 realtime tracking by default.
- Saves every tracked/edited mask as files under `output/hotrack_stage1/<video_name>/`.
- Allows interactive mask selection, deletion, current-frame removal, and brush edits.

It intentionally does not run mask merge/split/re-id, graph construction, or event detection.

## Run

Place a SAM2.1 checkpoint in the ignored checkpoint directory, for example:

```bash
mkdir -p segmentor/sam2_realtime/checkpoints
# copy or download the checkpoint here:
# segmentor/sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt
```

Then run the existing entry point:

```bash
python main_all.py
```

The interactive stage-1 path is enabled by default. To use the previous segmentor:

```bash
UVR_USE_INTERACTIVE_HOTRACK=0 python main_all.py
```

## Controls

- Left click: select a mask id under the cursor.
- `[` / `]`: cycle selected id.
- `d`: delete selected id from the tracker.
- `D`: remove selected id from the current frame output only.
- `e`: start brush editing the selected mask.
- Left drag while editing: add mask pixels.
- Right drag or Ctrl+left drag while editing: erase mask pixels.
- `+` / `-`: change brush size.
- `a`: apply edit back to the tracker.
- `z`: cancel edit.
- `s`: save the current frame/masks again.
- `q` or Esc: quit the online loop.

## Outputs

Default output path:

```text
output/hotrack_stage1/hl2_online/
  _meta.json
  ops_log.jsonl
  frames/000000.json
  masks_png/000000/id_000100.png
  overlays/000000.jpg
```

Masks are saved as single-channel PNG files. Object ids start at `100`; hand ids are reserved below `100` and are not saved unless hand tracking/output is explicitly enabled.

## Useful Environment Variables

- `UVR_HOTRACK_OUTPUT_DIR`: base output directory, default `output/hotrack_stage1`.
- `UVR_HOTRACK_VIDEO_NAME`: run name, default `hl2_online`.
- `UVR_HOTRACK_SAM2_VARIANT`: `tiny`, `small`, `base_plus`, or `large`; default `tiny` to avoid OOM.
- `UVR_HOTRACK_SAM2_CHECKPOINT`: explicit checkpoint path.
- `UVR_HOTRACK_MAX_SIDE`: resize long side before SAM2, default `960`.
- `UVR_HOTRACK_DETECT_INTERVAL`: YOLO detection interval, default `3` frames.
- `UVR_HOTRACK_MAX_OBJECTS`: max active object tracks, default `5`.
- `UVR_HOTRACK_TRACK_HANDS`: set `1` to track hands as SAM2 ids.
- `UVR_HOTRACK_INCLUDE_HANDS`: set `1` to include hand masks in overlays/saved masks.
- `UVR_HOTRACK_HAND_BACKEND`: `auto`, `mediapipe`, `yolo`, or `none`.
- `UVR_HOTRACK_OFFLOAD_VIDEO`: default `1`, keeps frames off GPU memory when possible.
- `UVR_HOTRACK_OFFLOAD_STATE`: default `1`, keeps tracker state off GPU memory when possible.
- `UVR_HOTRACK_STATE_WINDOW`: number of recent frames to keep in memory, default `9`.
- `UVR_HOTRACK_LOG_MEMORY`: set `1` to print GPU memory per frame.
