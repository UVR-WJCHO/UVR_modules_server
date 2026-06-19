# Interactive HoTrack Stage 1

This branch wires the hograph_plus Hotrack track-only flow into the existing `main_all.py` entry point.

## What It Does

- Receives live HL2 RGB/depth frames through the existing `Hl2Manager` loop.
- Automatically detects hand-object interactions with the hograph_plus `weights/yolo_100doh_best.pt` detector (`targetobject/hand_*` classes).
- Calls `Hotrack.process_frame_with_tracking()` directly and applies the same hograph_plus track-only runtime flags (`track_hand_masks=False`, `skip_existing_object_box_prompts=True`).
- Saves every tracked/edited mask as files under `output/hotrack_stage1/<video_name>/`.
- Allows interactive mask selection, deletion, current-frame removal, and brush edits.

It intentionally disables hograph/graph execution and sanitizes structural merge/split/id-transition fields, matching hograph_plus `runtime_mode=track_only`.

## Run

Place the hograph_plus YOLO checkpoint and a SAM2.1 checkpoint in ignored local paths:

```bash
mkdir -p weights segmentor/sam2_realtime/checkpoints
# weights/yolo_100doh_best.pt
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
  track_only_summary.json
  tracking/json/000000_online_track.json
  masks_png/000000_online/id_000100.png
  overlays/000000_online.jpg
  000000_online_masks.json
```

Masks are saved as single-channel PNG files for object ids (`>=100`). The tracking JSON is the hograph_plus Hotrack payload with `runtime_mode=track_only` and empty `struct_events`, `id_transitions`, and `id_remap`.

## Useful Environment Variables

- `UVR_HOTRACK_OUTPUT_DIR`: base output directory, default `output/hotrack_stage1`.
- `UVR_HOTRACK_VIDEO_NAME`: run name, default `hl2_online`.
- `UVR_HOTRACK_YOLO_MODEL`: hograph_plus HO detector checkpoint, default `weights/yolo_100doh_best.pt`.
- `UVR_HOTRACK_SAM2_VARIANT`: `tiny`, `small`, `base_plus`, or `large`; default `tiny`.
- `UVR_HOTRACK_SAM2_CHECKPOINT`: explicit SAM2 checkpoint path.
- `UVR_HOTRACK_MAX_SIDE`: optional resize long side before Hotrack/SAM2, default `0` (disabled, hograph_plus-compatible).
- `UVR_HOTRACK_HO_THRESH_HAND`: hand detector threshold, default `0.55`.
- `UVR_HOTRACK_HO_THRESH_OBJ`: object detector threshold, default `0.55`.
- `UVR_HOTRACK_TARGET_CONTACT`: target hand contact code, default `P`.
- `UVR_HOTRACK_BACKFILL_WINDOW`: Hotrack backfill history length for online mode, default `120`.
- `UVR_HOTRACK_SAVE_TRACKING_JSON`: save hograph_plus tracking JSON, default `1`.
- `UVR_HOTRACK_DINO_ALLOW_DOWNLOAD`: allow transformers to download DINO weights if needed, default `0`.

If the checkpoints already live in a sibling hograph_plus workspace, set `HOGRAPH_PLUS_ROOT=/path/to/hograph_plus`; the wrapper falls back to `weights/yolo_100doh_best.pt` and `third_party/sam2_realtime/checkpoints/*.pt` there when local files are missing.
