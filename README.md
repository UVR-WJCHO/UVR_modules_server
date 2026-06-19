# UVR Modules Server

This branch adds the online interactive HoTrack stage-1 tracker to the existing `main_all.py` flow.

## Main Entry Point

```bash
python main_all.py
```

By default, `main_all.py` now uses `InteractiveHoTrackSegmentor`, a thin online wrapper around the hograph_plus `Hotrack.process_frame_with_tracking()` track-only flow. It automatically detects hand-object interactions, tracks object masks, sanitizes structural fields, and saves masks/overlays/metadata under `output/hotrack_stage1/hl2_online/`.

To run the previous legacy segmentor path:

```bash
UVR_USE_INTERACTIVE_HOTRACK=0 python main_all.py
```

## Required Checkpoint

The hograph_plus track-only detector expects the `targetobject/hand_*` YOLO checkpoint, not the legacy `segmentor/100DOH_small.pt` interaction classifier. Place the weights locally before running:

```bash
mkdir -p weights segmentor/sam2_realtime/checkpoints
# expected default paths
# weights/yolo_100doh_best.pt
# segmentor/sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt
```

Set `UVR_HOTRACK_YOLO_MODEL=/path/to/yolo_100doh_best.pt` or `UVR_HOTRACK_SAM2_CHECKPOINT=/path/to/checkpoint.pt` if either checkpoint lives elsewhere.

See [HOTRACK_STAGE1.md](HOTRACK_STAGE1.md) for controls, output layout, and tuning environment variables.

If the checkpoints already live in a sibling hograph_plus workspace, set `HOGRAPH_PLUS_ROOT=/path/to/hograph_plus`; the wrapper falls back to `weights/yolo_100doh_best.pt` and `third_party/sam2_realtime/checkpoints/*.pt` there when local files are missing.
