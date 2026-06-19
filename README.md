# UVR Modules Server

This branch adds the online interactive HoTrack stage-1 tracker to the existing `main_all.py` flow.

## Main Entry Point

```bash
python main_all.py
```

By default, `main_all.py` now uses `InteractiveHoTrackSegmentor` to automatically detect hand-object interactions, track only the interacting object masks, and save masks/overlays/metadata under `output/hotrack_stage1/hl2_online/`.

To run the previous legacy segmentor path:

```bash
UVR_USE_INTERACTIVE_HOTRACK=0 python main_all.py
```

## Required Checkpoint

The YOLO interaction detector is already stored at `segmentor/100DOH_small.pt`. SAM2 checkpoints are intentionally ignored by git, so place one locally before running:

```bash
mkdir -p segmentor/sam2_realtime/checkpoints
# expected default path
# segmentor/sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt
```

Set `UVR_HOTRACK_SAM2_CHECKPOINT=/path/to/checkpoint.pt` if the checkpoint lives elsewhere.

See [HOTRACK_STAGE1.md](HOTRACK_STAGE1.md) for controls, output layout, and tuning environment variables.
