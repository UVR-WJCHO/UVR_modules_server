HoTrack runs inside the existing `modules/hotrack/hotrack.py::Hotrack` class.
`modules/modules_hotrack.py` keeps the interactive wrapper and calls the same
`process_frame_with_tracking()` entry point. No separate tracking package or
second Conda environment is required for the server's tracking entry point.

The default interactive wrapper selects the observation-only branch before the
legacy structural pipeline. It detects portable-object contact (`P`), seeds a
SAM2 object mask, and propagates its ID through subsequent frames. Existing
objects skip repeated box prompts. New masks use HoGraph's detector-projected
admission to avoid registering an already tracked object again. Lost objects
are not reassigned by DINO. The legacy structural implementation remains in the
class for existing callers that explicitly enable it.

Source: [sys7498/hograph](https://github.com/sys7498/hograph), commit
`bde9c9d2fc7f3778c65099260dbcf04e01b06705`.

| Original source | Server destination and adaptation |
| --- | --- |
| `hograph/tracking/frame_stages.py::InteractionDetectionPolicy`, `InteractionDetectionResult` | Same data classes in `modules/hotrack/hotrack.py` |
| `InteractionDetectionStage.run` | `Hotrack._detect_handheld_objects`; detector and target-suppressor calls use the existing server functions |
| `InteractionDetectionStage._best_matches_by_object`, `_collect_targets`, `_filter_oversize_targets`, `_hand_proxy_masks` | Same static methods on `Hotrack`, copied without algorithm changes |
| `hograph/tracking/identity_admission.py::ProjectedIdentityAdmission`, `_project_mask`, `detector_projected_identity_admission` | Same data class and static methods in `hotrack.py`; the projection helper is called through `Hotrack` |
| `Hotrack.process_frame_with_tracking` and `ObservationStage.run` | `_process_frame_tracking_only` adapts the init/add/get/prompt sequence to the server's flat state attributes; separation, joining, replay, graph projection and ID rewrites are omitted |
| `third_party/sam2_realtime/sam2/sam2_realtime_predictor.py::init_state`, `add_frame` | Same methods in the server's SAM2 fork, including device-aware frame loading |
| `third_party/sam2_realtime/sam2/utils/misc.py::load_frame`, `get_connected_components` | Frame loading and OpenCV fallback copied; the server retains its CUDA extension fast path when available |

The detector functions already match the source commit. SAM2 uses one canonical
`sam2` import and Hydra config module. The server's existing Tiny/Small/Large
configs remain available. Old frame tensors are released even before the first
object is detected; cleared image slots are not scanned again on every frame.

Create the environment from the repository root:

```bash
conda env create -f environment-metaboj_hotrack.yml
conda activate metaboj_hotrack
```

For an existing environment:

```bash
conda env update -n metaboj_hotrack -f environment-metaboj_hotrack.yml
```

The manifest targets Linux/NVIDIA with PyTorch 2.7.0 and CUDA 12.8 wheels. The
macOS markers install CPU-capable wheels for source inspection. The extracted
methods do not install the full HoGraph paper runner or its DINO dependencies.
TRELLIS, mesh-alignment and behavior dependencies remain separate from this
tracking manifest; add the existing reconstruction dependencies to the same
environment when enabling that pipeline.

Tracking requires two checkpoint files at runtime:

| Checkpoint | Default path |
| --- | --- |
| Custom 100DOH YOLO | `pretrained/object/yolo_100doh_best.pt` |
| SAM 2.1 Tiny | `modules/segmentor/sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt` |

Set `UVR_HOTRACK_YOLO_MODEL` and `UVR_HOTRACK_SAM2_CHECKPOINT` for files stored
elsewhere. `UVR_HOTRACK_SAM2_VARIANT` must match the SAM checkpoint (`tiny`,
`small`, `base_plus`, or `large`). `HOGRAPH_PLUS_ROOT` is also supported for
existing HoGraph weight directories; a sibling `hograph` checkout is searched
when that variable is unset. The checkpoint paths are checked before loading.
No model files are downloaded automatically by this integration.

To run tracking and RGB/depth captures without TRELLIS:

```bash
python comm_hub.py --port 37001
# In another terminal in the same Conda environment:
python main_metaobjrecon.py --host 127.0.0.1 --port 37001 --track-only
```

`SPACE` saves the chosen object's capture. `a` saves an assembly capture;
`ENTER` alignment is disabled in `--track-only`. The normal command retains
reconstruction, manual assembly, export and mesh-result upload. HoGraph's
automatic structural operations are disabled in both interactive paths.
Tracking-only mode also accepts RGB packets without depth; saving a capture
requires depth from the same packet.

Click an object to select it for capture. If that object disappears, a capture
is refused instead of switching objects. Without an explicit selection, the
wrapper uses the object that overlaps a current hand-associated detector box.
One visible object may continue after release; several unselected objects
without a current hand match require a click. `[` and `]` also cycle selection.
The mask is resized back to the RGB/depth capture resolution with nearest-neighbor
interpolation. Changing input resolution requires resetting the tracker.

Per-frame mask and tracking JSON dumps stay off by default. Enable
`UVR_HOTRACK_SAVE_MASKS=1` or `UVR_HOTRACK_SAVE_TRACKING_JSON=1` for later
inspection. The wrapper raises an error if structural results unexpectedly
appear; it does not hide them by rewriting output files.

Validation for this change is limited to parsing, lint and imports without model
construction. Model inference, HL2 streaming, CUDA performance and the full
reconstruction pipeline are deferred at the user's request.
