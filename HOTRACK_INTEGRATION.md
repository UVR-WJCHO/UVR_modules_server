HoTrack runs inside the existing `modules/hotrack/hotrack.py::Hotrack` class.
`modules/modules_hotrack.py` keeps the interactive wrapper and calls the same
`process_frame_with_tracking()` entry point. `metaboj_hotrack` covers the complete
tracking, reconstruction, alignment and GLB-export path in one Conda environment.

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

Install the complete pipeline from the repository root on the Linux x86_64/NVIDIA
execution host. Anaconda/Miniconda, Git, a compatible NVIDIA driver and the CUDA
12.8 toolkit (including `nvcc` and headers) must already be available. Set
`CUDA_HOME` if the toolkit is not on `PATH`.

```bash
python scripts/setup_metaboj_hotrack.py
conda activate metaboj_hotrack
```

The installer creates or updates exactly one environment, `metaboj_hotrack`.
It installs `environment-metaboj_hotrack.yml` first, including Python 3.11,
PyTorch 2.7.0/cu128, xFormers, spconv and the Python dependencies for HoTrack,
TRELLIS, alignment, GLB export and the existing behavior/teaser path. It then
builds the native renderers against that installed PyTorch in the same environment
with pip build isolation disabled. `conda env create` by itself installs only the
first phase; use the installer to include the native phase too.
`constraints-metaboj_hotrack.txt` keeps native installation from replacing the
shared PyTorch ABI or image-library versions.
Native runtime dependencies are included in the base manifest. Native packages
are reinstalled without dependency changes or cached wheels, so updating an
existing environment does not retain binaries built against an older PyTorch.

| Installation phase | Files and packages |
| --- | --- |
| Conda/compiler and Python runtime | `environment-metaboj_hotrack.yml`, `requirements-metaboj_hotrack.txt` |
| Native renderers and mesh operations | `requirements-metaboj_hotrack-native.txt`: nvdiffrast, PyTorch3D, mip-splatting Gaussian rasterizer, diffoctreerast and the matching Kaolin wheel |
| SAM2 | The existing `modules/segmentor/sam2_realtime` source, installed editable with its CUDA extension |

The source dependencies use fixed commits. `utils3d` comes from the
EasternJournalist repository used by TRELLIS; the unrelated PyPI package with
that name is not used. The xFormers and Kaolin wheels match the selected PyTorch
and CUDA versions. spconv uses its CUDA 12.6 distribution (there is no cu128
package) with the CUDA 12.x driver/runtime stack. The environment selects
xFormers attention and the spconv sparse backend, as `modules_mesh.py` expects.
The former tracking and reconstruction requirements files both include the
same shared requirements, rather than pinning conflicting versions separately.

`rembg` declares the headless OpenCV distribution. The installer restores the
matching GUI OpenCV wheel after all other packages so the existing `cv2.imshow`
windows remain available. A graphical session/display is still required for
those windows. The installer finishes with a package metadata check. It does
not construct a model, access an API, download a checkpoint or start an inference
test. It can be rerun to finish a stopped installation in the same environment.

Checkpoints remain outside installation and Git. HoTrack needs:

| Checkpoint | Default path |
| --- | --- |
| Custom 100DOH YOLO | `pretrained/object/yolo_100doh_best.pt` |
| SAM 2.1 Tiny | `modules/segmentor/sam2_realtime/checkpoints/sam2.1_hiera_tiny.pt` |

Set `UVR_HOTRACK_YOLO_MODEL` and `UVR_HOTRACK_SAM2_CHECKPOINT` for files stored
elsewhere. `UVR_HOTRACK_SAM2_VARIANT` must match the SAM checkpoint (`tiny`,
`small`, `base_plus`, or `large`). `HOGRAPH_PLUS_ROOT` is also supported for
existing HoGraph weight directories; a sibling `hograph` checkout is searched
when that variable is unset. TRELLIS and its image-conditioning weights are
also required for reconstruction; see [WEIGHTS.md](WEIGHTS.md) for the existing
server paths. Checkpoint preparation is deferred.

Run the normal server entry point in this environment:

```bash
conda activate metaboj_hotrack
python comm_hub.py --port 37001
# In another terminal, also activate metaboj_hotrack:
python main_metaobjrecon.py --host 127.0.0.1 --port 37001
```

No additional mode flag is required. The loop receives synchronized RGB/depth,
tracks the hand-associated object and feeds its mask into the existing capture
and reconstruction code. `SPACE` reconstructs a unit with TRELLIS and writes
`mesh.glb`; `a` records an assembly capture; `ENTER` aligns and combines the
captured meshes, exports the metaobj and uploads `MESH_RESULT` through comm_hub.
The existing behavior step stays enabled by its original `flag_behavior`.
HoGraph's automatic joining/separation and graph/ID rewrites are excluded from
the tracker; the server's manual assembly and mesh combination stay enabled.

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

Validation for this update covers Python syntax/lint, static inspection of the
default reconstruction path, Linux Conda solving in dry-run mode and Linux
Python-package metadata resolution. The Linux/CUDA
installation and source builds have not been run on this Mac. No checkpoint
download, inference, HL2, CUDA performance or full reconstruction test was run;
those checks are deferred at the user's request.
