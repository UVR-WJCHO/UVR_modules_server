# Batch Mesh Reconstruction Design

## Goal

Add a standalone command-line program that reconstructs a TRELLIS mesh for
every matching timestamped capture directory and writes `mesh.glb` back into
that same directory.

## Scope

- The default capture root is `output`.
- The default directory pattern is `20260715_*`.
- Each matching directory uses `rgb_masked.png` as its only reconstruction
  input, matching the `flag_recon_mesh=True` path in `main_meshrecon.py`.
- The output is always `<capture-directory>/mesh.glb`.
- An existing `mesh.glb` is overwritten.
- The HoloLens stream, segmentation, UDP notification, texture painting, and
  behavior-property estimation are outside this command's scope.

## Architecture

Create a repository-root executable module named `batch_meshrecon.py`. It
adds the existing `modules` directory to `sys.path`, imports
`MeshReconstructor`, discovers capture directories, loads each masked image as
a PIL RGB image, calls `MeshReconstructor.run()`, and exports the returned GLB.

The TRELLIS model is initialized exactly once per command invocation and is
reused for every capture directory. Directory discovery and per-directory
processing are separate functions so their behavior can be tested without
loading GPU models.

## Command-Line Interface

The command is run from the repository root:

```bash
conda activate metaobj
python batch_meshrecon.py
```

Supported options:

- `--output-root`: capture root directory, default `output`.
- `--pattern`: timestamp-directory glob, default `20260715_*`.
- `--modelpath`: TRELLIS model path, default
  `pretrained/meshrecon/diffusion`.

The default invocation processes the seven currently present
`output/20260715_*` directories.

## Data Flow

For each matching directory, sorted by directory name:

1. Resolve `<directory>/rgb_masked.png`.
2. If the image is missing, record a failure and continue.
3. Open the image with Pillow and convert it to RGB.
4. Pass the image to the shared `MeshReconstructor` instance.
5. Export the returned object to `<directory>/mesh.glb`, replacing an existing
   file at that path.
6. Release per-item references and clear the CUDA cache when CUDA is available.

## Error Handling and Exit Status

- No matching directories: print a clear message and return a nonzero exit
  status without initializing TRELLIS.
- Missing `rgb_masked.png`: report that directory as failed and continue.
- Reconstruction or export failure: report the exception for that directory
  and continue with the remaining directories.
- Print a final processed/succeeded/failed summary.
- Return exit status `0` only when at least one directory was processed and all
  matching directories succeeded; otherwise return `1`.

## Test Strategy

Tests use temporary capture directories and dependency injection. A fake
reconstructor records received images and returns a fake GLB exporter, so tests
do not import or initialize TRELLIS.

Tests cover:

- matching directories are sorted and non-directories are excluded;
- `rgb_masked.png` is converted to RGB and exported as `mesh.glb`;
- an existing `mesh.glb` is overwritten through the exporter;
- a missing input is reported while later directories still run;
- no matches avoids model construction and returns failure;
- mixed success/failure returns failure with accurate counts.

## Non-Goals

- Parallel GPU reconstruction.
- Automatic retry or resume state.
- Renaming capture files.
- Generating texture-painted or behavior-property outputs.
- Changing `main_meshrecon.py` or `modules/modules_mesh.py` behavior.
