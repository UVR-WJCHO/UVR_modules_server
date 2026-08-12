# Handover — meshalignment replacement

Written when `modules/meshalignment/` and `modules/modules_jointtrack.py` were
replaced. Everything that was here before is superseded and recoverable from
git (`git show HEAD:modules/meshalignment/...`).

## Why it was replaced, not patched

The old pipeline scored coarse pose hypotheses geometrically and only brought
texture in to re-rank a shortlist. For a part that is nearly a surface of
revolution that ranks the right turn no better than the wrong ones, so the
right answer was often gone before texture could speak.

Worse, the foreground mask thresholded brightness on `rgb_masked`. The
backgrounds in this data are attenuated, not erased, and the objects have large
genuinely-black regions — so a black band on a part was read as background, and
because it reached the silhouette boundary, hole-filling could not restore it.
One part's mask covered 62% of its own convex hull. **Every F1/IoU the old
pipeline reported for that frame was measured against a broken target.**
Fixing the mask alone took that part from IoU 0.477 to 0.920.

## What is here now

`modules_jointtrack.py` runs fit → assemble → handover in one command:

    python modules_jointtrack.py \
        --data_dir  /path/to/captures \
        --parts     2,3,01 \
        --assembly  23:2,3 \
        --assembly  012:01,2 \
        --order     01,2,3 \
        --output_dir /path/to/out

    out/stage1/            poses, per-part fit images, and meshes/ (corrected)
    out/C23/ out/C012/     per-assembly poses, side_*.png, views_*.png
    out/inputs/glbs/mesh_0.glb …
    out/inputs/transforms.json

`out/inputs/` is what `metaobj_wrapper/combine_rocket_glb.py` reads.

## Verified

End-to-end on the sample captures, on `uvr_integ` (torch 2.7.0+cu128,
nvdiffrast 0.3.3, RTX 5090):

| case | units | union IoU | depth | rgb | joint centres | joint planes | overlap |
|---|---|---|---|---|---|---|---|
| C23  | 2+3   | 0.97 | 2.6 mm | 0.18 | 0.9 mm | 0.10° | 1.1 mm |
| C012 | 01+2  | 0.94 | 4.3 mm | 0.20 | 0.7 mm | 0.79° | 1.1 mm |

Both assemblies chain through unit 2, and the bridge scale comes out 1.00000 —
the two solves agree on that part's size, which is a free check that the chain
is sound rather than merely consistent.

Reference results and per-stage images are in
`_extra/metaobj_alignment/final/`, produced by the same code.

## Open

1. **`combine_rocket_glb.py` requires exactly 5 parts**; this writes as many as
   there are units — 3 for the current data. That script needs changing, or the
   units need padding. Deliberately left alone.
2. **Unit → `mesh_N.glb` order** is whatever `--order` says. Nothing checks it
   against what the wrapper expects per index.
3. **Multi-part units** (`--assembly 012:0+1,2`) need `--init_dir/--init_cid`
   naming an earlier solve that fixed their internals. Untested since the
   current data uses the reconstructed `mesh_01` instead.
4. **No caller.** Nothing in the repo imports this; it is a CLI entry point, as
   the old one was.

## Things already tried that did not work

Recorded so they are not repeated.

- **Inferring a part's diameter at the joint.** The same step is closed by
  thinning the part that was already right, and the optimiser takes whichever
  is cheaper — it shrank a correctly-sized part by 11% and drove 8.5 mm of
  interpenetration. Diameter is settled in stage 1, where the part is fully
  visible, and stage 2 does not touch it (`--lock_scale`).
- **Correcting girth twice.** Stage 2 used to re-derive a correction from the
  width stage 1 reports *after* correcting. The two stages measure width
  differently — stage 1 across bands perpendicular to the axis, the reported
  figure from a bounding box that mixes axial and radial extent when the part
  is tilted — and they disagreed in sign. The second correction cannot be baked
  into the mesh, so it survived inside the pose as a scale differing across and
  along the axis, which `transforms.json` cannot express. `assemble.py` now
  refuses such a pose rather than rounding it off.
- **Weighting joint terms by feel.** Coaxial at 1.0, seam at 4000, rim at 3000
  each closed their own gap and broke the image fit — depth error went from
  3.9 mm to 24.6 mm in one case. Weights are set by matching cost per
  millimetre against the depth term, not by trial.
- **Judging a result by union IoU.** It rose while an assembly went from right
  to wrong, and fell while another went from wrong to right. It cannot see
  which part is where. `rgb_err`, per-unit visibility, and the joint numbers
  are what tracked correctness; `views_*.png` is the check on the side the
  capture never showed.
