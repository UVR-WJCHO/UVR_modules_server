# meshalignment

Fit reconstructed part meshes to RGB-D captures, place them into captures of
the assembled object, and hand `metaobj_wrapper` the meshes and transforms it
needs. Driven by `../modules_jointtrack.py`.

## Two stages

**Fit** (`solo.py`, `calibrate.py`) puts each mesh into the capture it was
reconstructed from. Every orientation hypothesis is rendered and scored with
the full objective, texture included — a part that is nearly a surface of
revolution has almost no silhouette signal about its turn, so a screen that
looks at geometry first ranks the right answer no higher than the wrong ones
and drops it before texture is ever consulted.

Girth is corrected here too. A reconstruction can come out the right length and
the wrong width, and that is only unambiguous while the part is alone and fully
in view. The correction scales distances from the part's central axis and
leaves distances along it, then is baked into the mesh — so **the pose belongs
to the corrected mesh in `<output>/stage1/meshes/`, not to the original.**

**Assemble** (`joint.py`) places those parts into a capture of them joined.
Parts are composited through a shared z-buffer before being compared, so each
pixel is judged against whatever is actually visible there rather than against
something hidden behind it. They go in one at a time, most visible first, each
searched against the region the placed ones leave unexplained.

The search is confined to what the capture can decide: tipping away from the
Stage 1 attitude is bounded, while turning about the part's own axis is not,
and is held where Stage 1 put it when the shape cannot resolve it. Held turns
are settled afterwards against the photograph alone, about the axis through the
mating circle's centre, which leaves the joint untouched.

## The joint

Parts of an assembly meet on circular faces, so the joint is a circle-to-circle
registration (`geom.rim_circles`, `joint._rim_match`): the two mating circles
have to become one, in centre, radius and plane. Seating, coaxiality and the
step at the seam are then a single constraint instead of three soft penalties
pulling against each other — which is what earlier attempts were, and they
could not be balanced without one always winning.

Two details make it work. Faces that mate look at each other, so their outward
normals oppose; that is what tells a top from a bottom, and without it a part
that has slid inside its neighbour pairs the wrong ends and stays inside.
And the weight is set so a millimetre of joint error costs what a millimetre of
depth error costs — the circles cannot feel the assembly tipping toward the
camera, and only depth can, so a joint term that outbids it walks the whole
stack off the measurement.

## Layout

    frames.py     loading, foreground, depth, support plane
    geom.py       Sim(3), rotations, central axis, rim circles, symmetry
    render.py     differentiable rasterisation (nvdiffrast)
    score.py      the fit objective — screening and refining use the same one
    sdf.py        per-mesh distance field, for shared-volume tests
    solo.py       stage 1
    calibrate.py  girth correction
    joint.py      stage 2
    assemble.py   chaining solves, transforms.json, staging the meshes
    viz.py        result images

## Input

Per frame id, either layout:

    <data_dir>/part_<fid>/{rgb,rgb_masked,depth,intrinsic,mesh}.*
    <data_dir>/{rgb,rgb_masked,depth,intrinsic,mesh}_<fid>.*

`rgb_masked` may attenuate the background rather than erase it. The mask reads
that by comparing against `rgb` — background keeps a fraction of its brightness,
foreground is untouched — and recovers parts of the object that are genuinely
black, which carry no evidence either way, by depth. A brightness threshold
alone discards them silently, and if such a region reaches the silhouette
boundary, hole-filling cannot put it back.

Needs a CUDA GPU. `ninja` must be on PATH or nvdiffrast cannot build its
kernels and silently falls back.
