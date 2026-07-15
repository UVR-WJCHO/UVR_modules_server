import os
os.environ.setdefault('ATTN_BACKEND', 'xformers')  # Can be 'flash-attn' or 'xformers'.
os.environ.setdefault('SPCONV_ALGO', 'native')     # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import cv2
import numpy as np
from PIL import Image
import aspose.threed as a3d

from meshrecon.trellis.pipelines import TrellisImageTo3DPipeline
from meshrecon.trellis.utils import render_utils, postprocessing_utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MeshReconstructor():
    def __init__(self, modelpath='pretrained/meshrecon/diffusion'):
        if modelpath.startswith(("pretrained/", "./", "../", "/")):
            config_path = os.path.join(modelpath, "pipeline.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Mesh reconstruction weights not found: {config_path}. "
                    "Place the files listed in WEIGHTS.md under pretrained/meshrecon/."
                )

        # Load a pipeline from a model folder or a Hugging Face model hub.
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(modelpath)
        self.pipeline.cuda()

    def _preprocess_rgba_graybg(self, image, bg=128, resolution=518):
        """For an RGBA input (mask in alpha), crop to the object (like TRELLIS's
        own preprocess) and composite over a neutral GRAY background instead of
        black. TRELLIS forces a black background (RGB*alpha), which makes an
        object's own dark/black regions merge into it -> TRELLIS then adds a flat
        ground-plane slab. A gray background keeps object colors untouched but
        keeps dark parts distinguishable, removing the slab."""
        rgba = np.array(image)
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        ys, xs = np.nonzero(alpha > 0.8)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = int(max(x1 - x0, y1 - y0) * 1.2)
        bx0, by0 = int(cx - size // 2), int(cy - size // 2)
        bx1, by1 = bx0 + size, by0 + size
        H, W = alpha.shape
        pad = ((max(0, -by0), max(0, by1 - H)), (max(0, -bx0), max(0, bx1 - W)))
        oy, ox = pad[0][0], pad[1][0]
        rgb = np.pad(rgb, pad + ((0, 0),))[by0 + oy:by1 + oy, bx0 + ox:bx1 + ox]
        alpha = np.pad(alpha, pad)[by0 + oy:by1 + oy, bx0 + ox:bx1 + ox]
        rgb = cv2.resize(rgb, (resolution, resolution))
        alpha = cv2.resize(alpha, (resolution, resolution))[..., None]
        comp = rgb * alpha + float(bg) * (1.0 - alpha)  # object unchanged; bg -> gray
        return Image.fromarray(comp.clip(0, 255).astype(np.uint8))

    def run(self, image, return_preview=False, preview_views=2, preview_resolution=384):
        # RGBA (mask in alpha) -> gray-bg preprocess + skip TRELLIS's black forcing
        # (avoids the dark-object-on-black -> floor-slab artifact). RGB -> default.
        if getattr(image, "mode", None) == "RGBA":
            cond_image = self._preprocess_rgba_graybg(image)
            do_preprocess = False
        else:
            cond_image = image
            do_preprocess = True
        outputs = self.pipeline.run(cond_image, seed=1,
            preprocess_image=do_preprocess,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 25,    # 12, 25
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 15,    # 10, 25
                "cfg_strength": 3,
            },
        )


        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,  # Ratio of triangles to remove in the simplification process
            texture_size=1024,  # Size of the texture used for the GLB
            bake_mode='opt'    # 'fast', 'opt'
        )
        if return_preview:
            preview = self._render_preview(glb, preview_views, preview_resolution)
            return glb, preview
        return glb

    def _render_preview(self, glb, num_views=2, resolution=384):
        """Render the reconstructed mesh geometry from a few evenly-spaced yaws
        into one side-by-side image, for a quick 'did it reconstruct?' check.
        Uses matplotlib (CPU/Agg): TRELLIS's GPU renderer crashes natively in
        this environment, and geometry from 2 views is enough to verify shape."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import trimesh

        mesh = glb.dump(concatenate=True) if isinstance(glb, trimesh.Scene) else glb
        v = np.asarray(mesh.vertices, dtype=float)
        f = np.asarray(mesh.faces)
        center = (v.max(0) + v.min(0)) / 2
        radius = (v.max(0) - v.min(0)).max() / 2 + 1e-8

        dpi = 100
        fs = resolution / dpi
        frames = []
        for azim in np.linspace(0, 360, num_views, endpoint=False):
            fig = plt.figure(figsize=(fs, fs), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            # glTF is Y-up; map mesh Y -> plot vertical (Z) so the object stands up.
            ax.plot_trisurf(v[:, 0], v[:, 2], v[:, 1], triangles=f,
                            color=(0.72, 0.72, 0.78), edgecolor="none", shade=True)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(center[1] - radius, center[1] + radius)
            ax.view_init(elev=15, azim=azim)
            ax.set_axis_off()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
            plt.close(fig)
        return np.concatenate(frames, axis=1)  # RGB uint8


def main():
    import time
    savepath = './outputs'
    modelpath = 'pretrained/meshrecon/diffusion'
    condpath = './examples/part0_1.jpg'
    savepath = os.path.join(savepath, condpath.split('/')[-1].split('.')[0])

    os.makedirs(savepath, exist_ok=True)

    meshrecon = MeshReconstructor(modelpath)

    image = Image.open(condpath)

    # Run the pipeline
    t1 = time.time()
    print("start inference")
    mesh_glb = meshrecon.run(image)
    print("done inference")
    print("inference time :", time.time() - t1)

    t2 = time.time()
    save_mesh_path = os.path.join(savepath, "output.glb")
    mesh_glb.export(save_mesh_path)

    scene = a3d.Scene.from_file(save_mesh_path)
    options = a3d.formats.FbxSaveOptions(a3d.FileFormat.FBX7500_BINARY)
    print("mesh extract time :", time.time() - t2)

    scene.save(os.path.join(savepath, "output.fbx"), options)
    print("save done")

if __name__ == '__main__':
    main()



