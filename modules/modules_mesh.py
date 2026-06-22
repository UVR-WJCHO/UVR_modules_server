import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
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
        # Load a pipeline from a model folder or a Hugging Face model hub.
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(modelpath)
        self.pipeline.cuda()

    def run(self, image):
        outputs = self.pipeline.run(image, seed=1,
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
        return glb


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





