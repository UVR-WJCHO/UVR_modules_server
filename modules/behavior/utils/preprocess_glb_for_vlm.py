import torch
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import trimesh
from PIL import Image as PILImage
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    HardFlatShader,
    TexturesVertex,
    TexturesUV,
    AmbientLights,
)


def get_location(foreground_mask):
    """Return bounding box [x, y, w, h] covering ALL foreground pixels."""
    ys, xs = np.where(foreground_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w = int(xs.max()) - x + 1
    h = int(ys.max()) - y + 1
    return [x, y, w, h]


def _extract_part_data(geom):
    visual = geom.visual
    flat_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    uvs = None
    tex_pil = None
    
    if isinstance(visual, trimesh.visual.TextureVisuals):
        mat = getattr(visual, 'material', None)
        if mat is not None:
            base = getattr(mat, 'baseColorFactor', None)
            if base is not None:
                flat_rgb = np.array(base[:3], dtype=np.float32) / 255.0

            raw_uvs = getattr(visual, 'uv', None)
            tex = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
            if raw_uvs is not None and len(raw_uvs) > 0 and tex is not None:
                try:
                    uvs = raw_uvs.astype(np.float32)
                    tex_pil = tex.convert('RGB')
                except Exception as e:
                    pass

    return {
        'vertices': geom.vertices.astype(np.float32),
        'faces':    geom.faces.astype(np.int64),
        'uvs':      uvs,
        'tex_pil':  tex_pil,
        'flat_rgb': flat_rgb,
    }


def build_texture_atlas(parts, atlas_height=2048):
    strips = []
    for part in parts:
        if part['uvs'] is not None and part['tex_pil'] is not None:
            tex = part['tex_pil']
            ow, oh = tex.size
            new_w = max(1, int(ow * atlas_height / oh))
            resized = tex.resize((new_w, atlas_height), PILImage.LANCZOS)
            strip = np.array(resized, dtype=np.float32) / 255.0
        else:
            color = part['flat_rgb']
            strip = np.tile(color, (atlas_height, 4, 1)).astype(np.float32)
        strips.append(strip)

    atlas_np = np.concatenate(strips, axis=1)
    atlas_W = atlas_np.shape[1]

    remapped_uvs = []
    x_off = 0
    for i, part in enumerate(parts):
        sw = strips[i].shape[1]
        u_start = x_off / atlas_W
        u_scale = sw / atlas_W

        if part['uvs'] is not None:
            uvs = part['uvs'].copy()
            uvs[:, 0] = np.clip(uvs[:, 0] * u_scale + u_start, 0.0, 1.0)
            uvs[:, 1] = np.clip(uvs[:, 1], 0.0, 1.0)
        else:
            n = len(part['vertices'])
            u_c = (x_off + sw / 2.0) / atlas_W
            uvs = np.full((n, 2), [u_c, 0.5], dtype=np.float32)

        remapped_uvs.append(uvs)
        x_off += sw

    return atlas_np, remapped_uvs


def create_vlm_dataset_from_glb(glb_path, output_base, case_name="scene", dpi=75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading GLB scene...")
    scene = trimesh.load(glb_path, force='scene')

    parts = []
    part_id = 1

    print("Extracting geometry and textures...")
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geom = scene.geometry[geometry_name].copy()
        geom.apply_transform(transform)

        if len(geom.vertices) == 0:
            continue
        data = _extract_part_data(geom)
        data['id']   = part_id
        data['name'] = node_name
        parts.append(data)
        part_id += 1

    if not parts:
        print("No geometries found")
        return

    print("\nBuilding texture atlas...")
    atlas_np, remapped_uvs = build_texture_atlas(parts)
    atlas_t = torch.tensor(atlas_np, dtype=torch.float32, device=device).unsqueeze(0)

    #-- Combine all geometry for scene rendering --#
    all_verts_np   = []
    all_faces_np   = []
    all_uvs_np     = []
    all_mask_np    = []
    part_info      = []
    vertex_count   = 0

    for i, part in enumerate(parts):
        p_id = part['id']
        v    = part['vertices']
        f    = part['faces']
        uvs  = remapped_uvs[i]

        r = (p_id % 256) / 255.0
        g = ((p_id // 256) % 256) / 255.0
        b = ((p_id // 65536) % 256) / 255.0
        mask_color = np.tile([r, g, b], (len(v), 1)).astype(np.float32)

        all_verts_np.append(v)
        all_faces_np.append(f + vertex_count)
        all_uvs_np.append(uvs)
        all_mask_np.append(mask_color)
        part_info.append({'id': p_id, 'name': part['name'], 'color': [r, g, b]})
        vertex_count += len(v)

    all_verts = np.concatenate(all_verts_np, axis=0)
    center = all_verts.mean(axis=0, keepdims=True)
    scale = np.abs(all_verts - center).max()
    
    for part in parts:
        part['vertices'] = (part['vertices'] - center) / scale
    
    all_verts = (all_verts - center) / scale

    verts_t = torch.tensor(all_verts, dtype=torch.float32, device=device)
    faces_t = torch.tensor(np.concatenate(all_faces_np, axis=0), dtype=torch.int64, device=device)
    uvs_t   = torch.tensor(np.concatenate(all_uvs_np, axis=0), dtype=torch.float32, device=device)
    mask_t  = torch.tensor(np.concatenate(all_mask_np, axis=0), dtype=torch.float32, device=device)

    mesh_rgb = Meshes(
        verts=[verts_t], faces=[faces_t], 
        textures=TexturesUV(maps=atlas_t, faces_uvs=faces_t.unsqueeze(0), verts_uvs=uvs_t.unsqueeze(0))
    )
    mesh_mask = Meshes(verts=[verts_t], faces=[faces_t], textures=TexturesVertex(verts_features=[mask_t]))

    #-- Mesh for each part --#
    part_meshes = {}
    for i, part in enumerate(parts):
        v = torch.tensor(part['vertices'], dtype=torch.float32, device=device)
        f = torch.tensor(part['faces'], dtype=torch.int64, device=device)
        u = torch.tensor(remapped_uvs[i], dtype=torch.float32, device=device)
        tex = TexturesUV(maps=atlas_t, faces_uvs=f.unsqueeze(0), verts_uvs=u.unsqueeze(0))
        part_meshes[part['id']] = Meshes(verts=[v], faces=[f], textures=tex)

    #-- Camera setup --#
    image_size = 2048
    fov        = 60
    worst_case_radius = float(np.linalg.norm(np.abs(all_verts).max(axis=0)))
    dist = worst_case_radius / np.tan(np.deg2rad(fov / 2)) * 1.5
    dist = max(dist, 2.0)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = AmbientLights(device=device)

    renderer_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=None, lights=lights),
    )
    renderer_mask = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=HardFlatShader(device=device, cameras=None, lights=lights),
    )

    #-- Output dirs --#
    case_dir    = os.path.join(output_base, case_name)
    seg_dir     = os.path.join(case_dir, "seg")
    images_dir  = os.path.join(case_dir, "images")
    gpt_dir     = os.path.join(case_dir, "gpt_input")
    for d in [seg_dir, images_dir, gpt_dir]:
        os.makedirs(d, exist_ok=True)

    elevations = [ 30, -30,  30, -30]
    azimuths   = [  0,   0, 180, 180]
    view_idx   = 1

    print("\nGenerating views and masks...")
    for elev, azim in zip(elevations, azimuths):
        R, T    = look_at_view_transform(dist=dist, elev=elev, azim=azim + 90, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        # Full Scene RGB render
        rendered_rgb = renderer_rgb(mesh_rgb, cameras=cameras)
        scene_img_float = rendered_rgb[0, ..., :3].cpu().numpy()
        scene_alpha = rendered_rgb[0, ..., 3].cpu().numpy() > 0
        scene_img = (scene_img_float * 255).astype(np.uint8)
        scene_img[~scene_alpha] = 0

        rendered_mask = renderer_mask(mesh_mask, cameras=cameras)
        mask_img      = rendered_mask[0, ..., :3].cpu().numpy()

        seg_map = -np.ones((image_size, image_size), dtype=np.int32)
        for info in part_info:
            r_c, g_c, b_c = info['color']
            diff      = np.abs(mask_img - np.array([r_c, g_c, b_c]))
            part_mask = np.all(diff < 1e-3, axis=-1) & scene_alpha
            seg_map[part_mask] = info['id']

        np.save(os.path.join(seg_dir, f"{view_idx:03d}_s.npy"), seg_map)
        imageio.imwrite(os.path.join(images_dir, f"{view_idx:03d}.png"), scene_img)

        scene_bbox = get_location(scene_alpha)
        if scene_bbox is not None:
            sx, sy, sw, sh = scene_bbox
            pad = max(20, int(image_size * 0.02))
            sx, sy = max(0, sx - pad), max(0, sy - pad)
            sw, sh = min(image_size - sx, sw + 2 * pad), min(image_size - sy, sh + 2 * pad)
            scene_crop = (sx, sy, sw, sh)
        else:
            scene_crop = None

        cur_gpt_dir = os.path.join(gpt_dir, f"{view_idx:02d}")
        os.makedirs(cur_gpt_dir, exist_ok=True)

        # X-Ray render for each part
        for info in part_info:
            p_id = info['id']
            p_mesh = part_meshes[p_id]
            
            p_rendered = renderer_rgb(p_mesh, cameras=cameras)
            p_img_float = p_rendered[0, ..., :3].cpu().numpy()
            p_alpha = p_rendered[0, ..., 3].cpu().numpy() > 0
            
            if not np.any(p_alpha):
                continue
                
            p_img = (p_img_float * 255).astype(np.uint8)

            dimmed_scene = (scene_img * 0.2).astype(np.uint8)
            x_ray_scene = dimmed_scene.copy()
            x_ray_scene[p_alpha] = p_img[p_alpha]

            blended_show = x_ray_scene

            # Cropped part image
            part_bbox = get_location(p_alpha.astype(np.uint8))
            if part_bbox is not None:
                px, py, pw, ph = part_bbox
                pad = max(20, int(image_size * 0.05))
                px, py = max(0, px - pad), max(0, py - pad)
                pw, ph = min(image_size - px, pw + 2 * pad), min(image_size - py, ph + 2 * pad)
                pad_seg_img = x_ray_scene[py:py+ph, px:px+pw] 
            else:
                pad_seg_img = x_ray_scene

            if scene_crop is not None:
                sx, sy, sw, sh = scene_crop
                img_show = scene_img[sy:sy+sh, sx:sx+sw]
                blended_show = blended_show[sy:sy+sh, sx:sx+sw]
            else:
                img_show = scene_img
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            ax1.imshow(img_show); ax1.set_title('Original Full Scene'); ax1.axis('off')
            ax2.imshow(blended_show); ax2.set_title('X-Ray Overlay'); ax2.axis('off')
            ax3.imshow(pad_seg_img); ax3.set_title('X-Ray Cropped Part'); ax3.axis('off')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(cur_gpt_dir, f"{p_id:02d}.png"), dpi=dpi)
            plt.close(fig)

        print(f"Processed View {view_idx}")
        view_idx += 1

    print(f"\nAll done! Saved to: {os.path.abspath(output_base)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./vlm_dataset")
    args = parser.parse_args()
    create_vlm_dataset_from_glb(args.glb_path, args.output_dir, case_name="scene")