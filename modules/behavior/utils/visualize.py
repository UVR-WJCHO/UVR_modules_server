import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.config import NUMERIC_PROPS

STRING_PROPS = ["material", "affordance"]

def visualize_results(base_path, case_name):
    json_path = os.path.join(base_path, case_name, 'vlm_result.json')
    if not os.path.exists(json_path):
        print(f"Result file not found: {json_path}")
        return

    print(f"Visualizing for case: {case_name}")

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    seg_dir = os.path.join(base_path, case_name, "seg")
    if not os.path.exists(seg_dir):
        return
    seg_files = sorted(f for f in os.listdir(seg_dir) if f.endswith('.npy'))

    for prop_key in NUMERIC_PROPS + STRING_PROPS:
        is_string = prop_key in STRING_PROPS

        part_values = {}
        for part in json_data["parts"]:
            val = part.get(prop_key)
            if val is None:
                continue
            if is_string:
                part_values[part["partId"]] = val
            else:
                r = val["range"] if isinstance(val, dict) else val
                part_values[part["partId"]] = (r[0] + r[1]) / 2

        if not part_values:
            continue

        all_labels = []
        label_to_idx = {}
        if is_string:
            all_labels = sorted(set(part_values.values()))
            label_to_idx = {label: i for i, label in enumerate(all_labels)}

        vis_output_dir = os.path.join(base_path, case_name, "property_visualizations", prop_key)
        os.makedirs(vis_output_dir, exist_ok=True)

        for seg_file in seg_files:
            mask = np.load(os.path.join(seg_dir, seg_file)).astype(np.float32)

            for part_id, val in part_values.items():
                mask[mask == part_id] = label_to_idx[val] if is_string else val

            masked = np.ma.masked_where(mask == -1, mask)
            vmin, vmax = np.min(masked), np.max(masked)

            cmap = plt.colormaps['tab20'].resampled(len(all_labels)) if is_string else 'viridis'
            plt.figure(figsize=(6, 6))
            im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            if is_string:
                n = len(all_labels)
                step = (vmax - vmin) / n if n > 1 else 1
                cbar.set_ticks([vmin + (i + 0.5) * step for i in range(n)])
                cbar.set_ticklabels(all_labels)
            cbar.set_label(prop_key, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=8)

            view_name = os.path.splitext(seg_file)[0]
            plt.savefig(os.path.join(vis_output_dir, f"vis_{view_name}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()

        print(f"[{case_name}] Saved {prop_key} visualizations to {vis_output_dir}/")

    visualize_object_names(base_path, case_name)


def visualize_object_names(base_path, case_name):
    json_path = os.path.join(base_path, case_name, 'vlm_result.json')
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    part_names = {p["partId"]: p.get("caption", "") for p in json_data["parts"]}
    if not any(part_names.values()):
        return

    part_ids = sorted(part_names.keys())
    cmap = plt.colormaps['tab20'].resampled(max(len(part_ids), 2))
    part_colors = {pid: cmap(i / max(len(part_ids) - 1, 1)) for i, pid in enumerate(part_ids)}

    seg_dir = os.path.join(base_path, case_name, "seg")
    if not os.path.exists(seg_dir):
        return
    seg_files = sorted(f for f in os.listdir(seg_dir) if f.endswith('.npy'))

    vis_output_dir = os.path.join(base_path, case_name, "property_visualizations", "caption")
    os.makedirs(vis_output_dir, exist_ok=True)

    for seg_file in seg_files:
        seg = np.load(os.path.join(seg_dir, seg_file))
        h, w = seg.shape
        img = np.zeros((h, w, 4), dtype=np.float32)

        for pid in part_ids:
            mask = seg == pid
            if not np.any(mask):
                continue
            img[mask] = part_colors[pid]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')

        for pid in part_ids:
            mask = seg == pid
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            name = part_names[pid]
            ax.text(cx, cy, name, fontsize=6, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, linewidth=0))

        view_name = os.path.splitext(seg_file)[0]
        plt.savefig(os.path.join(vis_output_dir, f"vis_{view_name}.png"), bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()

    print(f"[{case_name}] Saved caption visualizations to {vis_output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VLM property results from JSON file")
    parser.add_argument("--base_path", type=str, default="./vlm_input", help="Base path containing the preprocessed VLM datasets")
    args = parser.parse_args()

    if os.path.exists(args.base_path):
        for case_name in os.listdir(args.base_path):
            case_path = os.path.join(args.base_path, case_name)
            if os.path.isdir(case_path) and os.path.exists(os.path.join(case_path, 'vlm_result.json')):
                visualize_results(args.base_path, case_name)
