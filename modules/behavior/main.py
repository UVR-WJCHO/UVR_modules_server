import os
import argparse


def _load_dotenv(path):
    """Load KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


# Load the repo-root .env BEFORE importing query_vlm, because vlm_utils reads
# OPENAI_API_KEY at import time. Lets standalone runs pick up the key.
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from utils.preprocess_glb_for_vlm import create_vlm_dataset_from_glb
from utils.query_vlm import query_vlm
from utils.visualize import visualize_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path", type=str, default="./data/SaturnV/SaturnV.glb", help="Path to the input .glb file")
    parser.add_argument("--vlm_input_dir", type=str, default="./vlm_input", help="Directory to save the vlm inputs")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save the final JSON result (default: vlm_input_dir/<case_name>/vlm_result.json)")
    parser.add_argument("--vlm_type", type=str, default="gpt4", help="gpt4 | gemini | gemini_flash | qwen")
    parser.add_argument("--dpi", type=int, default=75, help="DPI for the VLM input images (lower = fewer tokens)")

    args = parser.parse_args()

    case_name = os.path.splitext(os.path.basename(args.glb_path))[0]

    # 1. Preprocess vlm inputs directly from GLB
    if os.path.exists(args.vlm_input_dir):
        ans = input(f"{args.vlm_input_dir} already exists. Re-run VLM preprocessing? [y/N] ").strip().lower()
        run_preprocess = ans == 'y'
    else:
        run_preprocess = True
        
    if run_preprocess:
        create_vlm_dataset_from_glb(args.glb_path, output_base=args.vlm_input_dir, case_name=case_name, dpi=args.dpi)

    # 2. Query VLM
    result_path = os.path.join(args.vlm_input_dir, case_name, "vlm_result.txt")
    if os.path.exists(result_path):
        ans = input("vlm_result.txt already exists. Re-run VLM query? [y/N] ").strip().lower()
        run_vlm = ans == 'y'
    else:
        run_vlm = True
    if run_vlm:
        query_vlm(base_path=args.vlm_input_dir, case_name=case_name, vlm_type=args.vlm_type, output_json=args.output_json)

    # 3. Visualize
    if os.path.exists(args.vlm_input_dir):
        case_path = os.path.join(args.vlm_input_dir, case_name)
        if os.path.isdir(case_path) and os.path.exists(os.path.join(case_path, 'vlm_result.json')):
            visualize_results(args.vlm_input_dir, case_name)
