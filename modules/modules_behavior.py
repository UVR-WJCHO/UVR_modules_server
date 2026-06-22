"""Wrapper for the behavior property-estimation pipeline (GLB -> property JSON).

Runs the front half of modules/behavior/main.py:
  1. create_vlm_dataset_from_glb : render the GLB into per-part VLM input images
  2. query_vlm                   : query the VLM and write vlm_result.json
The final visualization step of behavior/main.py is intentionally skipped.

Requires OPENAI_API_KEY in the environment (for vlm_type="gpt4") and the
behavior deps (pytorch3d, trimesh, imageio, matplotlib, openai).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# behavior/ ships its own `utils` package; put it on sys.path so the
# pipeline's `from utils...` imports resolve to modules/behavior/utils.
_BEHAVIOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavior")
if _BEHAVIOR_ROOT not in sys.path:
    sys.path.insert(0, _BEHAVIOR_ROOT)


def _load_api_keys(env_file):
    """Load KEY=VALUE lines from a local secrets file into os.environ.

    The file (default repo-root .env) is git-ignored, so API keys never get
    committed. Existing environment variables take precedence.
    """
    if not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


class BehaviorPropertyEstimator:
    """Estimate per-part material / affordance / physical properties from a GLB."""

    def __init__(self, vlm_input_dir="output/vlm_input", vlm_type="gpt4", dpi=75,
                 env_file=os.path.join(_REPO_ROOT, ".env")):
        self.vlm_input_dir = vlm_input_dir
        self.vlm_type = vlm_type
        self.dpi = dpi
        self.env_file = env_file

    def run(self, glb_path, output_json=None, vlm_input_dir=None):
        """GLB -> property JSON. Returns the path to the written vlm_result.json.

        Performs steps 1 (render) and 2 (VLM query) only; visualization is skipped.
        vlm_input_dir overrides the default output base for this call.
        """
        # Load API keys into os.environ BEFORE importing query_vlm, because
        # behavior/utils/vlm_utils.py reads OPENAI_API_KEY at import time.
        _load_api_keys(self.env_file)

        # Imported lazily so merely importing this module does not require the
        # heavy behavior deps (pytorch3d, openai, ...) to be installed.
        from utils.preprocess_glb_for_vlm import create_vlm_dataset_from_glb
        from utils.query_vlm import query_vlm

        base_dir = vlm_input_dir or self.vlm_input_dir
        case_name = os.path.splitext(os.path.basename(glb_path))[0]

        # 1. GLB -> per-part VLM input images
        create_vlm_dataset_from_glb(
            glb_path, output_base=base_dir, case_name=case_name, dpi=self.dpi
        )

        # 2. Query VLM -> vlm_result.json (property info)
        query_vlm(
            base_path=base_dir,
            case_name=case_name,
            vlm_type=self.vlm_type,
            output_json=output_json,
        )

        return output_json or os.path.join(base_dir, case_name, "vlm_result.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GLB -> property JSON via the behavior VLM pipeline (no visualization)"
    )
    parser.add_argument("--glb_path", type=str, required=True,
                        help="Path to the input .glb (e.g. output/mesh_0.glb)")
    parser.add_argument("--vlm_input_dir", type=str, default="output/vlm_input")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional extra path to also save the result JSON")
    parser.add_argument("--vlm_type", type=str, default="gpt4",
                        help="gpt4 | gemini | gemini_flash | qwen")
    parser.add_argument("--dpi", type=int, default=75)
    args = parser.parse_args()

    estimator = BehaviorPropertyEstimator(
        vlm_input_dir=args.vlm_input_dir, vlm_type=args.vlm_type, dpi=args.dpi
    )
    json_path = estimator.run(args.glb_path, output_json=args.output_json)
    print(f"[Behavior] property JSON saved to {json_path}")
