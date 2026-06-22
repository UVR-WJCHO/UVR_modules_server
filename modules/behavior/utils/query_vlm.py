import os
import json
import time
from collections import Counter
from openai import OpenAI
from utils.vlm_utils import GPT4V, encode_image
from utils.config import AFFORDANCES, MATERIAL_PROPERTY_PRIORS


def _collect_images(input_image_path):
    items = []
    for view_folder in sorted(os.listdir(input_image_path)):
        for image_file in sorted(os.listdir(os.path.join(input_image_path, view_folder))):
            part_id_str = os.path.splitext(image_file)[0]
            if not part_id_str.isdigit():
                continue
            part_id = int(part_id_str)
            image_path = os.path.join(input_image_path, view_folder, image_file)
            items.append((view_folder, part_id, image_path))
    return items


def _parse_message(message):
    for line in message.strip().splitlines():
        line = line.strip()
        if line.startswith("thinking:"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return parts[0], parts[1].lower(), parts[2]
    return "unknown", "unknown", "unknown"


def _save_results(votes, results_file_path, base_path, case_name, output_json):
    json_data = {"version": 1, "parts": []}
    for part_id, v in sorted(votes.items()):
        material = Counter(v["materials"]).most_common(1)[0][0]
        affordance = Counter(v["affordances"]).most_common(1)[0][0]
        caption = Counter(v["captions"]).most_common(1)[0][0]

        priors = MATERIAL_PROPERTY_PRIORS.get(material, {})

        part_json = {
            "objectName": caption,
            "colorHex": "#FFFFFF",
            "colorAlpha": 1.0,
            "smoothness": 0.5,
            "texture": "",
            "weight": "",
            "temperature": [],
            "textureList": [],

            "partId": part_id,
            "caption": caption,
            "material": material,
            "affordance": affordance,
        }
        for key, val in priors.items():
            part_json[key] = val
        json_data["parts"].append(part_json)

    case_json_path = os.path.join(base_path, case_name, "vlm_result.json")
    with open(case_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=4)

    print("Messages have been written to", results_file_path)
    print("JSON results have been saved to", case_json_path)


def _query_vlm_sync(base_path, case_name, vlm_type, system_prompt, prompt, results_file_path, output_json):
    input_image_path = os.path.join(base_path, case_name, "gpt_input")
    items = _collect_images(input_image_path)
    votes = {}

    for view_folder, part_id, image_path in items:
        try:
            if vlm_type == 'qwen':
                message = str(Qwen(image_path, prompt, system_prompt))
            elif vlm_type == 'gpt4':
                message = str(GPT4V(image_path, prompt, system_prompt))
            elif vlm_type == 'gemini':
                message = str(Gemini(image_path, prompt, system_prompt))
            elif vlm_type == 'gemini_flash':
                message = str(GeminiFlash(image_path, prompt, system_prompt))
            else:
                raise NotImplementedError(f"Unknown vlm_type: {vlm_type}")
        except Exception as e:
            print(f"Exception: {e} for image {image_path}")
            raise e

        with open(results_file_path, 'a') as f:
            f.write(f"{image_path},{message}\n")

        caption, material, affordance = _parse_message(message)
        if part_id not in votes:
            votes[part_id] = {"captions": [], "materials": [], "affordances": []}
        votes[part_id]["captions"].append(caption)
        votes[part_id]["materials"].append(material)
        votes[part_id]["affordances"].append(affordance)

    _save_results(votes, results_file_path, base_path, case_name, output_json)


def query_vlm(base_path, case_name, vlm_type="gpt4", output_json=None):
    material_library = "{" + ", ".join(MATERIAL_PROPERTY_PRIORS.keys()) + "}"
    affordance_library = "{" + ", ".join(AFFORDANCES) + "}"

    system_prompt = f"""
        You are a 3D object part analyst specializing in material and affordance estimation.

        ### Material Library
        {material_library}

        ### Material Selection Rules
        - Material must be selected exactly from the Material Library.
        - Do not invent new material names.
        - Do not output material names outside the Material Library.
        - Choose the most plausible material for that part type.

        ### Affordance List
        {affordance_library}

        ### Affordance Selection Rules
        - Affordance must be selected exactly from the Affordance List.
        - Choose the affordance of the bright/undimmed highlighted target part itself.
        - Interpret affordance as the part's role during assembly, not as the whole object's final use.
        - Select the affordance that best describes how this part would be used, positioned, connected, supported, contained, or handled when assembling the object.
        - Do not choose an affordance based only on what the complete object does.

        ### Reasoning Policy
        - Analyze the object, target part, visual characteristics, material, and affordance internally.
        - Do not output reasoning, thinking, chain-of-thought, analysis, or explanations.
        - Output only the final answer.

        ### Strict Output Rules
        - Output exactly one line. No more, no less.
        - Format: caption, material, affordance
        - No markdown, no extra explanation.
        - Do not include labels such as "caption:", "material:", or "affordance:".
        - Do not include "thinking:".
        - Use exactly one material from the Material Library.
        - Use exactly one affordance from the Affordance List.
        - Do NOT use semicolons or "||".
        """

    prompt = """
        ### Task
        You are analyzing a 3D object rendered as separate parts for part-level material and affordance estimation.

        The target part is isolated and highlighted in the X-Ray Overlay and Part Image.
        Infer the most plausible material and affordance for that part only, not for the entire object.

        Analyze ONLY the highlighted target part and provide:
        1. Caption: concise name/description of the target part.
        2. Material: choose the most plausible material from the provided Material Library.
        3. Affordance: choose the single most relevant interaction from the Affordance List.

        ### Input Data
        1. Original Image: Full view of the object for overall context only.
        2. X-Ray Overlay: The bright/undimmed region is the target part. The dark regions are surrounding context and should not be classified as the target.
        3. Part Image: Cropped close-up centered on the target part. Some surrounding geometry may still be visible, but it is context only unless it belongs to the highlighted target part.

        ### Internal Analysis Instructions
        Before answering, internally consider:
        - what the overall object is
        - where the bright/undimmed highlighted part is located on the object
        - what the bright/undimmed highlighted part is, based on its shape and location
        - the bright/undimmed highlighted part's structural function, without inferring hidden internal components
        - key visual characteristics of the bright/undimmed highlighted part
        - why the chosen material and affordance are most plausible for the bright/undimmed highlighted part

        Do not output this analysis.

        ### Output Format
        caption, material, affordance
        """
    
    os.makedirs(os.path.join(base_path, case_name), exist_ok=True)
    results_file_path = os.path.join(base_path, case_name, 'vlm_result.txt')
    if os.path.exists(results_file_path):
        os.remove(results_file_path)

    _query_vlm_sync(base_path, case_name, vlm_type, system_prompt, prompt, results_file_path, output_json)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Query VLM for material property prediction")
    parser.add_argument("--base_path", type=str, default="./vlm_input")
    parser.add_argument("--vlm_type", type=str, default="gpt4", help="gpt4 | gemini | gemini_flash | qwen")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    for case_name in os.listdir(args.base_path):
        query_vlm(args.base_path, case_name, vlm_type=args.vlm_type, output_json=args.output_json)
