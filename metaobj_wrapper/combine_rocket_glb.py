#!/usr/bin/env python
"""Combine five GLB parts into one transformed rocket GLB.

Default usage is intentionally hardcoded:

    python combine_rocket_glb.py

It reads:
    - inputs/glbs/mesh_0.glb
    - inputs/glbs/mesh_1.glb
    - inputs/glbs/mesh_2.glb
    - inputs/glbs/mesh_3.glb
    - inputs/glbs/mesh_4.glb
    - inputs/transforms.json

and writes:
    - outputs/rocket.glb
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from trimesh import transformations as tf


DEFAULT_PART_FILES = [
    "inputs/glbs/mesh_0.glb",
    "inputs/glbs/mesh_1.glb",
    "inputs/glbs/mesh_2.glb",
    "inputs/glbs/mesh_3.glb",
    "inputs/glbs/mesh_4.glb",
]
DEFAULT_TRANSFORM_JSON = "inputs/transforms.json"

ROOT_NODE_NAME = "rocket"
OBJECT_NAME_PREFIX = "control_parts"

DEFAULT_OUTPUT_FILE = f"outputs/{ROOT_NODE_NAME}_mesh.glb"
DEFAULT_METADATA_OUTPUT_FILE = f"outputs/{ROOT_NODE_NAME}_metadata.json"

# Current transforms are Blender-space object transforms. Blender's GLB exporter
# writes glTF Y-up coordinates, so Blender (x, y, z) becomes glTF (x, z, -y).
TRANSFORM_COORDINATE_SYSTEM = "blender"

PART_TRANSFORM_INDEXES: list[int | None] = [0, 1, 2, 3, 4]

# Default properties written for every part before the optional extra properties.
PART_METADATA_BASE_PROPERTIES: dict[str, Any] = {
    "colorHex": "#FFFFFF",
    "colorAlpha": 1.0,
    "metalic": 0.0,
    "smoothness": 0.5,
    "texture": "",
    "textureList": [],
}

# Add new common properties here. These are written after textureList in
# outputs/output.json. Delete a key from this dict if you no longer need it.
PART_METADATA_ADDITIONAL_PROPERTIES: dict[str, Any] = {
    "material": "aluminium",
    "affordance": "attach",
    "youngs_modulus_GPa": [68.0, 75.0],
    "density_g_cm3": [2.6, 2.85],
    "poissons_ratio": [0.32, 0.36],
    "tensile_strength_MPa": [70.0, 600.0],
    "hardness_HV": [20.0, 180.0],
    "thermal_conductivity_W_mK": [120.0, 235.0],
    "electrical_conductivity_MS_m": [18.0, 38.0],
    "thermal_expansion_coeff_1e-6_K": [21.0, 24.0],
    "fracture_toughness_MPa_sqrt_m": [15.0, 45.0]
}

# Per-part hardcoded values override the shared defaults. Add any key here when
# one part needs a different value from the rest.
PART_METADATA_OVERRIDES: dict[int, dict[str, Any]] = {
    0: {"colorHex": "#00FFFF"},
    1: {"colorHex": "#FF0000"},
    2: {"colorHex": "#12FF00"},
    3: {"colorHex": "#FFD100"},
    4: {"colorHex": "#FFFFFF"},
}


def _as_vec3(value: Any, field_name: str, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    if isinstance(value, (int, float)):
        return np.array([float(value), float(value), float(value)], dtype=float)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a number or a list of 3 numbers")
    return np.array(value, dtype=float)


def _scale_matrix(scale: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    matrix[2, 2] = scale[2]
    return matrix


def _matrix_from_flat(values: list[float], layout: str) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("matrix must contain 16 numbers")
    layout = layout.lower()
    if layout in {"row-major", "row_major", "row"}:
        return np.array(values, dtype=float).reshape((4, 4))
    if layout in {"column-major", "column_major", "column", "gltf"}:
        return np.array(values, dtype=float).reshape((4, 4), order="F")
    raise ValueError("matrix_layout must be row-major or column-major")


def _blender_to_gltf_matrix() -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return matrix


def _convert_matrix_space(matrix: np.ndarray, coordinate_system: str) -> np.ndarray:
    coordinate_system = coordinate_system.lower()
    if coordinate_system in {"gltf", "trimesh"}:
        return matrix
    if coordinate_system == "blender":
        blender_to_gltf = _blender_to_gltf_matrix()
        return blender_to_gltf @ matrix @ np.linalg.inv(blender_to_gltf)
    raise ValueError("coordinate_system must be blender or gltf")


def matrix_from_spec(spec: dict[str, Any], coordinate_system: str = TRANSFORM_COORDINATE_SYSTEM) -> np.ndarray:
    """Build a 4x4 transform matrix from one part's JSON spec."""
    if "matrix" in spec:
        matrix = spec["matrix"]
        if (
            isinstance(matrix, list)
            and len(matrix) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in matrix)
        ):
            return _convert_matrix_space(np.array(matrix, dtype=float), coordinate_system)
        if isinstance(matrix, list):
            return _convert_matrix_space(_matrix_from_flat(matrix, spec.get("matrix_layout", "row-major")), coordinate_system)
        raise ValueError("matrix must be a 4x4 list or a flat list of 16 numbers")

    translation = _as_vec3(
        spec.get("translation", spec.get("translate", spec.get("position"))),
        "translation",
        (0.0, 0.0, 0.0),
    )
    scale = _as_vec3(spec.get("scale"), "scale", (1.0, 1.0, 1.0))

    rotation = np.eye(4)
    if "rotation_quaternion" in spec or "rotation_quaternion_wxyz" in spec:
        quat = spec.get("rotation_quaternion", spec.get("rotation_quaternion_wxyz"))
        if not isinstance(quat, list) or len(quat) != 4:
            raise ValueError("rotation_quaternion must be [w, x, y, z]")
        rotation = tf.quaternion_matrix(quat)
    elif "rotation_quaternion_xyzw" in spec:
        quat = spec["rotation_quaternion_xyzw"]
        if not isinstance(quat, list) or len(quat) != 4:
            raise ValueError("rotation_quaternion_xyzw must be [x, y, z, w]")
        rotation = tf.quaternion_matrix([quat[3], quat[0], quat[1], quat[2]])
    else:
        euler = None
        units = "degrees"
        if "rotation_euler_radians" in spec:
            euler = spec["rotation_euler_radians"]
            units = "radians"
        elif "rotation_radians" in spec:
            euler = spec["rotation_radians"]
            units = "radians"
        elif "rotation_euler_degrees" in spec:
            euler = spec["rotation_euler_degrees"]
        elif "rotation_degrees" in spec:
            euler = spec["rotation_degrees"]
        elif "rotation" in spec:
            euler = spec["rotation"]
            units = spec.get("rotation_units", "degrees")

        if euler is not None:
            euler_vec = _as_vec3(euler, "rotation", (0.0, 0.0, 0.0))
            if units == "degrees":
                euler_vec = np.radians(euler_vec)
            elif units != "radians":
                raise ValueError("rotation_units must be degrees or radians")

            order = spec.get("rotation_order", "xyz").lower()
            if sorted(order) != ["x", "y", "z"]:
                raise ValueError("rotation_order must be a permutation of xyz")
            rotation = tf.euler_matrix(euler_vec[0], euler_vec[1], euler_vec[2], axes=f"s{order}")

    translate = tf.translation_matrix(translation)
    return _convert_matrix_space(translate @ rotation @ _scale_matrix(scale), coordinate_system)


def load_part_specs(transform_json: Path) -> list[dict[str, Any]]:
    with transform_json.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    if isinstance(raw, list):
        parts = raw
    elif isinstance(raw, dict) and isinstance(raw.get("parts"), list):
        parts = raw["parts"]
    else:
        raise ValueError("transform JSON must be a list or an object with a 'parts' list")

    if len(parts) != 5:
        raise ValueError(f"expected exactly 5 part transforms, got {len(parts)}")
    if not all(isinstance(part, dict) for part in parts):
        raise ValueError("every part transform must be an object")
    return parts


def specs_for_parts(part_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(PART_TRANSFORM_INDEXES) != 5:
        raise ValueError("PART_TRANSFORM_INDEXES must contain exactly 5 items")

    specs: list[dict[str, Any]] = []
    for index in PART_TRANSFORM_INDEXES:
        if index is None:
            specs.append({"name": "identity", "matrix": np.eye(4).tolist()})
            continue
        try:
            specs.append(part_specs[index])
        except IndexError as exc:
            raise ValueError(f"PART_TRANSFORM_INDEXES contains invalid index {index}") from exc
    return specs


def load_glb_scene(path: Path) -> trimesh.Scene:
    if not path.exists():
        raise FileNotFoundError(path)

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    raise TypeError(f"unsupported GLB content in {path}: {type(loaded).__name__}")


def transformed_nodes(scene: trimesh.Scene, matrix: np.ndarray) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    nodes: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    for node_data in scene.graph.to_flattened().values():
        geom_name = node_data.get("geometry")
        if geom_name is None:
            continue

        mesh = scene.geometry[geom_name]
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            continue

        local_transform = np.array(node_data["transform"], dtype=float)
        nodes.append((mesh.copy(), matrix @ local_transform))
    return nodes


def _matrix_to_trs(matrix: np.ndarray) -> dict[str, Any]:
    translation = matrix[:3, 3]
    basis = matrix[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    rotation_basis = basis.copy()
    for axis, axis_scale in enumerate(scale):
        if axis_scale > 0:
            rotation_basis[:, axis] /= axis_scale

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation_basis
    quat_wxyz = tf.quaternion_from_matrix(rotation_matrix)
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    def clean(values: Any) -> list[float]:
        cleaned = []
        for value in values:
            value = float(value)
            cleaned.append(0.0 if abs(value) < 1e-12 else value)
        return cleaned

    trs: dict[str, Any] = {}
    if not np.allclose(translation, 0.0, atol=1e-9):
        trs["translation"] = clean(translation)
    if not np.allclose(quat_xyzw, [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        trs["rotation"] = clean(quat_xyzw)
    if not np.allclose(scale, 1.0, atol=1e-9):
        trs["scale"] = clean(scale)
    return trs


def _read_glb_chunks(path: Path) -> tuple[dict[str, Any], list[tuple[int, bytes]]]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 GLB file")

    json_doc: dict[str, Any] | None = None
    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_doc = json.loads(chunk_data.decode("utf-8"))
        else:
            chunks.append((chunk_type, chunk_data))

    if json_doc is None:
        raise ValueError(f"{path} does not contain a JSON chunk")
    return json_doc, chunks


def _write_glb_chunks(path: Path, json_doc: dict[str, Any], chunks: list[tuple[int, bytes]]) -> None:
    json_bytes = json.dumps(json_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_chunk = json_bytes + (b" " * json_padding)

    output_chunks = [(0x4E4F534A, json_chunk)]
    for chunk_type, chunk_data in chunks:
        padding = (4 - (len(chunk_data) % 4)) % 4
        output_chunks.append((chunk_type, chunk_data + (b"\x00" * padding)))
    total_length = 12 + sum(8 + len(chunk_data) for _, chunk_data in output_chunks)

    with path.open("wb") as fp:
        fp.write(struct.pack("<4sII", b"glTF", 2, total_length))
        for chunk_type, chunk_data in output_chunks:
            fp.write(struct.pack("<II", len(chunk_data), chunk_type))
            fp.write(chunk_data)


def rewrite_glb_hierarchy_like_test(path: Path, root_name: str, object_names: list[str]) -> None:
    json_doc, chunks = _read_glb_chunks(path)
    scenes = json_doc.get("scenes")
    nodes = json_doc.get("nodes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("GLB must contain at least one scene")
    if not isinstance(nodes, list):
        raise ValueError("GLB must contain nodes")

    scene_index = int(json_doc.get("scene", 0))
    scene = scenes[scene_index]
    scene_root_indexes = scene.get("nodes", [])
    if not isinstance(scene_root_indexes, list) or not scene_root_indexes:
        raise ValueError("Scene must contain at least one node")

    child_indexes: list[int] = []
    root_transform: dict[str, Any] = {}

    if len(scene_root_indexes) == 1:
        exported_root = nodes[scene_root_indexes[0]]
        children = exported_root.get("children")
        if isinstance(children, list) and children:
            child_indexes = children
            if "matrix" in exported_root:
                matrix = np.array(exported_root["matrix"], dtype=float).reshape((4, 4), order="F")
                root_transform = _matrix_to_trs(matrix)
            else:
                for key in ("translation", "rotation", "scale"):
                    if key in exported_root:
                        root_transform[key] = exported_root[key]

    if not child_indexes:
        child_indexes = scene_root_indexes

    if len(child_indexes) != len(object_names):
        raise ValueError(f"Expected {len(object_names)} child nodes, got {len(child_indexes)}")

    rewritten_nodes: list[dict[str, Any]] = []
    for output_index, (old_node_index, object_name) in enumerate(zip(child_indexes, object_names)):
        old_node = nodes[old_node_index]
        new_node: dict[str, Any] = {"name": object_name}

        if "mesh" in old_node:
            new_node["mesh"] = old_node["mesh"]
        if "matrix" in old_node:
            matrix = np.array(old_node["matrix"], dtype=float).reshape((4, 4), order="F")
            new_node.update(_matrix_to_trs(matrix))
        else:
            for key in ("translation", "rotation", "scale"):
                if key in old_node:
                    new_node[key] = old_node[key]

        rewritten_nodes.append(new_node)

        meshes = json_doc.get("meshes")
        if isinstance(meshes, list) and output_index < len(meshes):
            meshes[output_index]["name"] = object_name

    root_index = len(rewritten_nodes)
    root_node: dict[str, Any] = {"name": root_name, "children": list(range(root_index))}
    root_node.update(root_transform)
    rewritten_nodes.append(root_node)

    json_doc["nodes"] = rewritten_nodes
    json_doc["scene"] = scene_index
    scene["name"] = root_name
    scene["nodes"] = [root_index]

    _write_glb_chunks(path, json_doc, chunks)


def _component_dtype(component_type: int) -> np.dtype:
    dtype_by_component = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    try:
        return np.dtype(dtype_by_component[component_type])
    except KeyError as exc:
        raise ValueError(f"unsupported component type: {component_type}") from exc


def _type_component_count(accessor_type: str) -> int:
    count_by_type = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    try:
        return count_by_type[accessor_type]
    except KeyError as exc:
        raise ValueError(f"unsupported accessor type: {accessor_type}") from exc


def _read_accessor(json_doc: dict[str, Any], bin_data: bytes, accessor_index: int) -> np.ndarray:
    accessor = json_doc["accessors"][accessor_index]
    buffer_view = json_doc["bufferViews"][accessor["bufferView"]]
    dtype = _component_dtype(accessor["componentType"])
    component_count = _type_component_count(accessor["type"])
    count = accessor["count"]

    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    byte_stride = buffer_view.get("byteStride")
    row_nbytes = dtype.itemsize * component_count

    if byte_stride is None or byte_stride == row_nbytes:
        data = np.frombuffer(bin_data, dtype=dtype, count=count * component_count, offset=byte_offset)
        return data.reshape((count, component_count)).copy()

    rows = []
    for row_index in range(count):
        row_offset = byte_offset + (row_index * byte_stride)
        row = np.frombuffer(bin_data, dtype=dtype, count=component_count, offset=row_offset)
        rows.append(row.copy())
    return np.vstack(rows)


def _append_accessor(
    json_doc: dict[str, Any],
    bin_data: bytearray,
    values: np.ndarray,
    accessor_type: str,
    target: int,
) -> int:
    values = np.asarray(values, dtype=np.float32)
    padding = (4 - (len(bin_data) % 4)) % 4
    if padding:
        bin_data.extend(b"\x00" * padding)

    byte_offset = len(bin_data)
    payload = values.tobytes(order="C")
    bin_data.extend(payload)

    buffer_view_index = len(json_doc["bufferViews"])
    json_doc["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(payload),
            "target": target,
        }
    )

    accessor_index = len(json_doc["accessors"])
    json_doc["accessors"].append(
        {
            "bufferView": buffer_view_index,
            "componentType": 5126,
            "count": int(values.shape[0]),
            "type": accessor_type,
        }
    )
    return accessor_index


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces = indices.reshape((-1, 3)).astype(np.int64)
    normals = np.zeros_like(positions, dtype=np.float64)
    triangles = positions[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])

    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0])
    return normals.astype(np.float32)


def _compute_tangents(positions: np.ndarray, uvs: np.ndarray, normals: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces = indices.reshape((-1, 3)).astype(np.int64)
    tangent_accum = np.zeros_like(positions, dtype=np.float64)
    bitangent_accum = np.zeros_like(positions, dtype=np.float64)

    for face in faces:
        p0, p1, p2 = positions[face]
        uv0, uv1, uv2 = uvs[face]
        edge1 = p1 - p0
        edge2 = p2 - p0
        delta_uv1 = uv1 - uv0
        delta_uv2 = uv2 - uv0
        determinant = (delta_uv1[0] * delta_uv2[1]) - (delta_uv2[0] * delta_uv1[1])

        if abs(float(determinant)) < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            bitangent = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            reciprocal = 1.0 / determinant
            tangent = (edge1 * delta_uv2[1] - edge2 * delta_uv1[1]) * reciprocal
            bitangent = (edge2 * delta_uv1[0] - edge1 * delta_uv2[0]) * reciprocal

        for vertex_index in face:
            tangent_accum[vertex_index] += tangent
            bitangent_accum[vertex_index] += bitangent

    tangents = np.zeros((len(positions), 4), dtype=np.float32)
    for vertex_index, normal in enumerate(normals.astype(np.float64)):
        tangent = tangent_accum[vertex_index]
        tangent = tangent - normal * np.dot(normal, tangent)
        tangent_length = np.linalg.norm(tangent)
        if tangent_length <= 1e-12:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent = tangent / tangent_length

        handedness = 1.0
        if np.dot(np.cross(normal, tangent), bitangent_accum[vertex_index]) < 0.0:
            handedness = -1.0

        tangents[vertex_index, :3] = tangent.astype(np.float32)
        tangents[vertex_index, 3] = handedness
    return tangents


def add_test_like_mesh_attributes(path: Path) -> None:
    json_doc, chunks = _read_glb_chunks(path)
    bin_chunk_index = next((index for index, (chunk_type, _) in enumerate(chunks) if chunk_type == 0x004E4942), None)
    if bin_chunk_index is None:
        raise ValueError(f"{path} does not contain a BIN chunk")

    bin_data = bytearray(chunks[bin_chunk_index][1])
    meshes = json_doc.get("meshes", [])
    materials = json_doc.get("materials", [])

    for material_index, material in enumerate(materials):
        material.setdefault("name", f"test{material_index}")
        material.pop("doubleSided", None)
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 0.5)

    for mesh_index, mesh in enumerate(meshes):
        mesh.pop("extras", None)
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode") == 4:
                primitive.pop("mode", None)

            attributes = primitive.setdefault("attributes", {})
            if "POSITION" not in attributes or "indices" not in primitive:
                continue

            positions = _read_accessor(json_doc, bytes(bin_data), attributes["POSITION"]).astype(np.float32)
            indices = _read_accessor(json_doc, bytes(bin_data), primitive["indices"]).reshape((-1,)).astype(np.uint32)

            if "NORMAL" in attributes:
                normals = _read_accessor(json_doc, bytes(bin_data), attributes["NORMAL"]).astype(np.float32)
            else:
                normals = _compute_vertex_normals(positions, indices)
                attributes["NORMAL"] = _append_accessor(json_doc, bin_data, normals, "VEC3", 34962)

            if "TEXCOORD_0" in attributes:
                texcoord_0 = _read_accessor(json_doc, bytes(bin_data), attributes["TEXCOORD_0"]).astype(np.float32)
            else:
                texcoord_0 = np.zeros((len(positions), 2), dtype=np.float32)
                attributes["TEXCOORD_0"] = _append_accessor(json_doc, bin_data, texcoord_0, "VEC2", 34962)

            if "TANGENT" not in attributes:
                tangents = _compute_tangents(positions, texcoord_0, normals, indices)
                attributes["TANGENT"] = _append_accessor(json_doc, bin_data, tangents, "VEC4", 34962)

            if "TEXCOORD_1" not in attributes:
                attributes["TEXCOORD_1"] = _append_accessor(json_doc, bin_data, texcoord_0, "VEC2", 34962)

            preferred_order = ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1")
            primitive["attributes"] = {
                **{key: attributes[key] for key in preferred_order if key in attributes},
                **{key: value for key, value in attributes.items() if key not in preferred_order},
            }

        if not mesh.get("name"):
            mesh["name"] = f"control_parts{mesh_index}"

    json_doc["buffers"][0]["byteLength"] = len(bin_data)
    chunks[bin_chunk_index] = (0x004E4942, bytes(bin_data))
    _write_glb_chunks(path, json_doc, chunks)


def combine_parts(part_files: list[Path], part_specs: list[dict[str, Any]], output_file: Path) -> None:
    if len(part_files) != 5:
        raise ValueError(f"expected exactly 5 GLB files, got {len(part_files)}")

    combined = trimesh.Scene()
    for index, (part_file, spec) in enumerate(zip(part_files, part_specs), start=1):
        part_name = spec.get("name") or part_file.stem or f"part_{index:02d}"
        scene = load_glb_scene(part_file)
        matrix = matrix_from_spec(spec)
        nodes = transformed_nodes(scene, matrix)
        if not nodes:
            raise ValueError(f"{part_file} did not contain any mesh geometry")

        for mesh_index, (mesh, transform) in enumerate(nodes):
            mesh_name = f"{part_name}_{mesh_index:02d}"
            combined.add_geometry(mesh, geom_name=mesh_name, node_name=mesh_name, transform=transform)

        print(f"[{index}/5] added {part_file} as {part_name} ({len(nodes)} mesh item(s))")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_file)
    object_names = [f"{OBJECT_NAME_PREFIX}{index}" for index in range(len(part_files))]
    rewrite_glb_hierarchy_like_test(output_file, ROOT_NODE_NAME, object_names)
    add_test_like_mesh_attributes(output_file)
    print(f"wrote {output_file}")


def build_part_metadata(part_index: int, object_name: str, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    part: dict[str, Any] = {"objectName": object_name}
    part.update(PART_METADATA_BASE_PROPERTIES)

    # Scalar additional properties stay flat; [min, max] properties are collected
    # into a single rangeProperties list of {label, min, max} objects.
    range_properties: list[dict[str, Any]] = []
    for key, value in PART_METADATA_ADDITIONAL_PROPERTIES.items():
        if isinstance(value, list) and len(value) == 2:
            range_properties.append({"label": key, "min": value[0], "max": value[1]})
        else:
            part[key] = value
    part["rangeProperties"] = range_properties

    part.update(PART_METADATA_OVERRIDES.get(part_index, {}))

    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise ValueError("part metadata must be an object")
        part.update(extra_metadata)

    return part


def write_metadata_json(output_file: Path, part_specs: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    object_names = [f"{OBJECT_NAME_PREFIX}{index}" for index in range(len(part_specs))]
    metadata = {
        "version": 1,
        "parts": [
            build_part_metadata(part_index, object_name, spec.get("metadata"))
            for part_index, (object_name, spec) in enumerate(zip(object_names, part_specs))
        ],
    }
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=4)
        fp.write("\n")
    print(f"wrote {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform and combine five GLB files into one GLB.")
    parser.add_argument(
        "--parts",
        nargs=5,
        default=DEFAULT_PART_FILES,
        metavar=("PART_01", "PART_02", "PART_03", "PART_04", "PART_05"),
        help="five input GLB paths, in order",
    )
    parser.add_argument(
        "--transforms",
        default=DEFAULT_TRANSFORM_JSON,
        help="transform JSON path",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="output GLB path",
    )
    parser.add_argument(
        "--metadata-output",
        default=DEFAULT_METADATA_OUTPUT_FILE,
        help="output JSON metadata path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_files = [Path(path) for path in args.parts]
    part_specs = specs_for_parts(load_part_specs(Path(args.transforms)))
    combine_parts(part_files, part_specs, Path(args.output))
    write_metadata_json(Path(args.metadata_output), part_specs)


if __name__ == "__main__":
    main()
