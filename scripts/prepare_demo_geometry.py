#!/usr/bin/env python3
"""Build licensed demo geometry without changing model or training artifacts.

Input downloads and their pinned upstream revisions are recorded in
demo/assets-source/download-receipts.json. Coordinates are measurements, never
fabricated for missing somata. The fly is an illustrative female body specimen.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demo/assets-source"
OUT = ROOT / "demo/public/assets"
BODY = SOURCE / "Xenova/fruit-fly-simulation/public/body/assets"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")


def rotation_quaternion(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rotation_axis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)*np.cos(angle) + (1-np.cos(angle))*np.outer(axis, axis) + np.sin(angle)*skew


def transform(rotation, translation=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def neutral_transforms(model):
    rest = model["rest"]
    root = rest[model["root"]]
    transforms = {model["root"]: transform(rotation_quaternion(root["quat"]), root["pos"])}
    for parent, child in model["joints"]:
        r = np.eye(3)
        for dof in model["dofs"]:
            if dof["parent"] == parent and dof["child"] == child:
                axis = dof["axis"] if isinstance(dof["axis"], list) else model["axisVector"][dof["axis"]]
                r = r @ rotation_axis(axis, np.deg2rad(model["neutralDeg"].get(dof["name"], 0)))
        transforms[child] = transforms[parent] @ transform(rotation_quaternion(rest[child]["quat"]), rest[child]["pos"]) @ transform(r)
    error = max(float(np.max(np.abs(transforms[name][:3, 3] - position)))
                for name, position in model["reference"]["neutralJointPositions"].items())
    if error > 1e-8:
        raise ValueError(f"Neutral pose differs from published joint positions: {error}")
    return transforms, error


def build_fly():
    model = json.loads((BODY / "model.json").read_text())
    transforms, error = neutral_transforms(model)
    gltf = {"asset": {"version": "2.0", "generator": "fly-wordbrain prepare_demo_geometry.py",
                       "copyright": "NeuroMechFly v2 / NeLy-EPFL; Apache-2.0; see assets/NOTICE.txt"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
            "buffers": [], "bufferViews": [], "accessors": []}
    binary = bytearray()
    bounds, head_bounds = [], None
    # Proper rotation from NeuroMechFly x-forward/y-left/z-up into glTF Y-up.
    to_gltf = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    def accessor(values, component, kind, target):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        raw = values.tobytes()
        binary.extend(raw)
        view = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        desc = {"bufferView": view, "componentType": component, "count": len(values), "type": kind}
        if kind == "VEC3":
            desc.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        gltf["accessors"].append(desc)
        return len(gltf["accessors"])-1

    triangles = 0
    for name, mesh in model["meshes"].items():
        data = (BODY / "meshes" / mesh["file"]).read_bytes()
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) != 84 + 50*count:
            raise ValueError("Invalid binary STL: " + mesh["file"])
        dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, offset=84, count=count)["vertices"].astype(float)
        vertices *= model["meshScale"]
        if mesh["mirror"]:
            vertices[:, :, 1] *= -1
            vertices = vertices[:, [0, 2, 1], :]
        matrix = transforms[name]
        vertices = (vertices.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]) @ to_gltf.T
        # Weld shared triangle vertices and average normals for a smooth mesh.
        _, first, index = np.unique(np.round(vertices, 5), axis=0, return_index=True, return_inverse=True)
        points = vertices[first].astype("<f4")
        index = index.astype("<u4")
        faces = points[index.reshape(-1, 3)]
        normals = np.zeros_like(points)
        face_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        for vertex in range(3):
            np.add.at(normals, index.reshape(-1, 3)[:, vertex], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        color = model["colors"][name]
        if len(color) == 3:
            color = color + [1]
        material = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": color, "metallicFactor": 0.15,
                     "roughnessFactor": 0.42}, "doubleSided": "wing" in name}
        if color[3] < 1:
            material["alphaMode"] = "BLEND"
        gltf["materials"].append(material)
        gltf["meshes"].append({"name": name, "primitives": [{"attributes": {
            "POSITION": accessor(points, 5126, "VEC3", 34962),
            "NORMAL": accessor(normals.astype("<f4"), 5126, "VEC3", 34962)},
            "indices": accessor(index, 5125, "SCALAR", 34963), "material": len(gltf["materials"])-1}]})
        gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"])-1})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"])-1)
        bbox = [points.min(0).tolist(), points.max(0).tolist()]
        bounds.extend(bbox)
        if name == "c_head":
            head_bounds = bbox
        triangles += count
    gltf["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(gltf, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    glb = struct.pack("<4sII", b"glTF", 2, 12+8+len(encoded)+8+len(binary))
    glb += struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    (OUT / "fly.glb").write_bytes(glb)
    meta = {"format_version": 1, "asset": "fly.glb", "license": "Apache-2.0", "specimen": "female NeuroMechFly v2",
            "forward_axis": "+X", "up_axis": "+Y", "left_axis": "-Z", "units": "millimeters",
            "body_to_gltf_rotation": to_gltf.tolist(), "bbox": [np.min(bounds, 0).tolist(), np.max(bounds, 0).tolist()],
            "head_bbox": head_bounds, "brain_anchor": np.mean(head_bounds, axis=0).tolist(),
            "brain_anchor_qualification": "Head mesh bounding-box center; illustrative cross-specimen placement, not an anatomical registration",
            "neutral_pose_max_joint_error": error, "meshes": len(gltf["meshes"]), "triangles": triangles,
            "sha256": sha(OUT / "fly.glb")}
    write_json(OUT / "fly-meta.json", meta)
    return meta


def build_neurons():
    neurons_path = SOURCE / "Xenova/fruit-fly-simulation/public/data/neurons.json.gz"
    neurons = json.loads(gzip.decompress(neurons_path.read_bytes()))
    published_map = np.fromfile(SOURCE / "ngxson/fly-llm-demo/public/data/sub2global.i32", dtype="<i4")
    groups = ROOT / "data/connectorch-groups-v1/groups.npz"
    with np.load(groups, allow_pickle=False) as archive:
        ids = archive["node_ids"]
        group_metadata = json.loads(str(archive["metadata_json"]))
    source_ids = np.asarray([row[0] for row in neurons], dtype=np.int64)
    if len(ids) != 49393 or not np.array_equal(source_ids[published_map], ids):
        raise ValueError("Published subset does not exactly match checkpoint biological ID ordering")
    rows = [neurons[i] for i in published_map]
    valid = [row[6] is not None and len(row[6]) == 3 for row in rows]
    measured = np.asarray([row[6] for row, present in zip(rows, valid) if present], dtype=float)
    result = {"format_version": 1, "node_count": len(ids), "positioned_count": sum(valid), "missing_count": len(ids)-sum(valid),
              "positions": [row[6] if present else None for row, present in zip(rows, valid)],
              "body_ids": ids.tolist(), "superclass": [row[2] for row in rows], "side": [row[3] for row in rows],
              "valid": valid, "units": "8nm voxels", "position_semantics": "MaleCNS somaLocation8nm; missing locations are null",
              "axis_description": {"x": "left-right", "y": "dorsal-ventral", "z": "anterior-posterior"},
              "bbox": [measured.min(0).tolist(), measured.max(0).tolist()],
              "license": "CC-BY-4.0", "attribution": "FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research",
              "mapping": "Model index i -> groups.npz node_ids[i] == neurons[sub2global[i]].bodyId; all49393 verified",
              "groups_sha256": sha(groups), "source_sha256": sha(neurons_path),
              "checkpoint_graph_sha256": group_metadata["frozen_buffers_sha256"],
              "qualification": "Somata are point locations, not full neuronal morphologies. Some ascending/descending somata lie outside the brain. Brain/body overlay is illustrative."}
    write_json(OUT / "neuron-geometry.json", result)
    write_json(ROOT / "results/story-demo-v1/neuron-layout-source.json", {
        "neurons": [{"node_index": i, "body_id": str(int(ids[i])), "position": row[6]}
                    for i, (row, present) in enumerate(zip(rows, valid)) if present],
        "coordinate_system": {"type": "anatomical", "units": "8 nm voxel coordinates",
            "source": "https://huggingface.co/spaces/Xenova/fruit-fly-simulation/resolve/776d115ee5aa934578a87fd6d260d138084f59c1/public/data/neurons.json.gz",
            "source_sha256": result["source_sha256"], "groups_sha256": result["groups_sha256"],
            "full_neuron_count": len(ids), "missing_positions": len(ids)-sum(valid),
            "axis_description": result["axis_description"], "license": result["license"],
            "attribution": result["attribution"], "mapping": result["mapping"],
            "qualification": result["qualification"]},
        "missing_positions": len(ids)-sum(valid), "full_neuron_count": len(ids)})
    return {key: result[key] for key in ("node_count", "positioned_count", "missing_count", "groups_sha256", "source_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Recreate ignored source cache from pinned public provenance")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = SOURCE / "download-receipts.json"
    if args.download:
        receipt = json.loads((OUT / "provenance.json").read_text())
        for item in receipt["assets"]:
            destination = ROOT / item["local_path"]
            if destination.is_file() and sha(destination) == item["sha256"]:
                continue
            data = urllib.request.urlopen(item["source_url"], timeout=60).read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Downloaded source hash differs: " + item["source_url"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        write_json(receipt_path, receipt)
    receipt = json.loads(receipt_path.read_text())
    for item in receipt["assets"]:
        if sha(ROOT / item["local_path"]) != item["sha256"]:
            raise ValueError("Downloaded asset changed: " + item["local_path"])
    for name in ("Body-Apache-2.0.txt", "Body-MIT.txt"):
        shutil.copyfile(SOURCE / "Xenova/fruit-fly-simulation/licenses" / name, OUT / name)
    shutil.copyfile(SOURCE / "Xenova/fruit-fly-simulation/LICENSE", OUT / "Xenova-LICENSE.txt")
    notice = (BODY / "NOTICE").read_text()
    notice += "\nDerived asset: fly.glb, converted into a neutral-pose, Y-up glTF mesh; original shape and uniform scales retained.\n"
    notice += "\nNeuron point data: MaleCNS v1.0. Credit FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research. CC BY4.0: https://creativecommons.org/licenses/by/4.0/\n"
    notice += "Source: https://male-cns.janelia.org/download/ ; packaged by https://huggingface.co/spaces/Xenova/fruit-fly-simulation and https://huggingface.co/spaces/ngxson/fly-llm-demo .\n"
    notice += "Neuron subset index mapping independently matched to the checkpoint's verified body-ID mapping. Missing coordinates remain null. Female body and male connectome are different specimens; placement and motion are illustrative.\n"
    (OUT / "NOTICE.txt").write_text(notice)
    write_json(OUT / "provenance.json", receipt)
    result = {"fly": build_fly(), "neurons": build_neurons(), "source_receipt_sha256": sha(SOURCE / "download-receipts.json")}
    write_json(OUT / "geometry-provenance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
