#!/usr/bin/env python3
"""Package the exact 49,393-neuron reference connectome as a redistributable dataset.

The graph this project trains on is currently reachable only by parsing a model
checkpoint. This writes it as plain typed arrays with a manifest, joined to the
biological MaleCNS body IDs, cell types, superclasses and soma positions that the
anatomical-group audit already verified against the source.

Two weight encodings are emitted. ``edges_weight.f32`` is canonical and exact.
``edges_weight.i16`` is a symmetric fixed-point quantisation for browser payloads;
its measured round-trip error is recorded in the manifest, and it is never the
reference for a numerical claim.

Nothing here is modified: the emitted CSR arrays are hashed and compared against
the frozen-buffer digests recorded in the trained checkpoints.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"


def digest_array(array):
    """Match the trainer's frozen-buffer digest: dtype, shape, then raw bytes."""
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(tuple(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def write_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    path.write_bytes(array.tobytes())
    return {"file": str(path.relative_to(path.parents[1])), "dtype": str(array.dtype),
            "shape": list(array.shape), "bytes": array.nbytes,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.write_bytes(body)
    return {"file": str(path.relative_to(path.parents[1])), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned reference model directory")
    parser.add_argument("--groups", type=Path, required=True, help="Verified anatomical groups NPZ")
    parser.add_argument("--geometry", type=Path, required=True, help="Demo neuron-geometry.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                 local_files_only=True, torch_dtype=torch.float32).cpu()
    brain = model.brain
    config = model.config
    buffers = {name: getattr(brain, name).detach().cpu().numpy() for name in
               ("w_offsets", "w_indices", "w_values", "in_index", "out_index")}
    frozen = {"brain." + name: digest_array(value) for name, value in buffers.items()}
    # The trainers hash these buffers in their original dtypes. Record those dtypes so a
    # third-party verifier re-derives the same digests instead of guessing an encoding.
    original_dtypes = {"brain." + name: str(value.dtype) for name, value in buffers.items()}

    with np.load(args.groups, allow_pickle=False) as z:
        node_ids = z["node_ids"]
        type_index = z["node_type_index"]
        type_labels = [str(x) for x in z["node_type_labels"]]
        cell_type = [str(x) for x in z["node_cell_type"]]
        group_metadata = json.loads(str(z["metadata_json"].item()))
    if group_metadata["frozen_buffers_sha256"] != frozen:
        raise SystemExit("Anatomical groups belong to a different reference graph")

    geometry = json.loads(args.geometry.read_text())
    if geometry["checkpoint_graph_sha256"] != frozen:
        raise SystemExit("Geometry belongs to a different reference graph")
    if [int(b) for b in geometry["body_ids"]] != node_ids.tolist():
        raise SystemExit("Geometry body IDs disagree with the anatomical-group mapping")

    offsets, indices, values = buffers["w_offsets"], buffers["w_indices"], buffers["w_values"]
    if int(indices.max()) >= 2 ** 16:
        raise SystemExit("Source indices no longer fit uint16; change the packed encoding")
    if offsets[-1] != values.size or offsets.size != config.n_neurons + 1:
        raise SystemExit("CSR offsets are inconsistent with the edge count")

    # Symmetric fixed point for the browser payload; the f32 array stays canonical.
    scale = float(np.abs(values).max()) / 32767.0
    packed = np.rint(values / scale).astype(np.int16)
    round_trip = packed.astype(np.float32) * scale
    quantisation = {"encoding": "symmetric fixed point, value = code * scale",
                    "scale": scale, "max_absolute_error": float(np.abs(round_trip - values).max()),
                    "max_relative_error": float(np.abs(round_trip - values).max() / np.abs(values).max()),
                    "sign_preserved": bool(np.array_equal(np.sign(round_trip), np.sign(values)))}

    positions = np.array([p if p else [0, 0, 0] for p in geometry["positions"]], dtype=np.int32)
    valid = np.array(geometry["valid"], dtype=np.uint8)

    out = args.output
    files = {
        "edges_offsets": write_array(out / "graph/edges_offsets.i32", offsets.astype(np.int32)),
        "edges_source": write_array(out / "graph/edges_source.u16", indices.astype(np.uint16)),
        "edges_weight_f32": write_array(out / "graph/edges_weight.f32", values.astype(np.float32)),
        "edges_weight_i16": write_array(out / "graph/edges_weight.i16", packed),
        "input_neurons": write_array(out / "interface/in_index.i32", buffers["in_index"].astype(np.int32)),
        "output_neurons": write_array(out / "interface/out_index.i32", buffers["out_index"].astype(np.int32)),
        "body_id": write_array(out / "neurons/body_id.i64", node_ids.astype(np.int64)),
        "type_index": write_array(out / "neurons/type_index.i32", type_index.astype(np.int32)),
        "soma_position": write_array(out / "neurons/soma_position.i32", positions),
        "soma_valid": write_array(out / "neurons/soma_valid.u8", valid),
        "type_labels": write_json(out / "neurons/type_labels.json", type_labels),
        "cell_type": write_json(out / "neurons/cell_type.json", cell_type),
        "superclass": write_json(out / "neurons/superclass.json", list(geometry["superclass"])),
        "side": write_json(out / "neurons/side.json", list(geometry["side"])),
    }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "fly-connectome-49k",
        "description": "The exact 49,393-neuron / 9,050,172-edge central-brain graph used by "
                       "the ngxson Fly LLM, joined to MaleCNS body IDs, cell types and somata.",
        "neurons": int(config.n_neurons),
        "edges": int(config.n_edges),
        "csr": {"orientation": "row = destination neuron, edges_source = presynaptic neuron index",
                "offsets_length": int(offsets.size),
                "reconstruct": "for dst in range(49393): for e in range(offsets[dst], offsets[dst+1]): "
                               "W[dst, edges_source[e]] = edges_weight[e]"},
        "weights": {"signed": True, "canonical": "graph/edges_weight.f32",
                    "negative_edges": int((values < 0).sum()), "positive_edges": int((values > 0).sum()),
                    "zero_edges": int((values == 0).sum()),
                    "absolute_max": float(np.abs(values).max()), "quantised": quantisation},
        "interface": {"n_in_declared": int(config.n_in), "in_index_length": int(buffers["in_index"].size),
                      "delay_slots": int(config.delay_k),
                      "neurons_per_slot": int(config.n_in // config.delay_k),
                      "note": "Eight delay slots each drive n_in // delay_k neurons; the remainder of "
                              "the declared n_in is unused by integer division."},
        "soma": {"positioned": int(valid.sum()), "missing": int(valid.size - valid.sum()),
                 "units": geometry["units"], "semantics": geometry["position_semantics"],
                 "missing_encoding": "soma_position rows are zero where soma_valid is 0",
                 "qualification": geometry["qualification"]},
        "types": {"groups": len(type_labels), "annotated": group_metadata["distinct_annotated_types"],
                  "untyped_nodes": group_metadata["untyped_nodes"],
                  "policy": group_metadata["unknown_type_policy"]},
        "provenance": {
            "reference_repository": "ngxson/fly-llm-hf", "reference_revision": REVISION,
            "reference_weights_sha256": group_metadata["model_sha256"],
            "source_connectome": "MaleCNS v1.0 central brain "
                                 "(cb_sensory, visual_projection, cb_intrinsic, ascending_neuron, descending_neuron)",
            "source_induced_edges": group_metadata["source_induced_edges"],
            "source_edges_not_in_checkpoint": group_metadata["source_edges_not_in_checkpoint"],
            "source_to_checkpoint_global_scale": group_metadata["source_to_checkpoint_global_scale"],
            "max_weight_absolute_error_against_source": group_metadata["max_weight_absolute_error"],
            "omission_reason": group_metadata["omission_reason"],
            "groups_sha256": geometry["groups_sha256"],
            "packager_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "frozen_buffers_sha256": frozen,
        "frozen_buffers_original_dtype": original_dtypes,
        "frozen_buffers_rebuild": {
            "brain.w_offsets": "graph/edges_offsets.i32", "brain.w_indices": "graph/edges_source.u16",
            "brain.w_values": "graph/edges_weight.f32", "brain.in_index": "interface/in_index.i32",
            "brain.out_index": "interface/out_index.i32",
            "note": "Cast each packed array back to its original dtype, then hash "
                    "dtype + shape + raw bytes to reproduce the checkpoint digest."},
        "files": files,
        "license": {"data": "CC-BY-4.0",
                    "attribution": "FlyEM / HHMI Janelia Research Campus, University of Cambridge, "
                                   "MRC Laboratory of Molecular Biology, Google Research",
                    "packaging_code": "MIT"},
    }
    write_json(out / "manifest.json", manifest)
    total = sum(f["bytes"] for f in files.values())
    print(json.dumps({"output": str(out), "total_bytes": total, "total_mb": round(total / 1e6, 2),
                      "edges": manifest["edges"], "neurons": manifest["neurons"],
                      "quantisation": quantisation, "frozen_buffers_verified": True}, indent=2))


if __name__ == "__main__":
    main()
