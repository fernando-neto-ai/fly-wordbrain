#!/usr/bin/env python3
"""Verify checkpoint-to-body-ID alignment before assigning anatomical gain groups."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
GRAPH_SHA256 = "08bd21db0883698ac98501f4d70bb01eb72ea2736874fdb9092f9243b889db7b"
MODEL_SHA256 = "355f06c44d14e38af50e9c801f51839c37a0a56ac4ca016da2be9b3ae6f215ad"
CLASSES = ("cb_sensory", "visual_projection", "cb_intrinsic", "ascending_neuron", "descending_neuron")
FROZEN = ("brain.w_offsets", "brain.w_indices", "brain.w_values", "brain.in_index", "brain.out_index")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def verify_alignment(ids, superclass, types, ptr, source, weight, checkpoint):
    """Match every checkpoint edge to annotated sorted biological node IDs.

    The checkpoint has fewer edges than the source induced subgraph. It stays
    authoritative: omitted source edges are never restored. Require every
    retained endpoint, sign and magnitude ratio to agree before labeling nodes.
    """
    if ids.dtype.kind not in "iu" or ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not np.all(np.diff(ids) > 0):
        raise ValueError("Source biological IDs must be unique sorted integers")
    keep = np.isin(superclass, CLASSES)
    selected = np.flatnonzero(keep)
    n = len(selected)
    wp, wc, wv = (checkpoint[k] for k in ("brain.w_offsets", "brain.w_indices", "brain.w_values"))
    if len(wp) != n + 1 or wp[0] != 0 or wp[-1] != len(wc) or len(wc) != len(wv):
        raise ValueError("Source central population and checkpoint CSR dimensions differ")
    remap = np.full(len(ids), -1, dtype=np.int32)
    remap[selected] = np.arange(n, dtype=np.int32)
    target = np.repeat(np.arange(len(ids), dtype=np.int32), np.diff(ptr))
    retained = keep[source] & keep[target]
    r, c, w = remap[target[retained]], remap[source[retained]], weight[retained]
    keys = r.astype(np.int64) * n + c
    order = np.argsort(keys)
    keys, w = keys[order], w[order]
    if len(keys) == 0 or np.any(np.diff(keys) <= 0):
        raise ValueError("Source induced graph has empty or ambiguous edge endpoints")
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(wp))
    ck = rows * n + wc
    if len(ck) == 0 or np.any(np.diff(ck) <= 0):
        raise ValueError("Checkpoint CSR is empty or not canonically sorted/distinct")
    positions = np.searchsorted(keys, ck)
    if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], ck):
        raise ValueError("A checkpoint edge is absent under the proposed biological ID mapping")
    aligned = w[positions]
    if not np.array_equal(np.sign(aligned), np.sign(wv)):
        raise ValueError("Checkpoint/source edge signs disagree")
    nonzero = aligned != 0
    if not np.any(nonzero):
        raise ValueError("No nonzero edge can verify the source normalization")
    scale = np.float32(np.median(wv[nonzero] / aligned[nonzero]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid global source-to-checkpoint scale")
    expected = aligned * scale
    if not np.allclose(wv, expected, rtol=2e-6, atol=1e-10):
        raise ValueError("Checkpoint weights are not a single global scaling of source edges")
    node_ids = ids[selected]
    cell_types = types[selected]
    labels = np.asarray(["type:" + t if t else "unknown_id:" + str(i)
                         for t, i in zip(cell_types, node_ids)])
    type_labels, node_type_index = np.unique(labels, return_inverse=True)
    pair_count = len(np.unique(node_type_index[wc] * len(type_labels) + node_type_index[rows]))
    metadata = {
        "node_mapping": "Sorted biological IDs for the five reference superclasses; every checkpoint edge, sign and scaled magnitude verified",
        "nodes": n, "checkpoint_edges": len(wc), "source_induced_edges": len(keys),
        "source_edges_not_in_checkpoint": len(keys) - len(wc),
        "omission_reason": "Not established by this mapping audit; checkpoint topology is retained exactly",
        "source_to_checkpoint_global_scale": float(scale),
        "max_weight_absolute_error": float(np.max(np.abs(wv - expected))),
        "zero_checkpoint_edges": int(np.sum(wv == 0)),
        "distinct_annotated_types": len(np.unique(cell_types[cell_types != ""])),
        "untyped_nodes": int(np.sum(cell_types == "")), "node_type_groups": len(type_labels),
        "full_type_pair_groups": pair_count, "factorized_gain_parameters": 2 * len(type_labels),
        "parameterization": "W0[e] * (1 + 0.1*tanh(theta_source[type(src)] + theta_destination[type(dst)]))",
        "unknown_type_policy": "A distinct group for each biological ID lacking an official type; no type aliases",
        "frozen_buffers_sha256": {k: array_hash(checkpoint[k]) for k in FROZEN},
    }
    return {"node_ids": node_ids, "node_type_index": node_type_index.astype(np.int64),
            "node_type_labels": type_labels, "node_cell_type": cell_types}, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data/plastic-graph/graph.npz")
    p.add_argument("--model", type=Path, default=ROOT / "data/ngxson-fly-llm-hf" / REVISION / "model.safetensors")
    p.add_argument("--output", type=Path, default=ROOT / "data/connectorch-groups-v1")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a grouping artifact")
    source_hash, model_hash = file_hash(args.graph), file_hash(args.model)
    if source_hash != GRAPH_SHA256 or model_hash != MODEL_SHA256:
        raise ValueError("Pinned source/checkpoint file hash mismatch")
    with safe_open(args.model, framework="np") as f:
        checkpoint = {k: f.get_tensor(k) for k in FROZEN}
    with np.load(args.graph, allow_pickle=False) as a:
        arrays, meta = verify_alignment(a["ids"], a["superclass"], a["node_type"],
                                       a["incoming_ptr"], a["incoming_pre"], a["incoming_weight"], checkpoint)
    meta.update(schema_version=1, graph_sha256=source_hash, model_sha256=model_hash,
                code_sha256=file_hash(__file__), model_revision=REVISION)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "groups.npz", **arrays, metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
    meta["groups_npz_sha256"] = file_hash(args.output / "groups.npz")
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
