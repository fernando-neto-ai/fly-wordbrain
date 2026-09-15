#!/usr/bin/env python3
"""Export every pinned Doomfly edge plus an explicitly annotated plastic subset.

The original outgoing CSR is also the transpose of the incoming-current matrix:
incoming A[post, pre], original A.T[pre, post]. No coalescing, pruning, row
normalization, type aliases, or neural simulation is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/doomfly"
GROUPS = ("hDeltaH", "hDeltaA", "hDeltaI", "hDeltaG")
SOURCE_TYPE = "hDeltaB"
EXPECTED_N = 166700
EXPECTED_E = 25582938
EXPECTED_CONTACTS = 124177617
EXPECTED_GRAPH_SHA256 = "fcd8c9cd5dbf7de751da339cf644c32e2d05f32304aa3f0d8a6512446290d061"
EXPECTED_CANONICAL_SHA256 = "19f1fe72719f8fdf5769ac50be6c18b8a02afc52a1bedb6b2527b05369b70210"


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(arrays):
    """Matches fly_wordbrain.brain.array_digest without importing neural code."""
    h = hashlib.sha256()
    for a in arrays:
        if a.dtype.hasobject:
            raise ValueError("Object arrays cannot have reproducible content hashes")
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def align_annotation_types(ids, body_ids, cell_types):
    """Exact bodyId join; preserve spelling and blank types, reject ambiguity."""
    body_ids = np.asarray(body_ids)
    if body_ids.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise ValueError("bodyId and graph IDs must be integer arrays")
    if len(body_ids) != len(cell_types) or len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("Official annotation bodyId must be unique and aligned")
    order = np.argsort(body_ids)
    sorted_ids = body_ids[order]
    positions = np.searchsorted(sorted_ids, ids)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(sorted_ids[positions], ids):
        raise ValueError("An original graph node lacks an exact official annotation bodyId")
    labels = []
    for value in cell_types:
        if value is None:
            labels.append("")
        elif isinstance(value, str):
            labels.append(value)
        else:
            raise ValueError("Annotation type must be a string or null; no coercion/aliases")
    return np.asarray(labels, dtype=np.str_)[order[positions]]


def validate_graph(graph):
    ids, ptr, post, weight = (graph[k] for k in ("ids", "ptr", "post", "weight"))
    n, e = len(ids), len(post)
    expected_dtypes = {"ids": "int64", "ptr": "int64", "post": "int32", "weight": "float32"}
    for key, dtype in expected_dtypes.items():
        if graph[key].ndim != 1 or str(graph[key].dtype) != dtype:
            raise ValueError("Canonical graph dtype/shape differs: " + key)
    if len(np.unique(ids)) != n or len(ptr) != n + 1 or len(weight) != e:
        raise ValueError("Malformed canonical graph dimensions/IDs")
    if ptr[0] != 0 or ptr[-1] != e or np.any(np.diff(ptr) < 0):
        raise ValueError("Malformed canonical outgoing CSR pointer")
    if np.any(post < 0) or np.any(post >= n) or not np.all(np.isfinite(weight)):
        raise ValueError("Invalid destination or nonfinite frozen weight")
    for key in ("retina",):
        nodes = graph[key]
        if nodes.ndim != 1 or nodes.dtype.kind not in "iu" or np.any(nodes < 0) or np.any(nodes >= n):
            raise ValueError("Invalid original sensory population: " + key)
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Duplicate original sensory node: " + key)
    if len(graph["superclass"]) != n:
        raise ValueError("Original superclass array is not node-aligned")


def distance_summary(distances):
    distances = np.asarray(distances)
    valid = distances[distances >= 0]
    values, counts = np.unique(valid, return_counts=True)
    return {"count": int(len(distances)), "reachable": int(len(valid)),
            "unreachable": int(len(distances) - len(valid)),
            "min": int(valid.min()) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "max": int(valid.max()) if len(valid) else None,
            "histogram": {str(int(v)): int(c) for v, c in zip(values, counts)}}


def shortest_hops(ptr, columns, seeds):
    """Multi-source, directed, unweighted distances; no dense source×node table."""
    n = len(ptr) - 1
    if not len(seeds):
        return np.full(n, -1, np.int32)
    adjacency = csr_matrix((np.ones(len(columns), dtype=np.uint8), columns, ptr), shape=(n, n))
    distances = dijkstra(adjacency, directed=True, indices=seeds, unweighted=True, min_only=True)
    result = np.full(n, -1, np.int32)
    finite = np.isfinite(distances)
    result[finite] = distances[finite].astype(np.int32)
    return result


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0., .25, .5, .9, .99, 1.)}


def build_export(graph, node_type):
    validate_graph(graph)
    n, e = len(graph["ids"]), len(graph["post"])
    if node_type.shape != (n,) or node_type.dtype.kind != "U":
        raise ValueError("Exact node_type labels must be a node-aligned Unicode array")
    missing_types = [label for label in (SOURCE_TYPE,) + GROUPS if not np.any(node_type == label)]
    if missing_types:
        raise ValueError("Exact official annotation types absent; aliases are forbidden: " + repr(missing_types))
    ptr, post, weight = (graph[k] for k in ("ptr", "post", "weight"))
    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
    incoming_order = np.argsort(post, kind="stable").astype(np.int64, copy=False)
    incoming_ptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype(np.int64)
    incoming_pre = pre[incoming_order]
    # Stable sort changes storage order only; every canonical edge, including
    # duplicates, self-edges, negative and zero weights, remains represented.
    exported = dict(graph)
    exported.update(incoming_ptr=incoming_ptr, incoming_pre=incoming_pre,
                    incoming_weight=weight[incoming_order], incoming_edge_ids=incoming_order,
                    node_type=node_type,
                    descending=np.flatnonzero(graph["superclass"] == "descending_neuron").astype(np.int32))
    group_for_node = np.full(n, -1, np.int8)
    for group, label in enumerate(GROUPS):
        group_for_node[node_type == label] = group
    candidates = np.flatnonzero((node_type == SOURCE_TYPE)[pre] & (group_for_node[post] >= 0)).astype(np.int64)
    exported.update(candidate_edge_ids=candidates, candidate_pre=pre[candidates],
                    candidate_post=post[candidates], candidate_group=group_for_node[post[candidates]])
    exported["incoming_signed_sum"] = np.bincount(post, weights=weight, minlength=n)
    exported["incoming_abs_sum"] = np.bincount(post, weights=np.abs(weight), minlength=n)
    exported["incoming_positive_sum"] = np.bincount(post, weights=np.maximum(weight, 0), minlength=n)
    exported["incoming_negative_magnitude_sum"] = np.bincount(post, weights=np.maximum(-weight, 0), minlength=n)
    emit("csr_export_ready", neurons=n, edges=e, candidate_edges=len(candidates))
    from_retina = shortest_hops(ptr, post, graph["retina"])
    to_descending = shortest_hops(incoming_ptr, incoming_pre, exported["descending"])
    exported.update(hops_from_retina=from_retina, hops_to_descending=to_descending)
    via = np.full(len(candidates), -1, np.int32)
    valid = (from_retina[pre[candidates]] >= 0) & (to_descending[post[candidates]] >= 0)
    via[valid] = from_retina[pre[candidates[valid]]] + 1 + to_descending[post[candidates[valid]]]
    exported["candidate_hops_retina_via_edge_to_descending"] = via
    groups = []
    for group, label in enumerate(GROUPS):
        mask = exported["candidate_group"] == group
        edge_ids = candidates[mask]
        groups.append({"index": group, "target_type": label,
                       "target_nodes": int(np.sum(node_type == label)),
                       "candidate_edges": int(len(edge_ids)), "zero_edges": bool(not len(edge_ids)),
                       "connected_presynaptic_nodes": int(len(np.unique(pre[edge_ids]))),
                       "connected_postsynaptic_nodes": int(len(np.unique(post[edge_ids]))),
                       "positive_edges": int(np.sum(weight[edge_ids] > 0)),
                       "negative_edges": int(np.sum(weight[edge_ids] < 0)),
                       "zero_weight_edges": int(np.sum(weight[edge_ids] == 0)),
                       "retina_to_connected_presynaptic": distance_summary(from_retina[np.unique(pre[edge_ids])]),
                       "connected_postsynaptic_to_descending": distance_summary(to_descending[np.unique(post[edge_ids])]),
                       "retina_via_candidate_edge_to_descending": distance_summary(via[mask])})
    summary = {"neurons": n, "edges": e, "retinal_inputs": int(len(graph["retina"])),
               "descending_neurons": int(len(exported["descending"])),
               "candidate_source_type": SOURCE_TYPE, "candidate_source_nodes": int(np.sum(node_type == SOURCE_TYPE)),
               "candidate_edges": int(len(candidates)), "candidate_groups": groups,
               "group_names": list(GROUPS),
               "type_mapping": {"source_columns": ["bodyId", "type"], "join": "exact bodyId equality",
                                "aliases": {}, "null_type": "empty string", "all_graph_nodes_retained": True},
               "raw_weight_diagnostics": {"positive_edges": int(np.sum(weight > 0)),
                                          "negative_edges": int(np.sum(weight < 0)),
                                          "zero_weight_edges": int(np.sum(weight == 0)),
                                          **{key: quantiles(exported[key]) for key in (
                                              "incoming_abs_sum", "incoming_signed_sum", "incoming_positive_sum",
                                              "incoming_negative_magnitude_sum")}},
               "reachability": {"retina_to_all_nodes": distance_summary(from_retina),
                                "all_nodes_to_descending": distance_summary(to_descending),
                                "retina_to_descending": distance_summary(from_retina[exported["descending"]]),
                                "retina_to_all_hDeltaB": distance_summary(from_retina[node_type == SOURCE_TYPE]),
                                "retina_via_candidate_edge_to_descending": distance_summary(via),
                                "definition": "Directed unweighted hops, -1 unreachable; via-edge sum is shortest upstream distance + 1 + shortest downstream distance. This is a constrained walk/reachability diagnostic, not effective activity or biological latency."}}
    return exported, summary


def power_diagnostics(exported, iterations, seed=1729):
    """Bounded norm-ratio probe, explicitly not a guaranteed spectral estimate."""
    if iterations <= 0:
        return {"performed": False}
    n = len(exported["ids"])
    a = csr_matrix((exported["incoming_weight"].astype(np.float64),
                    exported["incoming_pre"], exported["incoming_ptr"]), shape=(n, n))
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    trace = []
    for step in range(iterations):
        y = a @ x
        ratio = float(np.linalg.norm(y))
        rayleigh = float(x @ y)
        residual = float(np.linalg.norm(y - rayleigh * x))
        trace.append({"iteration": step + 1, "norm_ratio": ratio, "rayleigh": rayleigh,
                      "relative_eigen_residual": residual / max(ratio, 1e-300)})
        if ratio == 0 or not np.isfinite(ratio):
            break
        x = y / ratio
    return {"performed": True, "iterations": len(trace), "seed": seed,
            "matrix": "raw signed incoming CSR, float64 arithmetic", "trace": trace,
            "last_norm_ratio": trace[-1]["norm_ratio"],
            "warning": "A power norm ratio is only an estimate. Complex/multiple dominant eigenvalues and directed nonnormality can prevent convergence. This is not a stability proof or chosen normalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=VENDOR / "outputs/doom/malecns_v1/graph.npz")
    parser.add_argument("--annotations", type=Path, default=VENDOR / "connectome_data/malecns_v1/annotations.feather")
    parser.add_argument("--output", type=Path, default=ROOT / "data/plastic-graph")
    parser.add_argument("--power-iterations", type=int, default=40)
    args = parser.parse_args()
    if not 0 <= args.power_iterations <= 200:
        parser.error("--power-iterations must be between 0 and 200")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Output directory is nonempty; refusing to overwrite an export")
    start = time.monotonic()
    graph_sha = file_sha256(args.graph)
    if graph_sha != EXPECTED_GRAPH_SHA256:
        raise RuntimeError("Original graph archive differs from verified frozen experiment")
    source_lock = VENDOR / "data-provenance/malecns_v1/source.lock.json"
    vendor_source = ROOT / "vendor/doomfly-source.json"
    pinned = json.loads(vendor_source.read_text())
    if file_sha256(source_lock) != pinned["files"]["data-provenance/malecns_v1/source.lock.json"]:
        raise RuntimeError("Official source lock differs from pinned upstream")
    locked = json.loads(source_lock.read_text())
    sources = {}
    for name, receipt in locked.items():
        path = args.annotations if name == "annotations.feather" else args.annotations.parent / name
        actual = file_sha256(path)
        if actual != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
            raise RuntimeError("Official raw source changed: " + name)
        sources[name] = {**receipt, "path": str(path.resolve())}
    manifest_path = args.graph.with_name("manifest.json")
    original_manifest = json.loads(manifest_path.read_text())
    if any(original_manifest[k] != v for k, v in (("neurons", EXPECTED_N), ("edges", EXPECTED_E),
                                                ("synaptic_contacts", EXPECTED_CONTACTS))):
        raise RuntimeError("Original graph manifest counts changed")
    with np.load(args.graph, allow_pickle=False) as archive:
        graph = {key: archive[key] for key in archive.files}
    canonical_digest = array_digest([graph[k] for k in ("ids", "ptr", "post", "weight")])
    if canonical_digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError("Canonical frozen graph arrays changed")
    if len(graph["ids"]) != EXPECTED_N or len(graph["post"]) != EXPECTED_E:
        raise RuntimeError("Canonical full graph count mismatch")
    import pyarrow.feather as feather
    annotations = feather.read_table(args.annotations, columns=["bodyId", "type"])
    node_type = align_annotation_types(graph["ids"], annotations["bodyId"].to_numpy(),
                                       annotations["type"].to_pylist())
    emit("pinned_sources_verified", original_graph_sha256=graph_sha, annotation_rows=len(annotations))
    exported, metadata = build_export(graph, node_type)
    if metadata["descending_neurons"] != 1314:
        raise RuntimeError("All-descending-neuron population changed")
    emit("candidate_reachability", groups=metadata["candidate_groups"], reachability=metadata["reachability"])
    metadata["power_diagnostics"] = power_diagnostics(exported, args.power_iterations)
    # Receipts cover original arrays and every derived array, not NumPy object addresses.
    receipts = {key: {"dtype": str(value.dtype), "shape": list(value.shape),
                      "sha256": array_digest([value])} for key, value in exported.items()}
    metadata.update(schema_version=1, completed=True, host=platform.node(),
                    original_graph_path=str(args.graph.resolve()), original_graph_sha256=graph_sha,
                    canonical_graph_array_sha256=canonical_digest, synaptic_contacts=EXPECTED_CONTACTS,
                    original_graph_manifest_sha256=file_sha256(manifest_path), original_manifest=original_manifest,
                    source_lock_sha256=file_sha256(source_lock), vendor_source_sha256=file_sha256(vendor_source),
                    upstream_commit=pinned["commit"], sources=sources,
                    code_sha256={"prepare_plastic_graph.py": file_sha256(__file__)},
                    original_array_names=list(graph), arrays=receipts,
                    array_hash_algorithm="SHA256(str(dtype) + str(shape) + contiguous raw bytes), matching brain.py",
                    normalization={"kind": "global", "global_scale": 1.0, "applied": False,
                                   "note": "Export retains raw signed float32 weights; runner must record its chosen global scalar. No row normalization."},
                    csr_orientation={"incoming": "A[post,pre]: incoming_ptr/incoming_pre/incoming_weight",
                                     "transpose": "A.T[pre,post]: original ptr/post/weight",
                                     "edge_mapping": "incoming_edge_ids maps each incoming slot to original canonical edge; no coalescing"},
                    plasticity={"source_type": SOURCE_TYPE, "target_types": list(GROUPS),
                                "selection": "All existing edges with exact official source and target type labels",
                                "base_weights_frozen": True, "new_edges": 0, "removed_edges": 0,
                                "fast_state_exported": False})
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "graph.npz.partial"
    with temporary.open("wb") as f:
        np.savez(f, **exported)
    temporary.replace(args.output / "graph.npz")
    metadata["graph_npz_sha256"] = file_sha256(args.output / "graph.npz")
    metadata["graph_npz_bytes"] = (args.output / "graph.npz").stat().st_size
    metadata["elapsed_seconds"] = round(time.monotonic() - start, 3)
    temporary_metadata = args.output / "metadata.json.partial"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(args.output / "metadata.json")
    emit("plastic_graph_export_complete", output=str(args.output.resolve()),
         candidate_edges=metadata["candidate_edges"], graph_npz_sha256=metadata["graph_npz_sha256"],
         raw_power_last_norm_ratio=metadata["power_diagnostics"].get("last_norm_ratio"),
         elapsed_seconds=metadata["elapsed_seconds"])


if __name__ == "__main__":
    main()
