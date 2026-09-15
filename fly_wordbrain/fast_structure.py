"""Discover existing-edge engineering candidates without rewriting the brain.

Anatomical labels and directed reachability define candidate pools. They do not
establish that a selected synapse is biologically plastic. No text, targets,
validation data, neural simulation, or GPU is used in selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("candidate_edge_ids", "candidate_group", "candidate_pre", "candidate_post", "candidate_weight")
ORIGINAL_GROUPS = ("hDeltaH", "hDeltaA", "hDeltaI", "hDeltaG")
CX_PREFIXES = ("hDelta", "vDelta", "FC", "FB", "PF", "EPG", "PEN", "PEG", "ER")
GRAPH_KEYS = ("ptr", "post", "weight", "candidate_edge_ids", "candidate_group")
DISCOVERY_KEYS = GRAPH_KEYS + ("node_type", "superclass", "hops_from_retina", "hops_to_descending")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot identify immutable graph data")
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def original_selection_sha256(graph_arrays):
    return array_digest([np.asarray(graph_arrays[name], dtype=np.int64)
                         for name in ("candidate_edge_ids", "candidate_group")])


def _graph_dimensions(graph):
    ptr, post, weight = (np.asarray(graph[key]) for key in ("ptr", "post", "weight"))
    if (ptr.ndim != 1 or ptr.dtype.kind not in "iu" or len(ptr) < 2
            or post.ndim != 1 or post.dtype.kind not in "iu" or weight.shape != post.shape
            or weight.dtype != np.float32 or ptr[0] != 0 or ptr[-1] != len(post)
            or np.any(np.diff(ptr) < 0) or np.any(post < 0) or np.any(post >= len(ptr) - 1)
            or not np.isfinite(weight).all()):
        raise ValueError("Invalid canonical outgoing graph arrays")
    return len(ptr) - 1, len(post)


def validate_structure(arrays, metadata, graph_arrays, graph_sha256):
    """Verify sidecar IDs, group mapping, endpoints and exact original weights.

    Required graph arrays: ptr, post, weight, candidate_edge_ids, candidate_group.
    ``graph_sha256`` must be the caller's verified hash of the canonical archive.
    """
    n, edges = _graph_dimensions(graph_arrays)
    if metadata.get("schema") != 1 or metadata.get("kind") != "fast_structure":
        raise ValueError("Unknown fast-structure schema")
    if metadata.get("graph_sha256") != graph_sha256:
        raise ValueError("Sidecar belongs to a different canonical graph SHA")
    if metadata.get("original_selection_sha256") != original_selection_sha256(graph_arrays):
        raise ValueError("Original candidate selection changed")
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError("Sidecar arrays must contain exactly the declared candidate fields")
    size = len(arrays["candidate_edge_ids"])
    if not 1 <= size <= 16384:
        raise ValueError("Candidate sidecar must contain 1..16384 existing edges")
    for name in ARRAY_KEYS:
        array = np.asarray(arrays[name])
        dtype = np.float32 if name == "candidate_weight" else np.int64
        if array.shape != (size,) or array.dtype != dtype:
            raise ValueError("Invalid sidecar dtype/shape: " + name)
    ids, group = arrays["candidate_edge_ids"], arrays["candidate_group"]
    if np.any(ids < 0) or np.any(ids >= edges) or len(np.unique(ids)) != size:
        raise ValueError("Candidate edge IDs must be unique canonical IDs")
    names = metadata.get("group_names")
    if (not isinstance(names, list) or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names) or names[:4] != list(ORIGINAL_GROUPS)
            or not np.array_equal(np.unique(group), np.arange(len(names)))):
        raise ValueError("Groups must be contiguous, nonempty, and retain the original four names")
    original_ids = np.asarray(graph_arrays["candidate_edge_ids"], dtype=np.int64)
    original_group = np.asarray(graph_arrays["candidate_group"], dtype=np.int64)
    count = len(original_ids)
    if (not np.array_equal(ids[:count], original_ids) or not np.array_equal(group[:count], original_group)
            or np.any(group[count:] < 4)):
        raise ValueError("All original candidate IDs and groups must be preserved as the unchanged prefix")
    pre = np.searchsorted(graph_arrays["ptr"], ids, side="right") - 1
    if (not np.array_equal(arrays["candidate_pre"], pre)
            or not np.array_equal(arrays["candidate_post"], graph_arrays["post"][ids])):
        raise ValueError("Sidecar endpoint differs from an existing canonical edge")
    if not np.array_equal(arrays["candidate_weight"], graph_arrays["weight"][ids]):
        raise ValueError("Sidecar altered canonical base weights or signs")
    if metadata.get("selection_array_sha256") != array_digest([arrays[key] for key in ARRAY_KEYS]):
        raise ValueError("Candidate selection array checksum mismatch")
    if (metadata.get("candidate_edges") != size or metadata.get("neurons") != n
            or metadata.get("edges") != edges or metadata.get("original_candidate_edges") != count):
        raise ValueError("Sidecar counts disagree with the actual graph/selection")
    return True


def load_structure(structure_path, graph_arrays, graph_sha256):
    """Read the small NPZ plus same-stem JSON, then verify against the real graph."""
    path = Path(structure_path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != metadata.get("sidecar_sha256"):
        raise ValueError("Sidecar archive checksum mismatch")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    validate_structure(arrays, metadata, graph_arrays, graph_sha256)
    return arrays, metadata


def family_masks(graph, pre):
    """Priority-ordered, explicit label predicates; memberships may initially overlap."""
    node_type, superclass, post = graph["node_type"], graph["superclass"], graph["post"]
    cx = np.asarray([str(label).startswith(CX_PREFIXES) for label in node_type])
    kc, mbon = np.char.startswith(node_type, "KC"), np.char.startswith(node_type, "MBON")
    dn, vp, cb = (superclass == value for value in ("descending_neuron", "visual_projection", "cb_intrinsic"))
    hops = graph["hops_to_descending"]
    yield "KC_to_MBON", kc[pre] & mbon[post], "source type starts KC; target type starts MBON"
    yield "MBON_outputs", mbon[pre], "source type starts MBON"
    yield "central_complex_intrinsic", cx[pre] & cx[post], "both types match the explicitly listed central-complex prefixes"
    yield "central_complex_outputs", cx[pre] & ~cx[post] & (hops[post] <= 2), "source matches central-complex prefixes; target does not and lies at most two directed hops from descending readout"
    yield "visual_projection_to_descending", vp[pre] & dn[post], "source superclass visual_projection; target superclass descending_neuron"
    yield "central_brain_to_descending", cb[pre] & dn[post], "source superclass cb_intrinsic; target superclass descending_neuron"
    yield "visual_projection_to_output_afferent", vp[pre] & cb[post] & (hops[post] == 1), "source superclass visual_projection; target superclass cb_intrinsic and one hop from descending readout"
    yield "descending_recurrent", dn[pre] & dn[post], "both superclasses descending_neuron"


def _statistics(edge_ids, graph, pre):
    post, weight = graph["post"][edge_ids], graph["weight"][edge_ids]
    source_hops = graph["hops_from_retina"][pre[edge_ids]]
    output_hops = graph["hops_to_descending"][post]
    histogram = lambda values: {str(int(value)): int(count) for value, count in zip(*np.unique(values, return_counts=True))}
    return {"edges": len(edge_ids), "source_nodes": len(np.unique(pre[edge_ids])),
            "target_nodes": len(np.unique(post)), "positive_edges": int((weight > 0).sum()),
            "negative_edges": int((weight < 0).sum()), "zero_edges": int((weight == 0).sum()),
            "absolute_weight_sum": float(np.abs(weight).sum(dtype=np.float64)),
            "signed_weight_sum": float(weight.sum(dtype=np.float64)),
            "absolute_weight_quantiles": {str(q): float(np.quantile(np.abs(weight), q)) for q in (0., .5, .9, 1.)} if len(edge_ids) else {},
            "retinal_to_source_hops": histogram(source_hops), "target_to_descending_hops": histogram(output_hops),
            "retina_via_edge_to_descending_hops": histogram(source_hops + 1 + output_hops),
            "top_source_types": Counter(graph["node_type"][pre[edge_ids]].tolist()).most_common(8),
            "top_target_types": Counter(graph["node_type"][post].tolist()).most_common(8)}


def build_structure(graph, graph_sha256, target_edges=8192):
    """Retain the original subset, then evenly budget each nonempty family/sign pool."""
    n, edges = _graph_dimensions(graph)
    if type(target_edges) is not int or not len(graph["candidate_edge_ids"]) <= target_edges <= min(edges, 16384):
        raise ValueError("target_edges must retain the original selection and remain within 16384 existing edges")
    for name in ("node_type", "superclass", "hops_from_retina", "hops_to_descending"):
        if graph[name].shape != (n,):
            raise ValueError("Node annotation must align to every canonical node: " + name)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph["ptr"]))
    post, weight = graph["post"], graph["weight"]
    free = (graph["hops_from_retina"][pre] >= 0) & (graph["hops_to_descending"][post] >= 0) & (weight != 0)
    free[graph["candidate_edge_ids"]] = False
    pools, families = [], []
    for family, mask, predicate in family_masks(graph, pre):
        chosen = mask & free
        free[chosen] = False
        ids = np.flatnonzero(chosen).astype(np.int64)
        families.append({"family": family, "predicate": predicate, "available": _statistics(ids, graph, pre)})
        for sign, sign_mask in (("positive", weight[ids] > 0), ("negative", weight[ids] < 0)):
            signed = ids[sign_mask]
            if len(signed):
                pools.append({"name": family + "/" + sign, "family": family, "sign": sign, "edge_ids": signed})
    remaining = target_edges - len(graph["candidate_edge_ids"])
    if remaining > sum(len(pool["edge_ids"]) for pool in pools):
        raise ValueError("Anatomical candidate pools cannot supply the requested number of edges")
    quotas = np.zeros(len(pools), dtype=np.int64)
    # Balanced sign/family capacity; canonical pool order resolves quota remainders.
    while remaining:
        available = [i for i, pool in enumerate(pools) if quotas[i] < len(pool["edge_ids"])]
        share = max(1, remaining // len(available))
        for i in available:
            take = min(share, len(pools[i]["edge_ids"]) - int(quotas[i]), remaining)
            quotas[i] += take
            remaining -= take
            if not remaining:
                break
    selected = [np.asarray(graph["candidate_edge_ids"], dtype=np.int64)]
    assigned = [np.asarray(graph["candidate_group"], dtype=np.int64)]
    group_names, group_records = list(ORIGINAL_GROUPS), []
    for group, name in enumerate(ORIGINAL_GROUPS):
        ids = selected[0][assigned[0] == group]
        group_records.append({"index": group, "name": name, "preserved_original": True,
                              "selected": _statistics(ids, graph, pre)})
    for pool, quota in zip(pools, quotas):
        if not quota:
            continue
        ids = pool["edge_ids"]
        output_hops = graph["hops_to_descending"][post[ids]]
        score = np.abs(weight[ids]).astype(np.float64) / (1. + output_hops)
        via = graph["hops_from_retina"][pre[ids]] + 1 + output_hops
        order = np.lexsort((ids, via, -score))[:int(quota)]
        ids = ids[order]
        index = len(group_names)
        group_names.append(pool["name"])
        selected.append(ids)
        assigned.append(np.full(len(ids), index, dtype=np.int64))
        group_records.append({"index": index, "name": pool["name"], "family": pool["family"],
            "sign": pool["sign"], "preserved_original": False, "available_edges": len(pool["edge_ids"]),
            "selected": _statistics(ids, graph, pre)})
    ids = np.concatenate(selected)
    arrays = {"candidate_edge_ids": ids, "candidate_group": np.concatenate(assigned),
              "candidate_pre": pre[ids].astype(np.int64), "candidate_post": post[ids].astype(np.int64),
              "candidate_weight": weight[ids].copy()}
    metadata = {"schema": 1, "kind": "fast_structure", "graph_sha256": graph_sha256,
        "neurons": n, "edges": edges, "candidate_edges": len(ids),
        "original_candidate_edges": len(selected[0]), "group_names": group_names,
        "original_selection_sha256": original_selection_sha256(graph),
        "selection_array_sha256": array_digest([arrays[key] for key in ARRAY_KEYS]),
        "selection": {"source": "canonical graph embedded exact official node_type/superclass annotations",
            "central_complex_type_prefixes": list(CX_PREFIXES),
            "eligibility": "existing nonzero edge; source reachable from retina; target can reach a descending neuron",
            "overlap": "first matching family owns each new edge; original edges reserved first",
            "budget": "equal capacity-limited quota per nonempty family/sign pool; ordered remainder assignment",
            "ranking": "descending abs(base weight)/(1+target-to-descending hops), then shorter retinal-via-edge-output hops, then lower canonical edge ID",
            "data_used": "none; no training, validation or test text", "new_edges": 0,
            "removed_edges": 0, "base_weights_changed": 0,
            "interpretation": "Engineering candidate structures, not a claim of biological synaptic plasticity. Directed hops are reachability, not measured activity or causal efficacy."},
        "families": families, "groups": group_records, "selected": _statistics(ids, graph, pre)}
    validate_structure(arrays, metadata, graph, graph_sha256)
    return arrays, metadata


def save_structure(path, arrays, metadata):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Sidecar path must end in .npz")
    receipt = path.with_suffix(".json")
    if path.exists() or receipt.exists():
        raise ValueError("Refusing to overwrite an existing structure sidecar")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    saved = {**metadata, "sidecar_sha256": file_sha256(path), "sidecar_bytes": path.stat().st_size,
             "created_at_utc": datetime.now(timezone.utc).isoformat()}
    with receipt.open("x") as stream:
        json.dump(saved, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-edges", default=8192, type=int)
    args = parser.parse_args()
    path = args.graph / "graph.npz" if args.graph.is_dir() else args.graph
    before = file_sha256(path)
    with np.load(path, allow_pickle=False) as source:
        graph = {key: source[key] for key in DISCOVERY_KEYS}
    arrays, metadata = build_structure(graph, before, args.target_edges)
    saved = save_structure(args.output, arrays, metadata)
    load_structure(args.output, graph, before)
    if file_sha256(path) != before:
        raise RuntimeError("Canonical graph file changed during read-only discovery")
    print(json.dumps({"output": str(args.output.resolve()), "candidate_edges": saved["candidate_edges"],
                      "groups": len(saved["group_names"]), "graph_sha256": before,
                      "sidecar_sha256": saved["sidecar_sha256"], "sidecar_bytes": saved["sidecar_bytes"],
                      "graph_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
