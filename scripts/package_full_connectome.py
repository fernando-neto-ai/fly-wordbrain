#!/usr/bin/env python3
"""Package the complete MaleCNS v1.0 connectome as a redistributable dataset.

The fly-connectome-49k dataset carries only the central-brain subset the language
model runs on. This packages the whole graph the project started from — every
neuron, every edge, raw signed weights with no normalization applied — in the same
destination-major CSR convention, so the two datasets compose.

Weights are signed synaptic counts: magnitude from the flat connectome at minimum
confidence 0.5, sign from the neurotransmitter prediction. They are not
conductances and not measured physiological strengths.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The pin the anatomical-group audit verified against.
GRAPH_SHA256 = "08bd21db0883698ac98501f4d70bb01eb72ea2736874fdb9092f9243b889db7b"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def categorical(values):
    """Store repeated label strings as an index plus a label table."""
    labels = sorted({str(v) for v in values})
    lookup = {label: i for i, label in enumerate(labels)}
    return np.array([lookup[str(v)] for v in values], dtype=np.int32), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/plastic-graph/graph.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/plastic-graph/metadata.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")

    found = file_hash(args.graph)
    if found != GRAPH_SHA256:
        raise SystemExit(f"Source graph hash {found} is not the pinned {GRAPH_SHA256}")
    source_metadata = json.loads(args.metadata.read_text())
    if source_metadata["normalization"]["applied"]:
        raise SystemExit("This packager publishes raw weights; the source has been normalized")

    with np.load(args.graph, allow_pickle=False) as z:
        # Destination-major, matching the fly-connectome-49k convention exactly:
        # row = postsynaptic neuron, stored index = presynaptic neuron.
        offsets = z["incoming_ptr"]
        source = z["incoming_pre"]
        weight = z["incoming_weight"]
        body_id = z["ids"]
        superclass_raw = z["superclass"]
        cell_type_raw = z["node_type"]
        populations = {name: z[name] for name in ("retina", "lamina", "descending", "sugar")}
        hops_retina = z["hops_from_retina"]
        hops_descending = z["hops_to_descending"]

    neurons = int(body_id.size)
    edges = int(weight.size)
    if offsets.size != neurons + 1 or source.size != edges:
        raise SystemExit("CSR arrays are inconsistent")
    if int(offsets[-1]) != edges or int(offsets[0]) != 0:
        raise SystemExit("CSR offsets do not span the edge list")
    if int(source.max()) >= neurons or int(source.min()) < 0:
        raise SystemExit("Presynaptic indices fall outside the neuron range")
    if not np.isfinite(weight).all():
        raise SystemExit("Non-finite edge weight")
    if int(offsets.max()) > np.iinfo(np.int32).max:
        raise SystemExit("Edge offsets no longer fit int32")

    superclass_index, superclass_labels = categorical(superclass_raw)
    cell_type_index, cell_type_labels = categorical(cell_type_raw)

    out = args.output
    files = {
        "edges_offsets": write_array(out / "graph/edges_offsets.i32", offsets.astype(np.int32)),
        "edges_source": write_array(out / "graph/edges_source.i32", source.astype(np.int32)),
        "edges_weight": write_array(out / "graph/edges_weight.f32", weight.astype(np.float32)),
        "body_id": write_array(out / "neurons/body_id.i64", body_id.astype(np.int64)),
        "superclass_index": write_array(out / "neurons/superclass_index.i32", superclass_index),
        "cell_type_index": write_array(out / "neurons/cell_type_index.i32", cell_type_index),
        "superclass_labels": write_json(out / "neurons/superclass_labels.json", superclass_labels),
        "cell_type_labels": write_json(out / "neurons/cell_type_labels.json", cell_type_labels),
        "hops_from_retina": write_array(out / "derived/hops_from_retina.i32", hops_retina.astype(np.int32)),
        "hops_to_descending": write_array(out / "derived/hops_to_descending.i32", hops_descending.astype(np.int32)),
    }
    for name, values in populations.items():
        files[f"population_{name}"] = write_array(out / f"populations/{name}.i32", values.astype(np.int32))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "fly-connectome-malecns-166k",
        "description": "The complete MaleCNS v1.0 connectome as typed arrays: every neuron, "
                       "every directed edge, raw signed weights, destination-major CSR.",
        "neurons": neurons,
        "edges": edges,
        "synaptic_contacts": source_metadata["synaptic_contacts"],
        "csr": {"orientation": "row = postsynaptic neuron, edges_source = presynaptic neuron index",
                "offsets_length": int(offsets.size),
                "reconstruct": "for post in range(neurons): for e in range(offsets[post], offsets[post+1]): "
                               "W[post, edges_source[e]] = edges_weight[e]",
                "transpose_note": "Transpose for a presynaptic-major view; no edge is coalesced."},
        "weights": {
            "raw": True, "normalization_applied": False,
            "semantics": "Signed synaptic count: magnitude from the flat connectome at minimum "
                         "confidence 0.5, sign from the neurotransmitter prediction. Not a "
                         "conductance and not a measured physiological strength.",
            "positive_edges": int((weight > 0).sum()),
            "negative_edges": int((weight < 0).sum()),
            "zero_edges": int((weight == 0).sum()),
            "absolute_max": float(np.abs(weight).max()),
        },
        "neurons_detail": {
            "superclasses": len(superclass_labels),
            "cell_types": len(cell_type_labels),
            "untyped": int((cell_type_raw == "").sum()),
            "note": "Cell type is an empty label where the source carries no official type.",
        },
        "populations": {name: int(values.size) for name, values in populations.items()},
        "derived": {
            "hops_from_retina": "Unweighted shortest-path hop count from the retinal input set; "
                                "-1 where unreachable.",
            "hops_to_descending": "Unweighted shortest-path hop count to the descending set; "
                                  "-1 where unreachable.",
            "computed_by": "this project, not the upstream release",
        },
        "soma_positions": {
            "included": False,
            "note": "Soma coordinates are not part of this package. The central-brain subset "
                    "dataset fly-connectome-49k carries measured soma positions for its "
                    "49,393 neurons.",
        },
        "provenance": {
            "dataset": "MaleCNS v1.0 flat connectome",
            "upstream_files": source_metadata["sources"],
            "assembled_via": "github.com/nftechie/doomfly at " + source_metadata["upstream_commit"],
            "source_graph_sha256": found,
            "source_graph_bytes": int(Path(args.graph).stat().st_size),
            "packager_sha256": file_hash(Path(__file__)),
            "minimum_confidence": 0.5,
        },
        "related": {
            "central_brain_subset": "fernandofernandes/fly-connectome-49k",
            "models": ["fernandofernandes/fly-wordbrain-rank64",
                       "fernandofernandes/fly-wordbrain-rank32"],
        },
        "files": files,
        "license": {"data": "CC-BY-4.0",
                    "attribution": "FlyEM / HHMI Janelia Research Campus, University of Cambridge, "
                                   "MRC Laboratory of Molecular Biology, Google Research",
                    "packaging_code": "MIT"},
    }
    write_json(out / "manifest.json", manifest)
    total = sum(f["bytes"] for f in files.values())
    print(json.dumps({"output": str(out), "neurons": neurons, "edges": edges,
                      "total_mb": round(total / 1e6, 2),
                      "superclasses": len(superclass_labels), "cell_types": len(cell_type_labels),
                      "populations": manifest["populations"],
                      "positive": manifest["weights"]["positive_edges"],
                      "negative": manifest["weights"]["negative_edges"]}, indent=2))


if __name__ == "__main__":
    main()
