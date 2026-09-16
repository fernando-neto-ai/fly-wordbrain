#!/usr/bin/env python3
"""Verify a packaged connectome using only its own files and manifest.

No model, no checkpoint and no framework beyond NumPy: this is the check a third
party can run after downloading the dataset. It confirms every file's SHA-256,
rebuilds the CSR structure, re-derives the frozen-buffer digests that the trained
checkpoints record, and measures the quantised encoding against the canonical one.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DTYPES = {"i16": np.int16, "i32": np.int32, "i64": np.int64, "u8": np.uint8,
          "u16": np.uint16, "f32": np.float32}


def load(root, entry):
    raw = (root / entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise SystemExit("SHA-256 mismatch: " + entry["file"])
    if entry["file"].endswith(".json"):
        return json.loads(raw)
    array = np.frombuffer(raw, dtype=DTYPES[entry["file"].rsplit(".", 1)[1]])
    return array.reshape(entry["shape"])


def digest_array(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(tuple(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    root = args.package
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest["files"]
    checks = []

    offsets = load(root, files["edges_offsets"])
    source = load(root, files["edges_source"])
    weight = load(root, files["edges_weight_f32"])
    packed = load(root, files["edges_weight_i16"])
    checks.append(("every file SHA-256", True))

    n, e = manifest["neurons"], manifest["edges"]
    checks.append(("neuron and edge counts", offsets.size == n + 1 and source.size == e and weight.size == e))
    checks.append(("CSR offsets non-decreasing", bool(np.all(np.diff(offsets) >= 0))))
    checks.append(("CSR offsets span every edge", int(offsets[0]) == 0 and int(offsets[-1]) == e))
    checks.append(("source indices in range", int(source.max()) < n and int(source.min()) >= 0))
    checks.append(("weights finite", bool(np.isfinite(weight).all())))
    checks.append(("signed weight counts", int((weight < 0).sum()) == manifest["weights"]["negative_edges"]
                   and int((weight > 0).sum()) == manifest["weights"]["positive_edges"]))

    # The trainers hash these buffers in their original dtypes, which the manifest records,
    # so a third party reproduces the checkpoint digests without guessing an encoding.
    frozen = manifest["frozen_buffers_sha256"]
    dtypes = manifest["frozen_buffers_original_dtype"]
    packed_by_buffer = {"brain.w_offsets": offsets, "brain.w_indices": source,
                        "brain.w_values": weight,
                        "brain.in_index": load(root, files["input_neurons"]),
                        "brain.out_index": load(root, files["output_neurons"])}
    for name, array in packed_by_buffer.items():
        checks.append((f"frozen buffer {name}",
                       frozen[name] == digest_array(array.astype(np.dtype(dtypes[name])))))

    q = manifest["weights"]["quantised"]
    error = float(np.abs(packed.astype(np.float32) * q["scale"] - weight).max())
    checks.append(("quantised round-trip within manifest", error <= q["max_absolute_error"] * 1.000001))
    checks.append(("quantised signs preserved",
                   bool(np.array_equal(np.sign(packed.astype(np.float32)), np.sign(weight)))))

    labels = load(root, files["type_labels"])
    type_index = load(root, files["type_index"])
    checks.append(("type index covers its labels",
                   int(type_index.max()) == len(labels) - 1 and int(type_index.min()) == 0))
    valid = load(root, files["soma_valid"])
    checks.append(("positioned soma count", int(valid.sum()) == manifest["soma"]["positioned"]))
    position = load(root, files["soma_position"])
    checks.append(("missing somata are zeroed", bool((position[valid == 0] == 0).all())))
    body = load(root, files["body_id"])
    checks.append(("body IDs unique", len(np.unique(body)) == n))

    width = max(len(name) for name, _ in checks)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:{width}}")
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed.")
        return 1
    print(f"\nAll {len(checks)} checks passed: {n:,} neurons, {e:,} edges, "
          f"{int(valid.sum()):,} positioned somata, {len(labels):,} cell-type groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
