#!/usr/bin/env python3
"""Reproduce the pinned, complete Doomfly graph and native kernel on this host."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "doomfly"
DATA = VENDOR / "connectome_data" / "malecns_v1"
EXPECTED_COUNTS = {"neurons": 166700, "edges": 25582938, "synaptic_contacts": 124177617}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def verify(path, info):
    if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
        raise RuntimeError("Source integrity mismatch (will not overwrite): " + str(path))


def download(path, info):
    if path.exists():
        verify(path, info)
        emit("source_verified", file=path.name, cached=True)
        return
    part = path.with_suffix(path.suffix + ".partial")
    emit("download_start", file=path.name, expected_bytes=info["bytes"], url=info["url"])
    start = time.monotonic()
    count = last_report = 0
    with urllib.request.urlopen(info["url"], timeout=60) as response, part.open("wb") as f:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            count += len(chunk)
            if count - last_report >= 64 * 1024 * 1024:
                emit("download_progress", file=path.name, bytes=count,
                     expected_bytes=info["bytes"], seconds=round(time.monotonic() - start, 1))
                last_report = count
    verify(part, info)
    part.replace(path)
    emit("source_verified", file=path.name, cached=False, seconds=round(time.monotonic() - start, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "vendor/doomfly-source.json").read_text())
    for name, expected in source["files"].items():
        if digest(VENDOR / name) != expected:
            raise RuntimeError("Vendored upstream file changed: " + name)
    reference = source["reference_manifest"]
    reference_path = ROOT / "vendor" / reference["vendored_path"]
    if digest(reference_path) != reference["sha256"]:
        raise RuntimeError("Vendored upstream reference manifest changed")
    expected_manifest = json.loads(reference_path.read_text())
    locked = json.loads((VENDOR / "data-provenance/malecns_v1/source.lock.json").read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    for name, info in locked.items():
        download(DATA / name, info)
    if args.download_only:
        return
    import pyarrow as pa
    import pyarrow.ipc as ipc
    schemas = {}
    required = {
        "annotations.feather": {"bodyId", "superclass", "statusLabel", "status", "type", "assignedOlHex1", "assignedOlHex2", "rootSide", "somaSide"},
        "neurotransmitters.feather": {"body", "consensus_nt"},
        "edges.feather": {"body_pre", "body_post", "weight"},
    }
    for name, columns in required.items():
        reader = ipc.open_file(pa.memory_map(str(DATA / name), "r"))
        schemas[name] = [str(field) for field in reader.schema]
        missing = columns - set(reader.schema.names)
        if missing:
            emit("schema_mismatch", file=name, missing=sorted(missing), schema=schemas[name])
            raise RuntimeError("Official source schema differs from pinned importer; investigate before remapping.")
    emit("schemas_verified")
    env = dict(os.environ, PYTHONPATH=str(VENDOR))
    start = time.monotonic()
    for module in ("doom.connectome", "doom.prepare", "doom.build_kernel"):
        emit("upstream_module_start", module=module)
        subprocess.run([sys.executable, "-m", module], cwd=VENDOR, env=env, check=True)
        emit("upstream_module_complete", module=module)
    output = VENDOR / "outputs/doom/malecns_v1"
    manifest = json.loads((output / "manifest.json").read_text())
    for key, value in EXPECTED_COUNTS.items():
        if manifest[key] != value:
            raise RuntimeError("Full graph count mismatch: " + key)
    for key in ("retina_total", "retina_mapped", "retina_unmapped", "uncertain_sign_neurons"):
        if manifest[key] != expected_manifest[key]:
            raise RuntimeError("Prepared graph differs from upstream reference: " + key)
    # Upstream's saved manifest appends four BCI readouts to its ten biological
    # readouts. Current pinned prepare.py emits the same 14 records in ascending
    # node-index order. Check identity/type/side, not that historical list order.
    readout_key = lambda item: item["index"]
    if sorted(manifest["readouts"], key=readout_key) != sorted(expected_manifest["readouts"], key=readout_key):
        raise RuntimeError("Prepared readout identities differ from upstream reference")
    library = VENDOR / "outputs/doom" / ("libneural.dylib" if sys.platform == "darwin" else "libneural.so")
    build = json.loads(library.with_suffix(library.suffix + ".json").read_text())
    report = {
        "host": platform.node(), "python": sys.executable, "python_version": sys.version,
        "upstream_source": source, "source_hashes": locked, "source_schemas": schemas,
        "counts": EXPECTED_COUNTS, "graph_path": str(output / "graph.npz"),
        "graph_sha256": digest(output / "graph.npz"), "kernel_path": str(library), "kernel_build": build,
        "import_report": json.loads((DATA / "normalized/report.json").read_text()),
        "runtime_packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pyarrow", "numba")},
        "setup_seconds_excluding_download": round(time.monotonic() - start, 1),
        "preserved": {"raw_annotation_files": True, "all_upstream_retained_nodes_and_edges": True,
                      "upstream_native_dynamics_byte_identical": True, "dt_ms": 0.1},
        "upstream_reference_readouts": {
            "same_records_when_sorted_by_node_index": True,
            "same_list_order": manifest["readouts"] == expected_manifest["readouts"],
            "explanation": "Saved upstream manifest appends four BCI readouts; pinned prepare.py emits the same fourteen records in ascending node-index order."},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/graph-setup.json").write_text(json.dumps(report, indent=2) + "\n")
    emit("graph_setup_complete", report=str(ROOT / "results/graph-setup.json"),
         graph_path=report["graph_path"], graph_sha256=report["graph_sha256"], counts=EXPECTED_COUNTS)


if __name__ == "__main__":
    main()
