#!/usr/bin/env python3
"""Audit actual Metal CSR values/gradients and synchronized sparse-only timing.

Requires the exported full graph and real MPS hardware; never creates a dense
adjacency matrix or trains a model. CPU references are sparse, with float64
values for correctness and float32 for optional matched execution timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_wordbrain import metal_sparse
from fly_wordbrain.brain import array_digest
from fly_wordbrain.plastic_brain import frozen_sparse_mm, sorted_csr_columns


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpu_csr(ptr, columns, values, n, dtype=torch.float32):
    return torch.sparse_csr_tensor(torch.tensor(ptr, dtype=torch.int64),
                                   torch.tensor(columns, dtype=torch.int64),
                                   torch.tensor(values, dtype=dtype), size=(n, n),
                                   check_invariants=True)


def timed_calls(function, warmups, repetitions, synchronize):
    for _ in range(warmups):
        function()
    synchronize()
    samples = []
    for _ in range(repetitions):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(samples), "minimum_seconds": min(samples),
            "p90_seconds": float(np.quantile(samples, .9)), "samples_seconds": samples}


def errors(actual, reference, atol, rtol):
    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    difference = (actual - reference).abs()
    budget = atol + rtol * reference.abs()
    return {"passed": bool(torch.all(difference <= budget).item()),
            "max_absolute_error": float(difference.max().item()),
            "rms_error": float(torch.sqrt(torch.mean(difference.square())).item()),
            "max_error_over_tolerance": float((difference / budget).max().item()),
            "reference_rms": float(torch.sqrt(torch.mean(reference.square())).item()),
            "absolute_tolerance": atol, "relative_tolerance": rtol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-cpu-timing", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Actual MPS and compile_shader are required; no fallback benchmark")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK before benchmarking")
    if (not args.batches or any(b not in (1, 2, 8) for b in args.batches)
            or not 1 <= args.warmups <= 10 or not 1 <= args.repetitions <= 100
            or args.threads < 1 or args.atol <= 0 or args.rtol < 0):
        raise ValueError("Require batches1/2/8, bounded positive warmups/repeats, valid threads/tolerances")
    if args.output.exists():
        raise ValueError("Output exists; choose a fresh benchmark receipt")
    torch.set_num_threads(args.threads)
    path = args.graph / "graph.npz" if args.graph.is_dir() else args.graph
    started = time.perf_counter()
    source_hash = file_hash(path)
    with np.load(path, allow_pickle=False) as source:
        keys = ("ids", "ptr", "post", "weight", "incoming_ptr", "incoming_pre", "incoming_weight")
        graph = {key: source[key] for key in keys}
    n, e = len(graph["ids"]), len(graph["weight"])
    if n != 166700 or e != 25582938:
        raise ValueError("This performance receipt requires the full original graph")
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if source_hash != metadata["graph_npz_sha256"]:
        raise ValueError("Graph archive differs from its source receipt")
    for key, value in graph.items():
        if array_digest([value]) != metadata["arrays"][key]["sha256"]:
            raise ValueError("Graph array hash mismatch: " + key)
    out_col, out_weight, _ = sorted_csr_columns(graph["ptr"], graph["post"], graph["weight"])
    incoming = metal_sparse.DenseMpsCSR(graph["incoming_ptr"], graph["incoming_pre"], graph["incoming_weight"], (n, n))
    outgoing = metal_sparse.DenseMpsCSR(graph["ptr"], out_col, out_weight, (n, n))
    cpu_in = cpu_csr(graph["incoming_ptr"], graph["incoming_pre"], graph["incoming_weight"], n)
    cpu_out = cpu_csr(graph["ptr"], out_col, out_weight, n)
    oracle_in, oracle_out = cpu_in.to(torch.float64), cpu_out.to(torch.float64)
    identity = [array_digest([t.detach().cpu().numpy() for t in
                             (m.crow_indices(), m.col_indices(), m.values())]) for m in (incoming, outgoing)]
    report = {"scope": "Sparse matrix forward and explicit transpose state-gradient only; no neural dynamics, optimizer, or language performance measured",
              "graph_sha256": source_hash, "graph_metadata_sha256": file_hash(metadata_path),
              "neurons": n, "edges": e, "host": platform.node(), "platform": platform.platform(),
              "python": platform.python_version(), "torch": str(torch.__version__), "device": "mps",
              "fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
              "threads": args.threads, "warmups": args.warmups, "repetitions": args.repetitions,
              "source_sha256": {"benchmark_metal_sparse.py": file_hash(__file__),
                                "metal_sparse.py": file_hash(Path(metal_sparse.__file__)),
                                "plastic_brain.py": file_hash(ROOT / "fly_wordbrain/plastic_brain.py")},
              "setup_seconds_excluded_from_timings": time.perf_counter() - started, "batches": []}
    for batch in args.batches:
        generator = torch.Generator().manual_seed(1729 + batch)
        h_cpu = torch.randn(batch, n, generator=generator)
        upstream_cpu = torch.randn(batch, n, generator=generator)
        h, upstream = h_cpu.to("mps"), upstream_cpu.to("mps")
        expected_y = torch.sparse.mm(oracle_in, h_cpu.double().T).T
        expected_gradient = torch.sparse.mm(oracle_out, upstream_cpu.double().T).T
        launch_before = metal_sparse.kernel_launch_count()

        def forward():
            return metal_sparse.mps_csr_mm(h, incoming)

        def forward_backward(device="mps"):
            state = (h if device == "mps" else h_cpu).detach().requires_grad_()
            a, at = (incoming, outgoing) if device == "mps" else (cpu_in, cpu_out)
            value = frozen_sparse_mm(state, a, at)
            gradient = torch.autograd.grad(value, state, grad_outputs=upstream if device == "mps" else upstream_cpu)[0]
            return value, gradient

        observed_y, observed_gradient = forward_backward()
        torch.mps.synchronize()
        comparisons = {"forward": errors(observed_y, expected_y, args.atol, args.rtol),
                       "state_gradient": errors(observed_gradient, expected_gradient, args.atol, args.rtol)}
        entry = {"batch": batch, "float64_cpu_sparse_oracle": comparisons,
                 "metal_forward": timed_calls(forward, args.warmups, args.repetitions, torch.mps.synchronize),
                 "metal_forward_backward": timed_calls(forward_backward, args.warmups, args.repetitions, torch.mps.synchronize)}
        if not args.skip_cpu_timing:
            entry["cpu_forward"] = timed_calls(lambda: torch.sparse.mm(cpu_in, h_cpu.T).T,
                                               args.warmups, args.repetitions, lambda: None)
            entry["cpu_forward_backward"] = timed_calls(lambda: forward_backward("cpu"),
                                                        args.warmups, args.repetitions, lambda: None)
            entry["forward_speedup_over_cpu"] = entry["cpu_forward"]["median_seconds"] / entry["metal_forward"]["median_seconds"]
            entry["forward_backward_speedup_over_cpu"] = entry["cpu_forward_backward"]["median_seconds"] / entry["metal_forward_backward"]["median_seconds"]
        entry["actual_metal_kernel_launches"] = metal_sparse.kernel_launch_count() - launch_before
        assert entry["actual_metal_kernel_launches"] == 2 + 3 * (args.warmups + args.repetitions)
        report["batches"].append(entry)
        print(json.dumps({"event": "metal_sparse_batch", **entry}, allow_nan=False), flush=True)
    for matrix, expected in zip((incoming, outgoing), identity):
        matrix.verify_frozen()
        actual = array_digest([t.detach().cpu().numpy() for t in
                               (matrix.crow_indices(), matrix.col_indices(), matrix.values())])
        if actual != expected:
            raise RuntimeError("Fixed GPU CSR changed during benchmark")
    report["backend"] = metal_sparse.backend_metadata()
    report["frozen_matrices_verified"] = True
    report["passed"] = all(test["passed"] for entry in report["batches"]
                            for test in entry["float64_cpu_sparse_oracle"].values())
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    if not report["passed"]:
        raise RuntimeError("Metal/CPU numerical parity failed; inspect the saved benchmark receipt")


if __name__ == "__main__":
    main()
