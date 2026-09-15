#!/usr/bin/env python3
"""Audit exact graph import and full-size reference/Connectorch CPU/MPS gradients.

Uses only the first two training stories and 32 next-token targets per story.
One synchronized forward/backward per case is an execution check, not a
steady-state throughput benchmark. No training run or test-set evaluation occurs.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_SHA256 = "c1f553263545744786a8811ee6af7ace8c452820bbb99d15140b81625e40b3a8"
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
CRITERIA = {"loss_abs_lt": 1e-4, "all_parameter_gradient_relative_l2_lt": 1e-3,
            "all_parameters_and_gradients_finite": True, "padding_gradient_exactly_zero": True,
            "canonical_buffers_unchanged": True, "both_factorized_gain_gradients_nonzero": True,
            "native_mps_forward_and_backward_kernels": True, "source_files_unchanged": True}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_receipt():
    names = ["scripts/audit_connectorch.py", "scripts/prepare_ngxson.py", "scripts/train_ngxson.py",
             "scripts/prepare_connectorch_groups.py", "scripts/train_connectorch.py", "connectorch-source.json",
             "fly_wordbrain/connectorch_model.py", "fly_wordbrain/connectorch_backend.py",
             "fly_wordbrain/metal_sparse_trainable.py"]
    return {name: file_hash(ROOT / name) for name in names}


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_hashes(model):
    import torch
    result = {}
    for name, value in model.named_buffers():
        if name in ("brain.w_offsets", "brain.w_indices", "brain.w_values", "brain.in_index", "brain.out_index",
                    "brain.node_type_index", "brain.edge_group_index"):
            if value.requires_grad:
                raise ValueError("Canonical buffer requires gradients: " + name)
            array = value.detach().cpu().contiguous().numpy()
            result[name] = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()
    return result


def run_pass(model, inputs, targets, mask, device):
    import torch
    import torch.nn.functional as F
    model.train()
    before = checkpoint_hashes(model)
    model.zero_grad(set_to_none=True)
    if hasattr(model, "backend_metadata"):
        model.backend_metadata()  # Build/upload runtime layouts before timing.
    inputs, targets, mask = (value.to(device) for value in (inputs, targets, mask))
    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    out = model(input_ids=inputs, attention_mask=mask, use_cache=True)
    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    grads, finite = {}, {}
    for name, parameter in model.named_parameters():
        finite[name] = bool(torch.isfinite(parameter).all()) and parameter.grad is not None
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().cpu().clone()
        finite[name] = finite[name] and bool(torch.isfinite(gradient).all())
        grads[name] = gradient
    gain_names = ("brain.edge_theta_source", "brain.edge_theta_destination")
    gain_gradients = {name: {"norm": float(grads[name].norm()), "nonzero": int(grads[name].count_nonzero())}
                      for name in gain_names if name in grads}
    backend = model.backend_metadata() if hasattr(model, "backend_metadata") else {"backend": "pinned_upstream_cpu_csr"}
    pad_zero = int(model.brain.wte.weight.grad[model.config.pad_token_id].count_nonzero()) == 0
    graph_unchanged = before == checkpoint_hashes(model)
    if hasattr(model, "verify_frozen"):
        graph_unchanged = graph_unchanged and model.verify_frozen()
    native = True
    if device == "mps":
        info = backend["sparse_backend"]
        launches = info["kernel_launch_count"]
        native = (info["backend"] == "metal_trainable_csr" and info["host_fallback"] is False
                  and info["connectorch_backend"] == "metal_csr" and launches["forward"] == inputs.shape[1]
                  and launches["state_backward"] == inputs.shape[1] - 1
                  and launches["edge_backward"] == (inputs.shape[1] if gain_gradients else 0))
    record = {"device": device, "loss": float(loss.detach().cpu()), "targets": int((targets != 0).sum()),
              "forward_backward_seconds": elapsed, "parameters_and_gradients_finite": finite,
              "frozen_buffers_unchanged": graph_unchanged, "frozen_buffers_sha256": before,
              "padding_gradient_zero": pad_zero, "factorized_gain_gradients": gain_gradients,
              "backend": backend, "native_kernel_gate": native,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "passed": all(finite.values()) and bool(finite) and graph_unchanged and pad_zero and native
                        and all(item["nonzero"] > 0 for item in gain_gradients.values())}
    tensors = {"gradients": grads, "logits": out.logits.detach().cpu().clone(),
               "state": out.cache_params.state.detach().cpu().clone(),
               "last_tokens": out.cache_params.last_tokens.detach().cpu().clone(), "seq_len": out.cache_params.seq_len}
    return record, tensors


def compare(reference, observed):
    import torch
    a_record, a = reference
    b_record, b = observed
    if a["gradients"].keys() != b["gradients"].keys():
        raise ValueError("Parameter gradient keys differ between compared implementations")
    gradients = {}
    for name in a["gradients"]:
        x, y = a["gradients"][name], b["gradients"][name]
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Invalid gradient shape or nonfinite gradient: " + name)
        delta = x - y
        gradients[name] = {"max_abs": float(delta.abs().max()),
                           "relative_l2": float(delta.norm()) / max(float(x.norm()), 1e-12),
                           "reference_norm": float(x.norm()), "observed_norm": float(y.norm())}
    loss_error = abs(a_record["loss"] - b_record["loss"])
    worst = max(item["relative_l2"] for item in gradients.values())
    cache_equal = torch.equal(a["last_tokens"], b["last_tokens"]) and a["seq_len"] == b["seq_len"]
    return {"passed": a_record["passed"] and b_record["passed"] and loss_error < CRITERIA["loss_abs_lt"]
                      and worst < CRITERIA["all_parameter_gradient_relative_l2_lt"] and cache_equal,
            "loss_abs_difference": loss_error, "max_gradient_relative_l2": worst,
            "gradient_comparison": gradients, "logits_max_abs": float((a["logits"] - b["logits"]).abs().max()),
            "state_max_abs": float((a["state"] - b["state"]).abs().max()), "cache_ids_and_length_equal": cache_equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "data/ngxson-fly-llm-hf" / REVISION)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ngxson-tinystories-v1/dataset.json")
    parser.add_argument("--groups", type=Path, default=ROOT / "data/connectorch-groups-v1/groups.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh audit output; previous receipts are never overwritten")
    result = {"passed": False, "started_at": datetime.now(timezone.utc).isoformat(), "criteria": CRITERIA,
              "scope": "Exact full graph import and one teacher-forced forward/backward per case; 2 training stories x 32 tokens. Not convergence, throughput, or test-set evaluation.",
              "cases": {}, "comparisons": {}}
    try:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise ValueError("Run with PYTORCH_ENABLE_MPS_FALLBACK=0 explicitly")
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from fly_wordbrain.connectorch_backend import connectorch_receipt, exact_connectome_ir
        from fly_wordbrain.connectorch_model import build_connectorch_model
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("Native MPS shader runtime required")
        before_sources = source_receipt()
        result.update(source_sha256=before_sources, torch=torch.__version__, python=sys.version,
                      platform=platform.platform(), pytorch_enable_mps_fallback=os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                      connectorch=connectorch_receipt())
        prepared = load_helper("connectorch_audit_reference_download", ROOT / "scripts/prepare_ngxson.py")
        original_trainer = load_helper("connectorch_audit_original_trainer", ROOT / "scripts/train_ngxson.py")
        result["reference_files"] = {name: prepared.verify(args.model / name, spec) for name, spec in prepared.FILES.items()}
        dataset_hash, groups_hash = file_hash(args.data), file_hash(args.groups)
        if dataset_hash != DATA_SHA256:
            raise ValueError("TinyStories dataset differs from the pinned reference reconstruction")
        manifest = json.loads(args.groups.with_name("manifest.json").read_text())
        if groups_hash != manifest["groups_npz_sha256"]:
            raise ValueError("Grouping archive differs from its manifest hash")
        with np.load(args.groups, allow_pickle=False) as archive:
            node_types = np.asarray(archive["node_type_index"])
            node_ids = np.asarray(archive["node_ids"])
            node_cell_types = np.asarray(archive["node_cell_type"])
            group_metadata = json.loads(str(archive["metadata_json"].item()))
        if {k: v for k, v in manifest.items() if k != "groups_npz_sha256"} != group_metadata:
            raise ValueError("Grouping metadata and manifest differ")
        result.update(data_file_sha256=dataset_hash, groups_file_sha256=groups_hash, group_metadata=group_metadata,
                      model_directory=str(args.model.resolve()), data_file=str(args.data.resolve()), groups_file=str(args.groups.resolve()))
        dataset = json.loads(args.data.read_text())
        rows = dataset["train"][:2]
        if len(rows) != 2 or any(len(row["ids"]) < 33 for row in rows):
            raise ValueError("Audit requires exactly two complete training sequences with >=33 IDs")
        result["training_story_ids"] = [row["id"] for row in rows]
        inputs, targets, mask = original_trainer.batch_tensors(rows, "cpu")
        inputs, targets, mask = (value[:, :32].contiguous() for value in (inputs, targets, mask))
        reference = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, dtype=torch.float32)
        if reference.config.n_neurons != 49393 or reference.config.n_edges != 9050172:
            raise ValueError("Unexpected full-reference graph dimensions")
        if checkpoint_hashes(reference) != group_metadata["frozen_buffers_sha256"]:
            raise ValueError("Grouping does not bind these reference graph and interface buffers")
        exact_ir = exact_connectome_ir(dict(reference.named_buffers()), node_ids, node_cell_types)
        result["exact_ir"] = {"passed": True, "nodes": exact_ir.num_nodes, "edges": exact_ir.num_edges,
                              "all_node_ids_endpoints_and_weights_exact": True, "normalization": "none; released checkpoint values",
                              "grouping_archive_sha256": groups_hash}
        del exact_ir
        gc.collect()
        print("Verified exact Connectorch IR: 49,393 nodes / 9,050,172 edges", flush=True)
        result["initialization"] = original_trainer.initialize_from_scratch(reference, 42)
        original = run_pass(reference, inputs, targets, mask, "cpu")
        result["cases"]["upstream_d128_fixed_cpu"] = original[0]
        reference.zero_grad(set_to_none=True)
        for device in ("cpu", "mps"):
            model = build_connectorch_model(reference, d_embed=128, plasticity="fixed", node_type_index=node_types, seed=42, device=device)
            observed = run_pass(model, inputs, targets, mask, device)
            name = "connectorch_d128_fixed_" + device
            result["cases"][name] = observed[0]
            comparison = compare(original, observed)
            result["comparisons"]["upstream_cpu_vs_" + name] = comparison
            print(json.dumps({"case": name, "passed": comparison["passed"], "loss_abs_difference": comparison["loss_abs_difference"],
                              "max_gradient_relative_l2": comparison["max_gradient_relative_l2"]}), flush=True)
            del observed, model
            gc.collect()
            torch.mps.empty_cache()
            if not comparison["passed"]:
                raise RuntimeError("Full reference parity failed: " + name)
        del original
        gc.collect()
        cpu = build_connectorch_model(reference, d_embed=32, plasticity="bounded10", node_type_index=node_types, seed=42)
        cpu_result = run_pass(cpu, inputs, targets, mask, "cpu")
        result["cases"]["connectorch_d32_bounded10_cpu"] = cpu_result[0]
        del cpu
        gc.collect()
        mps = build_connectorch_model(reference, d_embed=32, plasticity="bounded10", node_type_index=node_types, seed=42, device="mps")
        mps_result = run_pass(mps, inputs, targets, mask, "mps")
        result["cases"]["connectorch_d32_bounded10_mps"] = mps_result[0]
        comparison = compare(cpu_result, mps_result)
        result["comparisons"]["connectorch_d32_bounded10_cpu_vs_mps"] = comparison
        result["bounded10_adaptation_audit"] = mps.adaptation_report()
        print(json.dumps({"case": "connectorch_d32_bounded10_cpu_vs_mps", "passed": comparison["passed"],
                          "loss_abs_difference": comparison["loss_abs_difference"], "max_gradient_relative_l2": comparison["max_gradient_relative_l2"]}), flush=True)
        result["sources_unchanged"] = before_sources == source_receipt()
        result["passed"] = (all(item["passed"] for item in result["comparisons"].values())
                            and result["sources_unchanged"] and all(case["targets"] == 64 for case in result["cases"].values()))
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects a receipt written by another concurrent process.
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "failure": result.get("failure")}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
