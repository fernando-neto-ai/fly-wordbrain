#!/usr/bin/env python3
"""Check rank-128 E/F initialization and full-graph CPU/MPS gradients on macm3.

Four forward/backward passes, no optimizer steps, and no reserved-test access.
The original full-readout audit and bound model/trainer sources are unchanged.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_connectorch as legacy

ARMS = {"E32rank128fixed": "fixed", "F32rank128bounded": "bounded10"}
EXTRA_PARAMETERS = {"brain.edge_theta_source", "brain.edge_theta_destination"}
COMMON_PARAMETERS = {"brain.wte.weight", "brain.in_proj", "brain.gain", "brain.bias", "brain.rec_gain",
                     "ln.weight", "ln.bias", "lm_head.0.weight", "lm_head.1.weight"}
TOTALS = {"fixed": 7183157, "bounded10": 7201479}
HOST_NAMES = {"macm3", "fernandos-macbook-pro-2"}
CONFIGS = {name: {"d_embed": 32, "readout_rank": 128, "history_length": 8,
                  "lag_map": list(range(8)), "seed": 42, "plasticity": plasticity}
           for name, plasticity in ARMS.items()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def process_guard(text, own_pid):
    records = []
    for line in text.splitlines():
        fields = line.strip().split(None, 3)
        require(len(fields) == 4, "Unparseable process inventory")
        records.append({"pid": int(fields[0]), "ppid": int(fields[1]), "state": fields[2], "command": fields[3]})
    by_pid = {row["pid"]: row for row in records}
    require(own_pid in by_pid, "Audit process absent from process inventory")
    ancestors, current = set(), own_pid
    while current in by_pid:
        current = by_pid[current]["ppid"]
        if current in ancestors:
            break
        ancestors.add(current)
    blocked, held, parents = [], [], []
    for row in records:
        if row["pid"] == own_pid or "Z" in row["state"]:
            continue
        try:
            tokens = shlex.split(row["command"])
        except ValueError:
            tokens = row["command"].split()
        executable = Path(tokens[0]).name.lower() if tokens else ""
        if not (executable.startswith("python") or executable.endswith(".py")):
            continue
        names = {Path(token).name for token in tokens}
        workers = any(name.startswith(("train_", "evaluate_", "audit_", "compare_ngxson_texts"))
                      and name.endswith(".py") for name in names)
        dispatcher = any(name.startswith(("run_connectorch_", "continue_connectorch_"))
                         and name.endswith(".py") for name in names)
        if workers:
            blocked.append(row)
        elif dispatcher and "T" in row["state"]:
            held.append(row)
        elif dispatcher and row["pid"] in ancestors:
            parents.append(row)
        elif dispatcher:
            blocked.append(row)
    require(not blocked, "Another training/audit worker or active dispatcher exists: " + json.dumps(blocked))
    return {"passed": True, "held_dispatchers": held, "ancestor_dispatchers": parents,
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def host_guard():
    hostname = socket.gethostname()
    require(platform.system() == "Darwin" and hostname.lower().removesuffix(".local") in HOST_NAMES,
            "Full decoder preflight must run on registered macm3")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(chip.startswith("Apple M3"), "Decoder preflight requires an Apple M3 host")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "Set PYTORCH_ENABLE_MPS_FALLBACK=0 explicitly")
    inventory = subprocess.check_output(["ps", "-axo", "pid=,ppid=,state=,command="], text=True)
    return {"hostname": hostname, "chip": chip, **process_guard(inventory, os.getpid())}


def source_receipt():
    import train_connectorch as trainer
    return {**legacy.source_receipt(), **trainer.source_receipt(),
            "scripts/audit_connectorch_decoder.py": legacy.file_hash(Path(__file__))}


def verify_initialization(fixed, bounded, fixed_logits, bounded_logits):
    """Accept only identical common tensors and the two zero gain vectors."""
    import torch
    import train_connectorch as trainer
    require(set(fixed) == COMMON_PARAMETERS, "Fixed initialization parameter names differ")
    require(set(bounded) == COMMON_PARAMETERS | EXTRA_PARAMETERS, "Bounded initialization parameter names differ")
    shared_equal = all(fixed[name].shape == bounded[name].shape
                       and fixed[name].dtype == bounded[name].dtype
                       and torch.equal(fixed[name], bounded[name]) for name in COMMON_PARAMETERS)
    require(shared_equal, "Shared E/F initial parameters differ")
    require(all(tuple(bounded[name].shape) == (9161,) and bounded[name].dtype == torch.float32
                and bool(torch.isfinite(bounded[name]).all()) and not bool(bounded[name].count_nonzero())
                for name in EXTRA_PARAMETERS), "Expected two zero-initialized 9,161-element gain vectors")
    require(fixed_logits.shape == bounded_logits.shape and bool(torch.isfinite(fixed_logits).all())
            and torch.equal(fixed_logits, bounded_logits), "E/F initial CPU logits differ")
    return {"passed": True, "shared_parameter_hashes_equal": True,
            "shared_parameter_hashes": trainer.parameter_audit(fixed),
            "extra_parameters": sorted(EXTRA_PARAMETERS), "extra_parameter_count": 18322,
            "zero_adaptive_parameters": True, "logits_equal": True, "device": "cpu"}


def verify_gradients(record, tensors, plasticity):
    import torch
    expected = COMMON_PARAMETERS | (EXTRA_PARAMETERS if plasticity == "bounded10" else set())
    require(set(tensors["gradients"]) == expected, "Missing or extra decoder-case gradient names")
    require(record["parameter_count"] == TOTALS[plasticity], "Unexpected rank-128 parameter count")
    require(record.get("passed") is True, "Base forward/backward audit failed")
    checks = {}
    for name, value in tensors["gradients"].items():
        nonzero = int(value.count_nonzero())
        finite = bool(torch.isfinite(value).all())
        require(finite and nonzero > 0, "Required gradient is zero or nonfinite: " + name)
        checks[name] = {"finite": finite, "nonzero_values": nonzero, "norm": float(value.norm())}
    record["required_gradients"] = checks
    record["all_required_gradients_nonzero"] = True
    return record


def required_gates_pass(result):
    cases = {name + "_" + device for name in ARMS for device in ("cpu", "mps")}
    comparisons = {name + "_cpu_vs_mps" for name in ARMS}
    return (set(result.get("cases", {})) == cases
            and set(result.get("comparisons", {})) == comparisons
            and all(case.get("passed") is True and case.get("all_required_gradients_nonzero") is True
                    and case.get("targets") == 64 for case in result["cases"].values())
            and all(value.get("passed") is True for value in result["comparisons"].values())
            and result.get("initialization_equivalence", {}).get("passed") is True
            and result.get("exact_ir", {}).get("passed") is True
            and result.get("sources_unchanged") is True
            and result.get("connectorch_unchanged") is True
            and result.get("inputs_unchanged") is True
            and result.get("final_host_guard", {}).get("passed") is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output; existing audit receipts are never overwritten")
    result = {"format_version": 1, "passed": False, "started_at": datetime.now(timezone.utc).isoformat(),
              "training_run": False, "optimizer_steps": 0, "reserved_test_evaluated": False,
              "scope": "Matched rank128 E/F initialization and four full-graph forward/backward checks on macm3; no convergence claim.",
              "criteria": {**legacy.CRITERIA, "both_readout_factor_gradients_nonzero": True,
                           "shared_initial_parameters_and_cpu_logits_exact": True},
              "arms": {name: dict(config) for name, config in CONFIGS.items()}, "cases": {}, "comparisons": {}}
    try:
        result["host_guard"] = host_guard()
        import numpy as np
        import torch
        import train_connectorch as trainer
        from transformers import AutoModelForCausalLM
        from fly_wordbrain.connectorch_backend import exact_connectome_ir
        torch.set_num_threads(4)
        require(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "Native MPS shader runtime required")
        before_sources = source_receipt()
        result.update(source_sha256=before_sources, torch=torch.__version__, python=sys.version,
                      platform=platform.platform(), pytorch_enable_mps_fallback=os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                      connectorch=trainer.connectorch_receipt())
        result["reference_files"] = trainer.verify_reference(args.model)
        dataset_hash, groups_hash = legacy.file_hash(args.data), legacy.file_hash(args.groups)
        require(dataset_hash == legacy.DATA_SHA256, "Dataset differs from pinned A/B reconstruction")
        groups_manifest = json.loads(args.groups.with_name("manifest.json").read_text())
        require(groups_hash == groups_manifest["groups_npz_sha256"], "Grouping archive differs from manifest")
        node_types, metadata = trainer.load_groups(args.groups)
        require({k: v for k, v in groups_manifest.items() if k != "groups_npz_sha256"} == metadata,
                "Grouping metadata and manifest differ")
        with np.load(args.groups, allow_pickle=False) as archive:
            node_ids, node_cell_types = np.asarray(archive["node_ids"]), np.asarray(archive["node_cell_type"])
        result.update(data_file_sha256=dataset_hash, groups_file_sha256=groups_hash, group_metadata=metadata,
                      model_directory=str(args.model.resolve()), data_file=str(args.data.resolve()), groups_file=str(args.groups.resolve()))
        dataset = json.loads(args.data.read_text())
        rows = dataset["train"][:2]
        require(len(rows) == 2 and all(len(row["ids"]) >= 33 for row in rows), "Need two complete training stories with >=33 IDs")
        inputs, targets, mask = trainer.batch_tensors(rows, "cpu")
        inputs, targets, mask = (value[:, :32].contiguous() for value in (inputs, targets, mask))
        require(int(targets.ne(0).sum()) == 64, "Preflight must cover 64 real next-token targets")
        result["training_story_ids"] = [row["id"] for row in rows]
        reference = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True,
                                                        torch_dtype=torch.float32).cpu()
        require(reference.config.n_neurons == 49393 and reference.config.n_edges == 9050172
                and reference.config.n_out == 49393 and reference.config.vocab_size == 1024, "Unexpected graph/readout dimensions")
        require(trainer.frozen_hashes(reference) == metadata["frozen_buffers_sha256"], "Groups bind a different canonical graph")
        ir = exact_connectome_ir(dict(reference.named_buffers()), node_ids, node_cell_types)
        result["exact_ir"] = {"passed": True, "nodes": ir.num_nodes, "edges": ir.num_edges,
                              "all_node_ids_endpoints_and_weights_exact": True, "normalization": "none; released checkpoint values"}
        del ir
        from types import SimpleNamespace
        models = {name: trainer.build_model(reference, SimpleNamespace(**config), node_types) for name, config in CONFIGS.items()}
        initial_logits, initial_parameters = {}, {}
        for name, model in models.items():
            require(tuple(model.lm_head[0].weight.shape) == (128, 49393)
                    and tuple(model.lm_head[1].weight.shape) == (1024, 128), "Wrong rank-128 readout shapes")
            config = SimpleNamespace(**CONFIGS[name])
            counts = trainer.parameter_counts(model)
            require(counts == trainer.expected_parameter_counts(model, config, node_types)
                    and counts["total"] == TOTALS[ARMS[name]], "Wrong rank-128 parameter partition")
            result["arms"][name]["parameter_counts"] = counts
            initial_parameters[name] = trainer.parameters_cpu(model)
            model.brain.sparse_runtime()
            # F lazily builds version-guarded edge-target buffers on its first
            # forward; no_grad retains version counters for those tensors.
            with torch.no_grad():
                initial_logits[name] = model(input_ids=inputs, attention_mask=mask, use_cache=True).logits.clone()
        result["initialization_equivalence"] = verify_initialization(initial_parameters["E32rank128fixed"],
            initial_parameters["F32rank128bounded"], initial_logits["E32rank128fixed"], initial_logits["F32rank128bounded"])
        del initial_logits, initial_parameters
        for name, plasticity in ARMS.items():
            host_guard()
            cpu = models.pop(name)
            cpu_case = legacy.run_pass(cpu, inputs, targets, mask, "cpu")
            result["cases"][name + "_cpu"] = verify_gradients(*cpu_case, plasticity)
            del cpu
            gc.collect()
            mps = trainer.build_model(reference, SimpleNamespace(**CONFIGS[name]), node_types).to("mps")
            mps_case = legacy.run_pass(mps, inputs, targets, mask, "mps")
            result["cases"][name + "_mps"] = verify_gradients(*mps_case, plasticity)
            comparison = legacy.compare(cpu_case, mps_case)
            result["comparisons"][name + "_cpu_vs_mps"] = comparison
            require(comparison["passed"], "Rank128 CPU/MPS parity failed: " + name)
            print(json.dumps({"case": name, "passed": True, "loss_abs_difference": comparison["loss_abs_difference"],
                              "max_gradient_relative_l2": comparison["max_gradient_relative_l2"]}), flush=True)
            del cpu_case, mps_case, mps
            gc.collect()
            torch.mps.empty_cache()
        result["final_host_guard"] = host_guard()
        result["sources_unchanged"] = before_sources == source_receipt()
        result["connectorch_unchanged"] = result["connectorch"] == trainer.connectorch_receipt()
        result["inputs_unchanged"] = (dataset_hash == legacy.file_hash(args.data) and groups_hash == legacy.file_hash(args.groups)
                                      and result["reference_files"] == trainer.verify_reference(args.model)
                                      and groups_manifest == json.loads(args.groups.with_name("manifest.json").read_text()))
        result["passed"] = required_gates_pass(result)
        require(result["passed"], "Incomplete decoder preflight or changed source/input bindings")
    except Exception as error:
        result["passed"] = False
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "failure": result.get("failure")}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
