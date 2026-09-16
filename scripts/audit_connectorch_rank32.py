#!/usr/bin/env python3
"""Full-graph rank32 fixed-edge CPU/MPS gradient preflight on idle macm3.

One forward/backward per device, zero optimizer steps, no test-set access.
All existing experiment, model and trainer sources remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_connectorch as legacy
import audit_connectorch_decoder as shared

ARM = "H32rank32fixed"
CONFIG = {"d_embed": 32, "readout_rank": 32, "history_length": 8,
          "lag_map": list(range(8)), "seed": 42, "plasticity": "fixed"}
COUNTS = {"embedding": 32768, "input_projection": 450048, "neurons": 148179,
          "edge_gains": 0, "layernorm": 98786, "readout": 1613344, "other": 0, "total": 2343125}
READOUT_SHAPES = {"lm_head.0.weight": [32, 49393], "lm_head.1.weight": [1024, 32]}
require = shared.require


def source_receipt():
    return {**shared.source_receipt(), "scripts/audit_connectorch_rank32.py": legacy.file_hash(Path(__file__))}


def verify_initialization(cpu_parameters, mps_parameters):
    import torch
    import train_connectorch as trainer
    require(set(cpu_parameters) == set(mps_parameters) == shared.COMMON_PARAMETERS,
            "Rank32 initialization parameter names differ")
    for name, expected_shape in READOUT_SHAPES.items():
        require(list(cpu_parameters[name].shape) == expected_shape, "Rank32 factor shape differs: " + name)
    require(all(value.dtype == torch.float32 and value.shape == mps_parameters[name].shape
                and value.dtype == mps_parameters[name].dtype and torch.equal(value, mps_parameters[name])
                for name, value in cpu_parameters.items()), "CPU/MPS initial parameter contents differ")
    hashes = trainer.parameter_audit(cpu_parameters)
    require(hashes == trainer.parameter_audit(mps_parameters), "CPU/MPS initial parameter hashes differ")
    require(sum(value.numel() for value in cpu_parameters.values()) == COUNTS["total"], "Rank32 initial parameter count differs")
    return {"passed": True, "cpu_mps_parameter_hashes_equal": True,
            "parameter_hashes": hashes, "readout_shapes": READOUT_SHAPES,
            "qualification": "Seed42 from scratch; rank32 second-factor std remains0.02, so initial logit scale differs from rank64."}


def verify_gradients(record, tensors):
    import torch
    gradients = tensors["gradients"]
    require(set(gradients) == shared.COMMON_PARAMETERS, "Missing or extra rank32 gradient names")
    require(record.get("passed") is True and record["parameter_count"] == COUNTS["total"],
            "Rank32 base audit or parameter count failed")
    checks = {}
    for name, value in gradients.items():
        if name in READOUT_SHAPES:
            require(list(value.shape) == READOUT_SHAPES[name], "Rank32 factor gradient shape differs: " + name)
        finite, nonzero = bool(torch.isfinite(value).all()), int(value.count_nonzero())
        require(finite and nonzero > 0, "Rank32 required gradient is zero or nonfinite: " + name)
        checks[name] = {"finite": finite, "nonzero_values": nonzero, "norm": float(value.norm())}
    record.update(required_gradients=checks, all_required_gradients_nonzero=True)
    return record


def required_gates_pass(result):
    return (set(result.get("cases", {})) == {ARM + "_cpu", ARM + "_mps"}
            and set(result.get("comparisons", {})) == {ARM + "_cpu_vs_mps"}
            and all(case.get("passed") is True and case.get("all_required_gradients_nonzero") is True
                    and case.get("targets") == 64 for case in result["cases"].values())
            and result["comparisons"][ARM + "_cpu_vs_mps"].get("passed") is True
            and result.get("initialization", {}).get("passed") is True
            and result.get("exact_ir", {}).get("passed") is True
            and result.get("sources_unchanged") is True and result.get("inputs_unchanged") is True
            and result.get("connectorch_unchanged") is True
            and result.get("final_host_guard", {}).get("passed") is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "groups", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output; existing preflight receipts are never overwritten")
    result = {"format_version": 1, "passed": False, "started_at": datetime.now(timezone.utc).isoformat(),
              "training_run": False, "optimizer_steps": 0, "reserved_test_evaluated": False,
              "scope": "H32rank32fixed full-graph CPU/MPS forward/backward parity; no optimizer, preserved G is the comparison baseline.",
              "arms": {ARM: {**CONFIG, "parameter_counts": COUNTS}},
              "criteria": {**legacy.CRITERIA, "both_factorized_gain_gradients_nonzero": False,
                           "both_readout_factor_gradients_nonzero": True, "initial_parameters_exact": True},
              "cases": {}, "comparisons": {}}
    try:
        result["host_guard"] = shared.host_guard()
        import numpy as np
        import torch
        import train_connectorch as trainer
        from transformers import AutoModelForCausalLM
        from fly_wordbrain.connectorch_backend import exact_connectome_ir
        torch.set_num_threads(4)
        require(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "Native MPS shaders required")
        before_sources = source_receipt()
        result.update(source_sha256=before_sources, torch=torch.__version__, python=sys.version,
                      platform=platform.platform(), pytorch_enable_mps_fallback=os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                      connectorch=trainer.connectorch_receipt())
        result["reference_files"] = trainer.verify_reference(args.model)
        data_hash, groups_hash = legacy.file_hash(args.data), legacy.file_hash(args.groups)
        require(data_hash == legacy.DATA_SHA256, "Dataset differs from pinned reconstruction")
        groups_manifest = json.loads(args.groups.with_name("manifest.json").read_text())
        node_types, metadata = trainer.load_groups(args.groups)
        require(groups_hash == groups_manifest["groups_npz_sha256"]
                and {k: v for k, v in groups_manifest.items() if k != "groups_npz_sha256"} == metadata,
                "Grouping archive/metadata differs from manifest")
        with np.load(args.groups, allow_pickle=False) as archive:
            node_ids, node_cell_types = np.asarray(archive["node_ids"]), np.asarray(archive["node_cell_type"])
        result.update(data_file_sha256=data_hash, groups_file_sha256=groups_hash, group_metadata=metadata,
                      model_directory=str(args.model.resolve()), data_file=str(args.data.resolve()), groups_file=str(args.groups.resolve()))
        rows = json.loads(args.data.read_text())["train"][:2]
        require(len(rows) == 2 and all(len(row["ids"]) >= 33 for row in rows), "Need two real training stories with >=33 IDs")
        inputs, targets, mask = trainer.batch_tensors(rows, "cpu")
        inputs, targets, mask = (value[:, :32].contiguous() for value in (inputs, targets, mask))
        require(int(targets.ne(0).sum()) == 64, "Require all64 real next-token targets")
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
        cpu = trainer.build_model(reference, SimpleNamespace(**CONFIG), node_types)
        require(trainer.parameter_counts(cpu) == COUNTS
                and trainer.expected_parameter_counts(cpu, SimpleNamespace(**CONFIG), node_types) == COUNTS,
                "Rank32 parameter partition differs")
        cpu_initial = trainer.parameters_cpu(cpu)
        cpu_case = legacy.run_pass(cpu, inputs, targets, mask, "cpu")
        result["cases"][ARM + "_cpu"] = verify_gradients(*cpu_case)
        require(trainer.parameter_audit(trainer.parameters_cpu(cpu)) == trainer.parameter_audit(cpu_initial),
                "CPU backward unexpectedly changed parameters")
        del cpu
        gc.collect()
        shared.host_guard()
        mps = trainer.build_model(reference, SimpleNamespace(**CONFIG), node_types).to("mps")
        result["initialization"] = verify_initialization(cpu_initial, trainer.parameters_cpu(mps))
        mps_case = legacy.run_pass(mps, inputs, targets, mask, "mps")
        result["cases"][ARM + "_mps"] = verify_gradients(*mps_case)
        require(trainer.parameter_audit(trainer.parameters_cpu(mps)) == trainer.parameter_audit(cpu_initial),
                "MPS backward unexpectedly changed parameters")
        result["comparisons"][ARM + "_cpu_vs_mps"] = legacy.compare(cpu_case, mps_case)
        del mps, cpu_case, mps_case, cpu_initial
        gc.collect()
        torch.mps.empty_cache()
        result["final_host_guard"] = shared.host_guard()
        result["sources_unchanged"] = before_sources == source_receipt()
        result["connectorch_unchanged"] = result["connectorch"] == trainer.connectorch_receipt()
        result["inputs_unchanged"] = (data_hash == legacy.file_hash(args.data) and groups_hash == legacy.file_hash(args.groups)
                                      and result["reference_files"] == trainer.verify_reference(args.model)
                                      and groups_manifest == json.loads(args.groups.with_name("manifest.json").read_text()))
        result["passed"] = required_gates_pass(result)
        require(result["passed"], "Rank32 numerical parity or binding gate failed")
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
