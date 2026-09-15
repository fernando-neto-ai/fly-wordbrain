#!/usr/bin/env python3
"""Compare all fixed greedy texts from the release and a frozen accuracy winner."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from evaluate_ngxson_quality import PROMPTS, generate, selected_validation
from train_ngxson import (atomic_json, file_hash, frozen_hashes, parameters_cpu,
                          parameter_audit, restore_parameters, verify_reference)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--selection-receipt", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reference_files = verify_reference(args.model)
    digest = file_hash(args.checkpoint)
    selection = json.loads(args.selection_receipt.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    metric = selected_validation(saved, "max-accuracy", selection, digest)
    assert saved["manifest"]["data_sha256"] == file_hash(args.training_data)
    expected_graph = saved["frozen_buffers_sha256"]
    values = saved["parameters"]
    audit = parameter_audit(values)
    assert all(audit[k]["sha256"] == saved["parameter_audit"][k]["sha256"] for k in audit)
    step = saved["cursor"]["updates"]
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": False, "host": socket.gethostname(), "device": args.device,
        "checkpoint_sha256": digest, "checkpoint_updates": step,
        "selected_validation": metric, "selection_receipt": selection,
        "reference_files": reference_files, "prompts": PROMPTS,
        "do_sample": False, "max_new_tokens": 80, "bos_once": True,
        "source_sha256": {name: file_hash(ROOT / name) for name in
            ("scripts/compare_ngxson_texts.py", "scripts/evaluate_ngxson_quality.py",
             "scripts/train_ngxson.py", "fly_wordbrain/ngxson_mps.py")}}
    atomic_json(args.output / "protocol.json", protocol)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    outputs = {}
    for name in ("released", "ours_accuracy"):
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     local_files_only=True).eval()
        assert frozen_hashes(model) == expected_graph
        if name == "ours_accuracy":
            restore_parameters(model, values)
        before = parameter_audit(parameters_cpu(model))
        if name == "ours_accuracy":
            assert all(before[k]["sha256"] == audit[k]["sha256"] for k in audit)
        if args.device == "mps":
            from fly_wordbrain.ngxson_mps import enable_mps
            model = enable_mps(model)
        model.requires_grad_(False)
        outputs[name] = {"samples": generate(model, tokenizer, args.device)}
        assert before == parameter_audit(parameters_cpu(model))
        assert frozen_hashes(model) == expected_graph
        assert all(p.grad is None for p in model.parameters())
        outputs[name]["parameters_and_graph_unchanged"] = True
        if args.device == "mps":
            from fly_wordbrain.ngxson_mps import mps_metadata
            outputs[name]["backend"] = mps_metadata(model, verify=True)
            assert outputs[name]["backend"]["backward_kernel_calls"] == 0
        atomic_json(args.output / (name + ".json"), outputs[name])
        del model
        gc.collect()
    assert file_hash(args.checkpoint) == digest
    report = {"protocol": protocol, "models": outputs}
    atomic_json(args.output / "comparison.json", report)
    lines = ["# Text comparison: highest validation accuracy versus released FlyLLM", "",
             f"Our frozen checkpoint: update {step:,}, validation accuracy {metric['top1_accuracy']:.2%}.", "",
             "Identical six prompts; greedy decoding, BOS once, at most 80 new BPE tokens. "
             "Abrupt endings can reflect the token limit. All outputs are retained, without correcting spelling or grammar.", ""]
    for i, prompt in enumerate(PROMPTS):
        lines.extend([f"## {i+1}. {prompt}", ""])
        for name, label in (("released", "Original"), ("ours_accuracy", "Ours")):
            lines.extend([f"**{label} continuation**", "", outputs[name]["samples"][i]["continuation"], ""])
    (args.output / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({"completed": True, "checkpoint_updates": step,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
