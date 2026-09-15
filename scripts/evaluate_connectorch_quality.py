#!/usr/bin/env python3
"""Replay both retained A/B selectors and compare six fixed texts on idle macm3.

Inference only: no optimizer, checkpoint writes, or reserved-test evaluation.
The local trainer's checkpoints contain Python RNG state and are loaded with
weights_only=False only after their local selector receipts are verified.
"""
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import torch.nn.functional as F
import train_connectorch as trainer
from evaluate_ngxson_quality import PROMPTS, aggregate

SELECTORS = {"minimum_validation_ce": "best.pt", "maximum_validation_accuracy": "best-accuracy.pt"}
HOST_NAMES = {"macm3", "fernandos-macbook-pro-2"}
DISPATCHERS = {"run_connectorch_campaign.py", "continue_connectorch_campaign.py"}


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_processes(text):
    records = []
    for line in text.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            raise ValueError("Unparseable process inventory")
        pid, parent, state, command = fields
        records.append({"pid": int(pid), "ppid": int(parent), "state": state, "command": command})
    return records


def process_guard(records, worker_pids=(), own_pid=None):
    """Zombies have exited; even SIGSTOP trainers still own GPU resources."""
    blocked, zombies, paused_dispatchers = [], [], []
    for record in records:
        if record["pid"] == own_pid:
            continue
        if "Z" in record["state"]:
            if record["pid"] in worker_pids:
                zombies.append(record)
            continue
        try:
            tokens = shlex.split(record["command"])
        except ValueError:
            tokens = record["command"].split()
        names = {Path(token).name for token in tokens}
        executable = Path(tokens[0]).name.lower() if tokens else ""
        python_worker = executable.startswith("python") or executable.endswith(".py")
        training = python_worker and any(name.startswith("train_") and name.endswith(".py") for name in names)
        inference = python_worker and any(name.startswith(("evaluate_", "compare_ngxson_texts")) and name.endswith(".py") for name in names)
        dispatcher = python_worker and bool(names & DISPATCHERS)
        if record["pid"] in worker_pids or training or inference or (dispatcher and "T" not in record["state"]):
            blocked.append(record)
        elif dispatcher:
            paused_dispatchers.append(record)
    require(not blocked, "GPU audit requires exited training workers and held dispatchers: " + json.dumps(blocked))
    return {"training_workers_exited": True, "exited_zombies": zombies,
            "paused_dispatchers": paused_dispatchers, "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def idle_host_guard(worker_pids):
    hostname = socket.gethostname()
    require(platform.system() == "Darwin" and hostname.lower().removesuffix(".local") in HOST_NAMES,
            "Real quality inference is restricted to the registered macm3 host")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(chip.startswith("Apple M3"), "Expected an Apple M3 GPU host")
    require(torch.backends.mps.is_available(), "MPS is unavailable")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "Disable MPS CPU fallback for this audit")
    processes = parse_processes(subprocess.check_output(["ps", "-axo", "pid=,ppid=,state=,command="], text=True))
    return {"hostname": hostname, "chip": chip, "device": "mps",
            **process_guard(processes, worker_pids, os.getpid())}


def verify_bound(path, digest, bindings):
    path = Path(path).resolve()
    require(path.is_file() and trainer.file_hash(path) == digest, "Bound file missing or changed: " + str(path))
    bindings[str(path)] = digest
    return path


def validation_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    records = [row for row in rows if row.get("event") == "validation"]
    require(bool(records), "No validation records")
    for row in records:
        metric = row["validation"]
        require(metric["tokens"] > 0 and metric["stories"] > 0 and 0 <= metric["correct"] <= metric["tokens"],
                "Invalid logged validation counts")
        require(math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0,
                "Invalid logged validation CE")
        require(abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) < 1e-12,
                "Logged accuracy differs from raw counts")
    return records


def verify_selectors(directory, manifest, records, bindings):
    receipt = read_json(directory / "selected-checkpoints.json")
    require(receipt.get("format_version") == 1 and set(receipt.get("selectors", {})) == set(SELECTORS),
            "Both native checkpoint selectors are required")
    for name, entry in receipt["selectors"].items():
        choose, metric = (min, "cross_entropy") if name == "minimum_validation_ce" else (max, "top1_accuracy")
        row = choose(records, key=lambda value: value["validation"][metric])
        path = directory / SELECTORS[name]
        require(Path(entry["path"]).resolve() == path.resolve(), "Selector points outside its arm checkpoint")
        verify_bound(path, entry["sha256"], bindings)
        require(all(entry["cursor"][key] == row[key] for key in ("updates", "epoch")), "Selector cursor differs from logged winner")
        require(entry["validation"] == row["validation"], "Selector scores differ from logged winner")
        require(entry.get("frozen_buffers_preserved") is True
                and entry["frozen_buffers_sha256"] == manifest["frozen_buffers_sha256"], "Selector graph identity differs")
    return receipt["selectors"]


def inspect_arm(directory, width):
    directory = directory.resolve()
    bindings = {}
    for name in ("manifest.json", "selected-checkpoints.json", "metrics.jsonl", "launch.json", "status.json", "process-status.json"):
        path = directory / name
        verify_bound(path, trainer.file_hash(path), bindings)
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("reference_repository") == "ngxson/fly-llm-hf" and manifest.get("reference_revision") == trainer.REVISION,
            "Arm uses a different reference")
    cfg = manifest["config"]
    expected = {"d_embed": width, "plasticity": "fixed", "readout_rank": 0, "history_length": 8,
                "device": "mps", "skip_final_test": True, "max_updates": None, "eval_limit": None}
    require(all(cfg.get(key) == value for key, value in expected.items()) and manifest.get("debug") is False,
            "Arm is not the declared full-data fixed-edge A/B architecture")
    require(trainer.source_receipt() == manifest["trainer_sources_sha256"], "Current trainer/model sources differ from the saved arm")
    for name, digest in manifest["trainer_sources_sha256"].items():
        verify_bound(ROOT / name, digest, bindings)
    require(trainer.connectorch_receipt() == manifest["connectorch_source"], "Installed ConnecTorch source differs")
    model_dir = Path(manifest["model_path"])
    require(trainer.verify_reference(model_dir) == manifest["reference_files"], "Pinned reference receipt differs")
    for name, receipt in manifest["reference_files"].items():
        verify_bound(model_dir / name, receipt["sha256"], bindings)
    for name, digest in manifest["source_sha256"].items():
        verify_bound(ROOT / "scripts/train_connectorch.py" if name == "trainer" else model_dir / name, digest, bindings)
    data_path = verify_bound(manifest["data_path"], manifest["data_sha256"], bindings)
    groups_path = verify_bound(manifest["groups_path"], manifest["groups_sha256"], bindings)
    data = read_json(data_path)
    for split in ("train", "validation"):
        require(len(data[split]) == manifest["split_sizes"][split], "Training/validation story count differs")
    groups, metadata = trainer.load_groups(groups_path)
    require(metadata == manifest["groups_provenance"] and int(groups.max()) + 1 == manifest["node_types"],
            "Group provenance or group count differs")
    records = validation_records(directory / "metrics.jsonl")
    target_count = sum(len(row["ids"]) - 1 for row in data["validation"])
    require(all(row["validation"]["stories"] == len(data["validation"]) and row["validation"]["tokens"] == target_count for row in records),
            "Logged validation coverage differs from the complete validation split")
    selectors = verify_selectors(directory, manifest, records, bindings)
    dispositions = {}
    for name in ("accepted-early-stop.json", "stop-receipt.json", "results.json"):
        path = directory / name
        if path.exists():
            verify_bound(path, trainer.file_hash(path), bindings)
            dispositions[name] = read_json(path)
    return {"directory": directory, "manifest": manifest, "bindings": bindings, "data": data,
            "groups": groups, "records": records, "selectors": selectors, "dispositions": dispositions,
            "launch": read_json(directory / "launch.json")}


def verify_checkpoint(saved, arm, selector):
    entry = arm["selectors"][selector]
    require(saved.get("format_version") == 1 and saved["manifest"] == arm["manifest"], "Checkpoint manifest differs from arm")
    require(saved["cursor"] == entry["cursor"], "Checkpoint cursor differs from selector receipt")
    require(saved["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"], "Checkpoint graph identity differs")
    require(saved["model_audit"]["frozen_buffers_preserved"] is True
            and saved["model_audit"]["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"], "Saved model graph audit differs")
    best = saved["best" if selector == "minimum_validation_ce" else "best_accuracy"]
    require(all(best[key] == entry["cursor"][key] for key in ("updates", "epoch"))
            and all(best[key] == entry["validation"][key] for key in ("cross_entropy", "top1_accuracy")),
            "Checkpoint does not contain its declared winner")
    audit = trainer.parameter_audit(saved["parameters"])
    require(set(audit) == set(saved["parameter_audit"]), "Saved parameter audit names differ")
    require(all(all(info[key] == saved["parameter_audit"][name][key] for key in ("sha256", "values", "finite")) for name, info in audit.items()),
            "Saved parameter contents differ from their hashes")
    require(all(value.dtype == torch.float32 for value in saved["parameters"].values()), "Checkpoint parameters must be float32")
    return audit


def restore_model(reference, arm, saved):
    config = SimpleNamespace(**arm["manifest"]["config"])
    reference_hashes = trainer.frozen_hashes(reference)
    require(arm["manifest"]["groups_provenance"]["frozen_buffers_sha256"] == reference_hashes, "Groups reference a different graph")
    require(arm["groups"].numel() == reference.config.n_neurons, "Grouping neuron count differs")
    model = trainer.build_model(reference, config, arm["groups"])
    counts = trainer.parameter_counts(model)
    require(counts == trainer.expected_parameter_counts(model, config, arm["groups"])
            and counts == arm["manifest"]["parameter_counts"], "Restored architecture parameter count differs")
    require(trainer.frozen_hashes(model) == arm["manifest"]["frozen_buffers_sha256"], "Restored graph/group hashes differ")
    trainer.restore_parameters(model, saved["parameters"])
    require(trainer.parameter_audit(trainer.parameters_cpu(model)) == trainer.parameter_audit(saved["parameters"]),
            "Restored parameter hashes differ")
    return model


def validate_tokenized_rows(rows, tokenizer):
    seen = set()
    for row in rows:
        require(row["id"] not in seen, "Repeated story ID")
        seen.add(row["id"])
        ids = [tokenizer.bos_token_id, *tokenizer.encode(row["text"], add_special_tokens=False), tokenizer.eos_token_id]
        require(ids == row["ids"] and 2 <= len(ids) <= 320, "Story tokenization or length differs")
        require(hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"], "Story text hash differs")
        require(hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).hexdigest() == row["normalized_sha256"],
                "Normalized story hash differs")


@torch.inference_mode()
def score_validation(model, rows, device="mps", batch_size=8, chunk_size=32):
    records = []
    pad = model.config.pad_token_id
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        inputs, targets, mask = trainer.batch_tensors(group, device, pad)
        cache = None
        totals = torch.zeros(len(group), dtype=torch.float64)
        counts = torch.zeros(len(group), dtype=torch.long)
        hits = torch.zeros_like(counts)
        for offset in range(0, inputs.shape[1], chunk_size):
            sl = slice(offset, offset + chunk_size)
            out = model(input_ids=inputs[:, sl], attention_mask=mask[:, sl], cache_params=cache, use_cache=True, return_dict=True)
            cache = out.cache_params
            require(bool(torch.isfinite(out.logits).all()), "Nonfinite validation logits")
            target = targets[:, sl]
            valid = target.ne(pad)
            losses = F.cross_entropy(out.logits.flatten(0, 1), target.flatten(), ignore_index=pad, reduction="none").reshape_as(target)
            totals += losses.cpu().double().sum(dim=1)
            counts += valid.sum(dim=1).cpu()
            hits += ((out.logits.argmax(-1) == target) & valid).sum(dim=1).cpu()
        for i, row in enumerate(group):
            require(int(counts[i]) == len(row["ids"]) - 1, "Validation lost next-token targets")
            records.append({"id": row["id"], "tokens": int(counts[i]), "correct": int(hits[i]), "nll_sum": float(totals[i])})
    return {"summary": aggregate(records), "per_story": records}


def verify_replay(replayed, logged):
    require(all(replayed[key] == logged[key] for key in ("stories", "tokens", "correct")), "Validation replay count/accuracy mismatch")
    delta = replayed["cross_entropy"] - logged["cross_entropy"]
    require(abs(delta) < 1e-4, "Validation CE replay mismatch: " + str(delta))
    return {"passed": True, "cross_entropy_delta": delta, "ce_tolerance": 1e-4, "counts_match_exactly": True}


@torch.inference_mode()
def generate(model, tokenizer, device="mps", prompts=PROMPTS, max_new_tokens=80):
    require(max_new_tokens > 0, "Generation token budget must be positive")
    samples = []
    for prompt in prompts:
        prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
        tokens, new = list(prompt_ids), []
        cache = None
        incoming = prompt_ids
        for _ in range(max_new_tokens):
            ids = torch.tensor([incoming], dtype=torch.long, device=device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), cache_params=cache, use_cache=True, return_dict=True)
            require(bool(torch.isfinite(out.logits).all()), "Nonfinite generation logits")
            token = int(out.logits[0, -1].argmax().item())
            new.append(token)
            tokens.append(token)
            if token == tokenizer.eos_token_id:
                break
            cache, incoming = out.cache_params, [token]
        samples.append({"prompt": prompt, "prompt_ids": prompt_ids, "all_ids": tokens, "new_ids": new,
                        "new_tokens": len(new), "stopped_on_eos": new[-1] == tokenizer.eos_token_id,
                        "text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "continuation": tokenizer.decode(new, skip_special_tokens=True)})
    return samples


def training_overlap(samples, train_rows):
    corpus = [(row["id"], row["text"].casefold().split()) for row in train_rows]
    results = []
    for sample in samples:
        words = sample["continuation"].casefold().split()
        best = {"words": 0, "training_story_id": None, "matched_span": ""}
        for story_id, story in corpus:
            match = SequenceMatcher(None, words, story, autojunk=False).find_longest_match()
            if match.size > best["words"]:
                best = {"words": match.size, "training_story_id": story_id,
                        "continuation_word_offset_zero_based": match.a, "training_word_offset_zero_based": match.b,
                        "matched_span": " ".join(words[match.a:match.a + match.size])}
        full = " ".join(sample["text"].casefold().split())
        prefixes = [story_id for story_id, words_in_story in corpus if full and " ".join(words_in_story).startswith(full)]
        results.append({"prompt": sample["prompt"], "continuation_words": len(words), "longest_contiguous_match": best,
                        "full_prompt_plus_continuation_is_normalized_training_prefix": bool(prefixes),
                        "exact_prefix_training_story_ids": prefixes})
    return results


def compare_scores(a, b):
    ce = b["cross_entropy"] - a["cross_entropy"]
    accuracy = b["top1_accuracy"] - a["top1_accuracy"]
    return {"direction": "B minus A", "delta_cross_entropy": ce, "delta_accuracy": accuracy,
            "ce_loss_over_0_10": ce > .10, "accuracy_loss_over_1pp": accuracy < -.01,
            "review_flag": ce > .10 or accuracy < -.01,
            "qualification": "Descriptive review flags on checkpoint-selection validation; not statistical significance or test equivalence."}


def report_markdown(result):
    lines = ["# Reduced-encoder quality assessment: A versus B", "",
             "Inference only on macm3. Both retained selectors replay the complete validation set; the reserved test is untouched.",
             "Retained winners may have unequal training budgets. Common-update rows below compare logged scores, not unavailable historical checkpoint weights.", "",
             "| Arm | Selector | Update | CE | Accuracy | Replay |", "|---|---|---:|---:|---:|---|"]
    for label, selectors in result["arms"].items():
        for selector, value in selectors.items():
            metric = value["validation"]["summary"]
            lines.append(f"| {label} | {selector} | {value['cursor']['updates']:,} | {metric['cross_entropy']:.5f} | {metric['top1_accuracy']:.3%} | passed |")
    lines.extend(["", "## B minus A at retained winners", ""])
    for selector, value in result["comparisons"].items():
        lines.append(f"- {selector}: ΔCE {value['delta_cross_entropy']:+.5f}; accuracy {100 * value['delta_accuracy']:+.3f} pp; practical review flag: {value['review_flag']}.")
    lines.extend(["", "The review flags are CE loss greater than 0.10 or accuracy loss greater than 1 pp. These are practical flags, not significance tests. Text coherence still requires inspection.", "",
                  "## All fixed greedy continuations", "", "Same six prompts, fresh cache and one BOS per prompt, greedy decoding, EOS stopping and at most 80 new BPE tokens. Outputs are unedited.", ""])
    for i, prompt in enumerate(PROMPTS):
        lines.extend([f"### {i + 1}. {prompt}", ""])
        for label, selectors in result["arms"].items():
            for selector, value in selectors.items():
                overlap = value["training_overlap"][i]
                lines.extend([f"**{label} / {selector}**", "", value["samples"][i]["continuation"], "",
                              f"Longest contiguous training overlap: {overlap['longest_contiguous_match']['words']} whitespace words; "
                              f"full generated text is a normalized training prefix: {overlap['full_prompt_plus_continuation_is_normalized_training_prefix']}.", ""])
    lines.extend(["## Common logged updates", "", "| Update | A CE | B CE | ΔCE | A accuracy | B accuracy | Δaccuracy pp |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result["matched_update_comparison"]:
        a, b, delta = row["A"], row["B"], row["comparison"]
        lines.append(f"| {row['updates']:,} | {a['cross_entropy']:.5f} | {b['cross_entropy']:.5f} | {delta['delta_cross_entropy']:+.5f} | {a['top1_accuracy']:.3%} | {b['top1_accuracy']:.3%} | {100 * delta['delta_accuracy']:+.3f} |")
    lines.extend(["", "All parameter/graph hashes stayed unchanged, saved validation counts replayed exactly, and native backward-kernel counters remained zero.",
                  "Training overlap uses only train rows, casefolded whitespace words and punctuation retained. Short shared phrases alone do not establish memorization. Six fixed greedy samples are descriptive.", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-a", type=Path, required=True)
    p.add_argument("--arm-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("mps",), default="mps")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    require(args.threads > 0, "Threads must be positive")
    worker_pids = set()
    for directory in (args.arm_a, args.arm_b):
        for name, keys in (("status.json", ("pid",)), ("process-status.json", ("worker_pid",))):
            item = read_json(directory / name)
            worker_pids.update(item[key] for key in keys if isinstance(item.get(key), int))
    host = idle_host_guard(worker_pids)
    require(not args.output.exists(), "Use a new audit output directory")
    torch.set_num_threads(args.threads)
    arms = {"A128fixed": inspect_arm(args.arm_a, 128), "B32fixed": inspect_arm(args.arm_b, 32)}
    a, b = arms.values()
    for key in ("data_sha256", "groups_sha256", "reference_files", "frozen_buffers_sha256", "trainer_sources_sha256", "connectorch_source"):
        require(a["manifest"][key] == b["manifest"][key], "A/B comparison identity differs: " + key)
    require({k: v for k, v in a["manifest"]["config"].items() if k != "d_embed"}
            == {k: v for k, v in b["manifest"]["config"].items() if k != "d_embed"}, "A/B recipe differs beyond encoder width")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a["manifest"]["model_path"], trust_remote_code=True, local_files_only=True)
    require((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2), "Pinned special-token IDs differ")
    validate_tokenized_rows(a["data"]["train"] + a["data"]["validation"], tokenizer)
    bindings = {key: value for arm in arms.values() for key, value in arm["bindings"].items()}
    for name in ("scripts/evaluate_connectorch_quality.py", "scripts/evaluate_ngxson_quality.py", "scripts/train_ngxson.py"):
        verify_bound(ROOT / name, trainer.file_hash(ROOT / name), bindings)
    args.output.mkdir(parents=True)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "training_run": False, "host_guard": host,
                "device": "mps", "torch": torch.__version__, "threads": args.threads,
                "reserved_test_evaluated": False, "validation_used_for_checkpoint_selection": True,
                "checkpoint_load": "Trusted local trainer pickle, SHA256 bound to native selector receipt",
                "generation": {"prompts": PROMPTS, "max_new_tokens": 80, "do_sample": False, "bos_once": True, "fresh_cache_per_prompt": True, "eos_stopping": True},
                "bindings_sha256": bindings,
                "arms": {name: {"directory": str(arm["directory"]), "config": arm["manifest"]["config"], "selectors": arm["selectors"],
                                  "dispositions": arm["dispositions"], "experiment_git": arm["launch"].get("experiment_git")} for name, arm in arms.items()}}
    trainer.atomic_json(args.output / "protocol.json", protocol)
    outputs = {}
    for label, arm in arms.items():
        outputs[label] = {}
        for selector, entry in arm["selectors"].items():
            idle_host_guard(worker_pids)
            checkpoint = Path(entry["path"])
            verify_bound(checkpoint, entry["sha256"], bindings)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            verify_checkpoint(saved, arm, selector)
            reference = AutoModelForCausalLM.from_pretrained(arm["manifest"]["model_path"], trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32).cpu()
            model = restore_model(reference, arm, saved)
            del reference
            before = trainer.parameter_audit(trainer.parameters_cpu(model))
            graph_before = trainer.frozen_hashes(model)
            cursor = dict(saved["cursor"])
            del saved
            model = model.to("mps").eval().requires_grad_(False)
            # Build version-guarded sparse buffers outside inference_mode.
            model.brain.sparse_runtime()
            model.verify_frozen()
            validation = score_validation(model, arm["data"]["validation"], batch_size=arm["manifest"]["config"]["batch_size"], chunk_size=arm["manifest"]["config"]["chunk_size"])
            replay = verify_replay(validation["summary"], entry["validation"])
            samples = generate(model, tokenizer)
            torch.mps.synchronize()
            backend = model.backend_metadata()
            counts = backend["sparse_backend"]["kernel_launch_count"]
            require(counts["forward"] > 0 and counts["state_backward"] == counts["edge_backward"] == 0, "Expected forward-only native Metal execution")
            require(backend["sparse_backend"]["host_fallback"] is False, "Sparse backend used host fallback")
            after = trainer.parameter_audit(trainer.parameters_cpu(model))
            graph_after = trainer.frozen_hashes(model)
            require(before == after and graph_before == graph_after, "Evaluation mutated model parameters or graph")
            require(all(not value.requires_grad and value.grad is None for value in model.parameters()), "Evaluation created parameter gradients")
            model.verify_frozen()
            outputs[label][selector] = {"cursor": cursor, "checkpoint_sha256": entry["sha256"], "validation": validation,
                "replay": replay, "samples": samples, "training_overlap": training_overlap(samples, arm["data"]["train"]),
                "parameter_hashes_before": before, "parameter_hashes_after": after,
                "graph_hashes_before": graph_before, "graph_hashes_after": graph_after,
                "parameters_and_graph_unchanged": True, "zero_backward_kernel_calls": True, "backend": backend}
            trainer.atomic_json(args.output / (label + ".json"), outputs[label])
            trainer.atomic_json(args.output / "status.json", {"status": "running", "completed": label + "/" + selector})
            del model
            gc.collect()
            torch.mps.empty_cache()
    final_host = idle_host_guard(worker_pids)
    for path, digest in bindings.items():
        verify_bound(path, digest, {})
    require(trainer.connectorch_receipt() == a["manifest"]["connectorch_source"], "Installed ConnecTorch changed during audit")
    comparisons = {selector: compare_scores(outputs["A128fixed"][selector]["validation"]["summary"], outputs["B32fixed"][selector]["validation"]["summary"]) for selector in SELECTORS}
    history_a = {row["updates"]: row["validation"] for row in a["records"]}
    history_b = {row["updates"]: row["validation"] for row in b["records"]}
    common = [{"updates": step, "A": history_a[step], "B": history_b[step], "comparison": compare_scores(history_a[step], history_b[step])} for step in sorted(history_a.keys() & history_b.keys())]
    result = {"status": "completed", "protocol": protocol, "arms": outputs, "comparisons": comparisons,
              "matched_update_comparison": common, "final_host_guard": final_host, "reserved_test_evaluated": False,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    trainer.atomic_json(args.output / "results.json", result)
    (args.output / "report.md").write_text(report_markdown(result))
    trainer.atomic_json(args.output / "status.json", {"status": "completed", "comparisons": comparisons})
    print(json.dumps({"completed": True, "output": str(args.output), "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
