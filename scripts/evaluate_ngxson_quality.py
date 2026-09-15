#!/usr/bin/env python3
"""Inference-only, paired quality audit of a frozen local checkpoint and HF release."""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from train_ngxson import (atomic_json, batch_tensors, file_hash, frozen_hashes,
                          parameter_audit, parameters_cpu, restore_parameters, verify_reference)

PROMPTS = [
    "Once upon a time, there was a",
    "One day, a little girl named Lily found a needle",
    "Tom and his dog",
    "Sara put the red ball inside the box. Later,",
    "The little bird was afraid of the rain. Its mother",
    "Ben wanted to share his cake, but",
]


def aggregate(rows):
    tokens = sum(r["tokens"] for r in rows)
    if tokens <= 0 or not rows:
        raise ValueError("Empty evaluation")
    nll = sum(r["nll_sum"] for r in rows)
    hits = sum(r["correct"] for r in rows)
    return {"stories": len(rows), "tokens": tokens, "nll_sum": nll,
            "correct": hits, "cross_entropy": nll / tokens,
            "perplexity": math.exp(nll / tokens), "top1_accuracy": hits / tokens}


def paired_comparison(ours, reference, resamples=10000, seed=1729):
    if [r["id"] for r in ours] != [r["id"] for r in reference]:
        raise ValueError("Paired story IDs differ")
    n = np.array([r["tokens"] for r in ours], dtype=np.float64)
    if not np.array_equal(n, [r["tokens"] for r in reference]):
        raise ValueError("Paired target counts differ")
    losses = np.array([a["nll_sum"] - b["nll_sum"] for a, b in zip(ours, reference)])
    hits = np.array([a["correct"] - b["correct"] for a, b in zip(ours, reference)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(n), size=(resamples, len(n)))
    denom = n[indices].sum(axis=1)
    ce_ci = np.quantile(losses[indices].sum(axis=1) / denom, [.025, .975]).tolist()
    accuracy_ci = np.quantile(hits[indices].sum(axis=1) / denom, [.025, .975]).tolist()
    delta_ce, delta_accuracy = float(losses.sum() / n.sum()), float(hits.sum() / n.sum())
    # These operational margins are written to protocol.json before model scoring.
    return {"direction": "ours minus released", "delta_cross_entropy": delta_ce,
            "delta_accuracy": delta_accuracy,
            "perplexity_ratio": math.exp(delta_ce), "ce_95_ci": ce_ci,
            "accuracy_95_ci": accuracy_ci,
            "point_estimates_within_margins": abs(delta_ce) <= .10 and abs(delta_accuracy) <= .02,
            "ce_noninferiority_supported": ce_ci[1] <= .10,
            "accuracy_noninferiority_supported": accuracy_ci[0] >= -.02,
            "noninferiority_supported": ce_ci[1] <= .10 and accuracy_ci[0] >= -.02,
            "two_sided_equivalence_supported": ce_ci[0] >= -.10 and ce_ci[1] <= .10
                and accuracy_ci[0] >= -.02 and accuracy_ci[1] <= .02,
            "bootstrap": {"unit": "paired whole story", "resamples": resamples, "seed": seed,
                          "estimator": "token-weighted ratio of resampled story totals"}}


def selected_validation(checkpoint, criterion, receipt, checkpoint_sha256):
    """Bind an accuracy selection to this checkpoint rather than its embedded CE winner."""
    if criterion == "min-ce":
        if checkpoint["cursor"]["updates"] != checkpoint["best"]["updates"]:
            raise ValueError("This checkpoint is not its stored minimum-CE winner")
        return {"cross_entropy": checkpoint["best"]["cross_entropy"]}
    if receipt is None or receipt["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Accuracy selection needs a receipt bound to this checkpoint SHA")
    row = receipt["validation_record"]
    if row.get("event") != "validation" or any(row[k] != checkpoint["cursor"][k] for k in ("updates", "epoch")):
        raise ValueError("Validation record does not match checkpoint cursor")
    metric = row["validation"]
    if metric["tokens"] <= 0 or metric["stories"] <= 0 or not 0 <= metric["correct"] <= metric["tokens"]:
        raise ValueError("Invalid validation counts")
    if not math.isfinite(metric["cross_entropy"]) or metric["cross_entropy"] < 0:
        raise ValueError("Invalid validation CE")
    if not math.isfinite(metric["top1_accuracy"]) or abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) > 1e-12:
        raise ValueError("Accuracy differs from raw counts")
    return metric


@torch.inference_mode()
def score(model, rows, label, output, batch_size=8, chunk_size=32, device="cpu"):
    model.eval()
    records = []
    started = time.monotonic()
    pad = model.config.pad_token_id
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        inputs, targets, mask = batch_tensors(group, device, pad)
        cache = None
        totals = torch.zeros(len(group), dtype=torch.float64)
        counts = torch.zeros(len(group), dtype=torch.long)
        hits = torch.zeros_like(counts)
        for offset in range(0, inputs.shape[1], chunk_size):
            sl = slice(offset, offset + chunk_size)
            out = model(input_ids=inputs[:, sl], attention_mask=mask[:, sl],
                        cache_params=cache, use_cache=True, return_dict=True)
            cache = out.cache_params
            target = targets[:, sl]
            valid = target.ne(pad)
            if not torch.isfinite(out.logits).all():
                raise FloatingPointError("Nonfinite logits")
            losses = F.cross_entropy(out.logits.flatten(0, 1), target.flatten(),
                                     ignore_index=pad, reduction="none").reshape_as(target)
            totals += losses.cpu().double().sum(dim=1)
            counts += valid.sum(dim=1).cpu()
            hits += ((out.logits.argmax(-1) == target) & valid).sum(dim=1).cpu()
        for i, row in enumerate(group):
            assert int(counts[i]) == len(row["ids"]) - 1
            records.append({"id": row["id"], "tokens": int(counts[i]),
                            "correct": int(hits[i]), "nll_sum": float(totals[i])})
        status = {"stage": label, "stories_completed": len(records), "stories_total": len(rows),
                  "seconds": time.monotonic() - started, "partial": aggregate(records)}
        atomic_json(output / "status.json", status)
        print(json.dumps(status), flush=True)
    return {"summary": aggregate(records), "per_story": records,
            "seconds": time.monotonic() - started}


@torch.inference_mode()
def generate(model, tokenizer, device="cpu"):
    samples = []
    for prompt in PROMPTS:
        ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
        tokens = model.generate(torch.tensor([ids], device=device), max_new_tokens=80, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)[0].tolist()
        samples.append({"prompt": prompt, "prompt_ids": ids, "all_ids": tokens,
                        "new_tokens": len(tokens) - len(ids),
                        "text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "continuation": tokenizer.decode(tokens[len(ids):], skip_special_tokens=True)})
        print(json.dumps({"generation": samples[-1]}, ensure_ascii=False), flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--selection", choices=("min-ce", "max-accuracy"), default="min-ce")
    parser.add_argument("--selection-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    verified = verify_reference(args.model)
    original_data = json.loads(args.training_data.read_text())
    audit_data = json.loads(args.audit_data.read_text())
    selection_receipt = json.loads(args.selection_receipt.read_text()) if args.selection_receipt else None
    audit_rows = audit_data["audit"]
    existing = [r for s in ("train", "validation", "test") for r in original_data[s]]
    existing_ids = {r["id"] for r in existing}
    existing_hashes = {r["normalized_sha256"] for r in existing}
    assert len(audit_rows) == 200
    assert not existing_ids.intersection(r["id"] for r in audit_rows)
    assert not existing_hashes.intersection(r["normalized_sha256"] for r in audit_rows)
    assert len({r["id"] for r in audit_rows}) == len(audit_rows)
    assert len({r["normalized_sha256"] for r in audit_rows}) == len(audit_rows)
    for row in audit_rows:
        assert 2 <= len(row["ids"]) <= 320 and row["ids"][0] == 1 and row["ids"][-1] == 2
        assert all(isinstance(i, int) and 0 < i < 1024 for i in row["ids"])
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": False, "device": args.device, "hostname": socket.gethostname(),
        "platform": platform.platform(), "torch": torch.__version__, "threads": args.threads,
        "selection": args.selection,
        "selection_receipt": selection_receipt,
        "selection_receipt_sha256": file_hash(args.selection_receipt) if args.selection_receipt else None,
        "primary_split": "200 new official-validation stories excluded from all 1200 existing rows",
        "diagnostic_split": "Existing 100 validation stories; used in checkpoint selection",
        "existing_final_test_evaluated": False,
        "audit_data_sha256": file_hash(args.audit_data),
        "training_data_sha256": file_hash(args.training_data),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "evaluator_sha256": file_hash(Path(__file__)),
        "helper_sources": {name: file_hash(ROOT / name) for name in
            ("scripts/train_ngxson.py", "scripts/prepare_ngxson.py")},
        "reference_files": verified,
        "practical_margins": {"cross_entropy_nats": .10, "accuracy_absolute": .02,
            "interpretation": "Operational comparison threshold, not a literature standard. Both metrics must pass.",
            "equivalence": "Both paired 95% CIs entirely inside the two-sided margins",
            "noninferiority": "CE upper CI <=0.10 and accuracy lower CI >=-0.02"},
        "generation": {"prompts": PROMPTS, "max_new_tokens": 80, "do_sample": False,
                       "bos": True, "eos": True},
        "limitations": ["Author's training IDs, exact trainer and tokenizer fitting-set overlap are unknown.",
            "One reconstructed training seed and one audit corpus; this is not an exact training reproduction.",
            "After this audit, the new audit set is revealed and must not be treated as unseen after tuning.",
            "The max-accuracy comparison reuses the audit corpus already revealed by the min-CE comparison; it is a descriptive follow-up, not a new untouched test.",
            "Normalized full-text deduplication does not prove absence of semantic overlap."]}
    atomic_json(args.output / "protocol.json", protocol)
    # This file was produced locally by our own trainer, not downloaded executable pickle.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert file_hash(args.checkpoint) == protocol["checkpoint_sha256"], "Checkpoint changed during audit"
    assert checkpoint["manifest"]["data_sha256"] == protocol["training_data_sha256"]
    selected_metric = selected_validation(checkpoint, args.selection, selection_receipt, protocol["checkpoint_sha256"])
    values = checkpoint["parameters"]
    saved_audit = checkpoint["parameter_audit"]
    current_audit = parameter_audit(values)
    assert all(current_audit[k]["sha256"] == saved_audit[k]["sha256"] for k in current_audit)
    checkpoint_meta = {"cursor": checkpoint["cursor"], "best": checkpoint["best"],
                       "selection": args.selection, "selected_validation": selected_metric,
                       "frozen_buffers_sha256": checkpoint["frozen_buffers_sha256"]}
    del checkpoint
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    results = {}
    for name in ("released", "ours_best"):
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     local_files_only=True).eval()
        assert frozen_hashes(model) == checkpoint_meta["frozen_buffers_sha256"]
        if name == "ours_best":
            restore_parameters(model, values)
        before = parameter_audit(parameters_cpu(model))
        if name == "ours_best":
            assert all(before[k]["sha256"] == current_audit[k]["sha256"] for k in before)
        else:
            assert any(before[k]["sha256"] != current_audit[k]["sha256"] for k in before)
        if args.device == "mps":
            from fly_wordbrain.ngxson_mps import enable_mps
            model = enable_mps(model)
        model.requires_grad_(False)
        results[name] = {}
        for split, rows in (("validation", original_data["validation"]), ("audit", audit_rows)):
            results[name][split] = score(model, rows, name + "/" + split, args.output, device=args.device)
            atomic_json(args.output / (name + ".json"), results[name])
        if name == "ours_best":
            replayed = results[name]["validation"]["summary"]
            gap = replayed["cross_entropy"] - selected_metric["cross_entropy"]
            results[name]["saved_validation_ce_replay_delta"] = gap
            assert abs(gap) < 1e-4, f"Saved validation score failed replay: {gap}"
            if "correct" in selected_metric:
                assert all(replayed[k] == selected_metric[k] for k in ("tokens", "stories", "correct")), "Saved validation accuracy/counts failed replay"
        results[name]["samples"] = generate(model, tokenizer, args.device)
        after = parameter_audit(parameters_cpu(model))
        assert before == after
        assert frozen_hashes(model) == checkpoint_meta["frozen_buffers_sha256"]
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        results[name]["parameters_unchanged"] = True
        results[name]["frozen_graph_unchanged"] = True
        if args.device == "mps":
            from fly_wordbrain.ngxson_mps import mps_metadata
            results[name]["mps_backend"] = mps_metadata(model, verify=True)
            assert results[name]["mps_backend"]["backward_kernel_calls"] == 0
        atomic_json(args.output / (name + ".json"), results[name])
        del model
        gc.collect()
    comparisons = {s: paired_comparison(results["ours_best"][s]["per_story"],
                                        results["released"][s]["per_story"])
                   for s in ("validation", "audit")}
    report = {"protocol": protocol, "checkpoint": checkpoint_meta,
              "models": results, "comparisons": comparisons,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output / "results.json", report)
    lines = ["# FlyLLM checkpoint quality comparison", "",
        f"Inference only on macm3 ({args.device}). Frozen checkpoint: update {checkpoint_meta['cursor']['updates']}; selection: {args.selection}.", "",
        "| Split | Model | Stories | Tokens | CE ↓ | Perplexity ↓ | Token accuracy ↑ |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("audit", "validation"):
        for name in results:
            r = results[name][split]["summary"]
            lines.append(f"| {split} | {name} | {r['stories']} | {r['tokens']} | {r['cross_entropy']:.4f} | {r['perplexity']:.2f} | {r['top1_accuracy']:.2%} |")
    c = comparisons["audit"]
    lines.extend(["", f"Audit ΔCE (ours − released): {c['delta_cross_entropy']:.4f}; paired story-bootstrap 95% CI {c['ce_95_ci']}.",
        f"Audit accuracy difference: {100*c['delta_accuracy']:.2f} percentage points; 95% CI {[100*x for x in c['accuracy_95_ci']]}.",
        f"Two-sided equivalence supported within ±0.10 nats / ±2pp: **{c['two_sided_equivalence_supported']}**.",
        f"Non-inferiority supported with those margins: **{c['noninferiority_supported']}**.", "",
        f"Accuracy alone passes its −2pp non-inferiority margin: **{c['accuracy_noninferiority_supported']}**. CE alone passes its +0.10 margin: **{c['ce_noninferiority_supported']}**.", "",
        "The original final test split was not scored. Validation was used to select this checkpoint; the separate 200-story audit was excluded from every existing split.", "",
        "The author's exact data and trainer are unavailable. A quality match supports comparable results from a documented reconstruction, not exact training reproducibility.", "",
        "The max-accuracy follow-up uses the previously revealed audit corpus. Selection uses validation alone; historical logged peaks whose weights were overwritten are distinguished from the best retained checkpoint in selection_receipt.", "",
        "## All fixed greedy continuations", ""])
    for i, prompt in enumerate(PROMPTS):
        lines.append(f"### Prompt {i+1}: {prompt}\n")
        for name in results:
            lines.append(f"**{name}:** {results[name]['samples'][i]['text']}\n")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(args.output / "status.json", {"status": "completed", "comparisons": comparisons})
    print(json.dumps({"completed": True, "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
