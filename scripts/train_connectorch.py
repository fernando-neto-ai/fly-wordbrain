"""Train controlled ConnecTorch adaptations of the pinned ngxson reservoir.

The author released an inference model and a recipe, not the original trainer.
This is a documented reconstruction, with explicit initialization and optimizer
choices. Checkpoints reference immutable graph buffers in --model and contain
all trainable values, optimizer/scheduler/RNG state, and the TBPTT cursor/cache.
The original reference trainer is deliberately independent and remains unchanged.
"""
import argparse
import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np


REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
GRAPH_NAMES = ("brain.w_offsets", "brain.w_indices", "brain.w_values",
               "brain.in_index", "brain.out_index")


def verify_reference(directory):
    # Load the local verification module by absolute path because third-party
    # packages can own a top-level `scripts` module on the training host.
    spec = importlib.util.spec_from_file_location("ngxson_pinned_download", ROOT / "scripts/prepare_ngxson.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: module.verify(directory / name, info) for name, info in module.FILES.items()}


def source_receipt():
    paths = [Path(__file__), ROOT / "scripts/prepare_ngxson.py",
             ROOT / "scripts/prepare_connectorch_groups.py", ROOT / "connectorch-source.json"]
    paths.extend(ROOT / "fly_wordbrain" / name for name in
                 ("connectorch_model.py", "connectorch_backend.py", "metal_sparse_trainable.py", "metal_sparse.py", "plastic_brain.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def connectorch_receipt():
    from fly_wordbrain.connectorch_backend import connectorch_receipt as installed_receipt
    return installed_receipt()


def environment_receipt():
    versions = {}
    for name in ("torch", "transformers", "numpy", "safetensors", "connectorch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "mps_available": torch.backends.mps.is_available(),
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hashes(model):
    buffers = dict(model.named_buffers())
    result = {}
    names = list(GRAPH_NAMES)
    names.extend(name for name in ("brain.edge_group_index", "brain.node_type_index") if name in buffers)
    for name in names:
        value = buffers[name]
        if value.requires_grad:
            raise ValueError("Frozen graph buffer requires gradients: " + name)
        array = value.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


def load_groups(path):
    """Read graph-bound canonical-node types without allowing Python pickles."""
    with np.load(path, allow_pickle=False) as arrays:
        if "node_type_index" not in arrays or "metadata_json" not in arrays:
            raise ValueError("Grouping NPZ must contain node_type_index and metadata_json")
        groups = np.asarray(arrays["node_type_index"])
        if groups.ndim != 1 or groups.size == 0 or groups.dtype.kind not in "iu":
            raise ValueError("node_type_index must be a nonempty integer vector")
        if groups.min() < 0 or groups.max() > np.iinfo(np.int32).max:
            raise ValueError("Node type indices must be nonnegative int32-compatible values")
        unique = np.unique(groups)
        if not np.array_equal(unique, np.arange(int(groups.max()) + 1)):
            raise ValueError("Node type indices must cover consecutive groups starting at zero")
        metadata = json.loads(str(arrays["metadata_json"].item()))
        if not isinstance(metadata, dict) or "frozen_buffers_sha256" not in metadata:
            raise ValueError("Grouping metadata must bind the original frozen graph and interface hashes")
        return torch.from_numpy(groups.astype(np.int64, copy=True)), metadata


def build_model(reference, args, groups):
    from fly_wordbrain.connectorch_model import build_connectorch_model
    lag_map = list(range(args.history_length)) * (8 // args.history_length)
    return build_connectorch_model(reference, d_embed=args.d_embed, plasticity=args.plasticity,
        leak=getattr(args, "leak", "fixed"),
        node_type_index=groups, readout_rank=args.readout_rank, lag_map=lag_map,
        seed=args.seed, copy_reference_parameters=False, device="cpu")


def parameter_counts(model):
    """Independent partition catches missing, extra or unexpectedly frozen parameters."""
    counts = {"embedding": 0, "input_projection": 0, "neurons": 0,
              "edge_gains": 0, "layernorm": 0, "readout": 0, "other": 0}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name == "brain.wte.weight":
            group = "embedding"
        elif name == "brain.in_proj":
            group = "input_projection"
        elif name in ("brain.gain", "brain.rec_gain", "brain.bias", "brain.leak_delta"):
            group = "neurons"
        elif name.startswith("brain.edge_theta"):
            group = "edge_gains"
        elif name.startswith("ln."):
            group = "layernorm"
        elif name.startswith("lm_head."):
            group = "readout"
        else:
            group = "other"
        counts[group] += value.numel()
    counts["total"] = sum(counts.values())
    return counts


def expected_parameter_counts(model, args, groups):
    cfg = model.config
    neurons, vocab, width, output = cfg.n_neurons, cfg.vocab_size, args.d_embed, cfg.n_out
    counts = {"embedding": vocab * width,
              "input_projection": model.brain.in_index.numel() * width,
              "neurons": (4 if getattr(args, "leak", "fixed") == "trainable" else 3) * neurons,
              "edge_gains": 2 * (int(groups.max()) + 1) if args.plasticity == "bounded10" else 0,
              "layernorm": 2 * output,
              "readout": output * vocab if args.readout_rank == 0 else args.readout_rank * (output + vocab),
              "other": 0}
    counts["total"] = sum(counts.values())
    return counts


def model_audit(model, expected_hashes=None):
    hashes = frozen_hashes(model)
    if expected_hashes is not None and hashes != expected_hashes:
        raise RuntimeError("Frozen graph, interface or edge-group buffers changed")
    model.verify_frozen()
    return {"parameter_counts": parameter_counts(model), "adaptation": model.adaptation_report(),
            "frozen_buffers_sha256": hashes, "frozen_buffers_preserved": True}


def batch_tensors(rows, device, pad_id=0):
    lengths = [len(row["ids"]) for row in rows]
    ids = torch.full((len(rows), max(lengths)), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :lengths[i]] = torch.tensor(row["ids"], dtype=torch.long, device=device)
        mask[i, :lengths[i]] = 1
    return ids[:, :-1], ids[:, 1:], mask[:, :-1]


def detach_cache(cache):
    if cache is None:
        return None
    cache.state = cache.state.detach()
    cache.last_tokens = cache.last_tokens.detach()
    return cache


def cache_state(cache):
    if cache is None:
        return None
    return {"state": cache.state.detach().cpu().clone(),
            "last_tokens": cache.last_tokens.detach().cpu().clone(), "seq_len": int(cache.seq_len)}


def restore_cache(value, device):
    if value is None:
        return None
    # The original forward reads these three attributes; it returns its own
    # upstream FlyCache on the next call. No alternate recurrent logic is used.
    return SimpleNamespace(state=value["state"].to(device),
                           last_tokens=value["last_tokens"].to(device), seq_len=value["seq_len"])


def chunk_loss(model, inputs, targets, mask, cache=None, pad_id=0):
    output = model(input_ids=inputs, attention_mask=mask, cache_params=cache,
                   use_cache=True, return_dict=True)
    # Targets are already shifted globally before slicing chunks. Passing
    # labels to the upstream model would shift a second time and lose boundaries.
    summed = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                             targets.reshape(-1), ignore_index=pad_id, reduction="sum")
    count = int(targets.ne(pad_id).sum().item())
    if count == 0:
        raise ValueError("A chunk contains no supervised target tokens")
    correct = int(((output.logits.argmax(-1) == targets) & targets.ne(pad_id)).sum().item())
    return summed / count, count, correct, output.cache_params


def epoch_batches(rows, epoch, batch_size, seed):
    indices = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def epoch_updates(rows, epoch, batch_size, chunk_size, seed):
    return sum(math.ceil((max(len(rows[i]["ids"]) for i in group) - 1) / chunk_size)
               for group in epoch_batches(rows, epoch, batch_size, seed))


def evaluate(model, rows, device, batch_size=8, chunk_size=32, pad_id=0, limit=None):
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Evaluation split is empty")
    was_training = model.training
    model.eval()
    summed, count, correct = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, targets, mask = batch_tensors(rows[start:start + batch_size], device, pad_id)
            cache = None  # Every batch starts independent stories from zero state.
            for offset in range(0, inputs.shape[1], chunk_size):
                sl = slice(offset, offset + chunk_size)
                loss, n, hits, cache = chunk_loss(model, inputs[:, sl], targets[:, sl], mask[:, sl], cache, pad_id)
                summed += float(loss.item()) * n
                count += n
                correct += hits
    model.train(was_training)
    return {"cross_entropy": summed / count, "top1_accuracy": correct / count,
            "correct": correct, "tokens": count, "stories": len(rows)}


def optimizer_for(model, phase):
    rates = (1e-3, 1e-4) if phase == 0 else (3e-4, 3e-5)
    brain, readout = [], []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        (brain if name.startswith("brain.") else readout).append(value)
    return torch.optim.AdamW([{"params": brain, "lr": rates[0], "name": "input_and_neurons"},
                              {"params": readout, "lr": rates[1], "name": "readout_and_norm"}],
                             betas=(.9, .999), weight_decay=.01)


def cosine_scheduler(optimizer, total_updates):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: .5 * (1 + math.cos(math.pi * min(step, total_updates) / max(1, total_updates))))


def parameters_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def parameter_audit(values, initial=None):
    result = {}
    for name, value in values.items():
        finite = bool(torch.isfinite(value).all())
        if not finite:
            raise FloatingPointError("Non-finite parameter: " + name)
        digest = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        result[name] = {"finite": finite, "sha256": digest, "values": value.numel(),
                        "changed_from_initialization": None if initial is None else digest != initial[name]["sha256"]}
    return result


def gradient_audit(model):
    result = {}
    for name, value in model.named_parameters():
        grad = value.grad
        result[name] = {"present": grad is not None, "finite": None if grad is None else bool(torch.isfinite(grad).all().item()),
                        "nonzero_values": 0 if grad is None else int(grad.count_nonzero().item()),
                        "values": value.numel(), "norm": None if grad is None else float(grad.norm().item())}
        if grad is not None and not result[name]["finite"]:
            raise FloatingPointError("Non-finite gradient: " + name)
    if hasattr(model.brain, "wte"):
        grad = model.brain.wte.weight.grad
        result["padding_embedding_gradient_nonzero"] = None if grad is None else int(grad[model.config.pad_token_id].count_nonzero().item())
    return result


def restore_parameters(model, values):
    current = dict(model.named_parameters())
    if current.keys() != values.keys():
        raise ValueError("Checkpoint parameter names differ from pinned architecture")
    with torch.no_grad():
        for name, parameter in current.items():
            if parameter.shape != values[name].shape:
                raise ValueError("Checkpoint parameter shape mismatch: " + name)
            parameter.copy_(values[name])


def rng_state(device):
    result = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.device(device).type == "mps":
        result["torch_mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(value, device):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.device(device).type == "mps" and "torch_mps" in value:
        torch.mps.set_rng_state(value["torch_mps"])


def save_checkpoint(path, model, optimizer, scheduler, cursor, cache, best, manifest, device, best_accuracy=None):
    structure = model_audit(model, manifest["frozen_buffers_sha256"])
    hashes = structure["frozen_buffers_sha256"]
    parameters = parameters_cpu(model)
    audit = parameter_audit(parameters, manifest["initial_parameter_audit"])
    if source_receipt() != manifest["trainer_sources_sha256"]:
        raise RuntimeError("Training source changed during the run")
    if connectorch_receipt() != manifest["connectorch_source"]:
        raise RuntimeError("Installed Connectorch source changed during the run")
    value = {"format_version": 1, "parameters": parameters, "parameter_audit": audit,
             "model_audit": structure,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "cursor": dict(cursor), "cache": cache_state(cache), "best": dict(best),
             "best_accuracy": dict(best_accuracy) if best_accuracy is not None else None,
             "rng": rng_state(device), "manifest": manifest, "frozen_buffers_sha256": hashes}
    temp = Path(str(path) + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def select_validation_checkpoints(metric, cursor, best, best_accuracy):
    """Independent selectors retain the earlier checkpoint on exact metric ties."""
    if not math.isfinite(metric["cross_entropy"]) or not 0 <= metric["top1_accuracy"] <= 1:
        raise ValueError("Checkpoint selection requires finite validation metrics")
    ce_improved = best["cross_entropy"] is None or metric["cross_entropy"] < best["cross_entropy"]
    accuracy_improved = best_accuracy["top1_accuracy"] is None or metric["top1_accuracy"] > best_accuracy["top1_accuracy"]
    selected = {"cross_entropy": metric["cross_entropy"], "top1_accuracy": metric["top1_accuracy"],
                "updates": cursor["updates"], "epoch": cursor["epoch"]}
    return (dict(selected) if ce_improved else best, dict(selected) if accuracy_improved else best_accuracy,
            ce_improved, accuracy_improved)


def selected_checkpoint_receipt(path, cursor, metric, audit):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path), "cursor": dict(cursor),
            "validation": dict(metric), "frozen_buffers_preserved": audit["frozen_buffers_preserved"],
            "frozen_buffers_sha256": audit["frozen_buffers_sha256"]}


def validate_resume_selection(saved, receipt, directory):
    """Fail closed after a partial multi-file save or an attempt to rewind a run."""
    selectors = receipt.get("selectors", {})
    expected = {"minimum_validation_ce": ("best.pt", saved["best"], "cross_entropy", min),
                "maximum_validation_accuracy": ("best-accuracy.pt", saved["best_accuracy"], "top1_accuracy", max)}
    if receipt.get("format_version") != 1 or set(selectors) != set(expected):
        raise ValueError("Resume requires both retained validation checkpoint receipts")
    records = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
    records = [record for record in records if record.get("event") == "validation"]
    if not records or any(record["updates"] > saved["cursor"]["updates"] for record in records):
        raise ValueError("Validation history is ahead of the resume checkpoint; recover a consistent checkpoint set")
    for name, (filename, selection, metric_name, choose) in expected.items():
        entry, path = selectors[name], Path(directory) / filename
        record = choose(records, key=lambda record: record["validation"][metric_name])
        if (Path(entry.get("path", "")).resolve() != path.resolve()
                or file_hash(path) != entry.get("sha256")
                or entry.get("cursor", {}).get("updates") != selection["updates"]
                or entry["cursor"].get("epoch") != selection["epoch"]
                or entry.get("validation") != record["validation"]
                or record["updates"] != selection["updates"]
                or any(entry["validation"].get(key) != selection.get(key) for key in ("cross_entropy", "top1_accuracy"))
                or entry.get("frozen_buffers_preserved") is not True
                or entry.get("frozen_buffers_sha256") != saved["manifest"]["frozen_buffers_sha256"]):
            raise ValueError("Retained selector and resume checkpoint disagree: " + name)


def validate_data(data, config):
    seen = set()
    for split in ("train", "validation", "test"):
        if not data.get(split):
            raise ValueError("Missing or empty split: " + split)
        for row in data[split]:
            ids = row["ids"]
            if row["id"] in seen:
                raise ValueError("Repeated story ID across data: " + str(row["id"]))
            seen.add(row["id"])
            if not (2 <= len(ids) <= 320 and ids[0] == config.bos_token_id and ids[-1] == config.eos_token_id):
                raise ValueError("Stories must have BOS/EOS and 2..320 tokens")
            if any(not isinstance(i, int) or i <= config.pad_token_id or i >= config.vocab_size for i in ids):
                raise ValueError("Story contains padding or an invalid token ID")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--d-embed", type=int, choices=(32, 128), default=128)
    p.add_argument("--plasticity", choices=("fixed", "bounded10"), default="fixed")
    p.add_argument("--leak", choices=("fixed", "trainable"), default="fixed",
                   help="trainable: one per-neuron time constant, an offset from the pinned "
                        "0.9 in logit space so it starts identical to fixed")
    p.add_argument("--readout-rank", type=int, default=0)
    p.add_argument("--history-length", type=int, choices=(1, 2, 4, 8), default=8)
    p.add_argument("--skip-final-test", action="store_true",
                   help="Reserve the test split for a later validation-selected campaign result")
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--second-epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--eval-limit", type=int)
    p.add_argument("--eval-interval-updates", type=int, default=100)
    p.add_argument("--resume", type=Path)
    return p


def run(args):
    reference_files = verify_reference(args.model)
    from transformers import AutoModelForCausalLM
    if min(args.epochs, args.batch_size, args.chunk_size) < 1 or args.second_epochs < 0:
        raise ValueError("Epochs, batch and chunk sizes must be positive; second epochs may be zero")
    if args.eval_interval_updates < 0 or any(x is not None and x < 1 for x in (args.max_updates, args.eval_limit)):
        raise ValueError("Evaluation interval must be nonnegative and debug limits positive")
    if args.readout_rank < 0:
        raise ValueError("Readout rank must be nonnegative; zero selects the full readout")
    if not args.resume and any((args.output / x).exists() for x in ("manifest.json", "latest.pt", "metrics.jsonl")):
        raise FileExistsError("Output already contains a run; use a new directory or explicit --resume")
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError("Resume in the checkpoint's output directory to preserve best.pt and metric history")
    if args.resume and (args.output / "results.json").exists():
        completed = json.loads((args.output / "results.json").read_text())
        if completed.get("test_evaluated"):
            raise ValueError("The run already evaluated its selected final test; resuming would repeat test access")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = json.loads(args.data.read_text())
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                   local_files_only=True, torch_dtype=torch.float32).cpu()
    reference_hashes = frozen_hashes(reference)
    groups, group_metadata = load_groups(args.groups)
    if groups.numel() != reference.config.n_neurons:
        raise ValueError("Grouping has a different neuron count from the reference graph")
    if group_metadata["frozen_buffers_sha256"] != reference_hashes:
        raise ValueError("Grouping metadata belongs to a different reference graph or interface")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(reference, args, groups)
    adapted_hashes = frozen_hashes(model)
    for name, digest in reference_hashes.items():
        if adapted_hashes[name] != digest:
            raise RuntimeError("Adaptation changed an original reference buffer: " + name)
    del reference
    validate_data(data, model.config)
    before = frozen_hashes(model)
    counts = parameter_counts(model)
    expected = expected_parameter_counts(model, args, groups)
    if counts != expected:
        raise ValueError("Adaptation parameter count mismatch: " + str({"actual": counts, "expected": expected}))
    init = {"trainable_parameters": counts["total"], "parameter_counts": counts,
            "model_audit": model_audit(model, before)}
    initial_audit = parameter_audit(parameters_cpu(model))
    if frozen_hashes(model) != before:
        raise RuntimeError("Initialization mutated frozen buffers")
    epoch_steps = [epoch_updates(data["train"], epoch, args.batch_size, args.chunk_size, args.seed)
                   for epoch in range(args.epochs + args.second_epochs)]
    phase_steps = [sum(epoch_steps[:args.epochs]), sum(epoch_steps[args.epochs:])]
    config = {key: getattr(args, key) for key in ("epochs", "second_epochs", "batch_size", "chunk_size", "seed",
                                                "max_updates", "eval_limit", "eval_interval_updates", "device",
                                                "d_embed", "plasticity", "readout_rank", "history_length", "skip_final_test")}
    source = {name: file_hash(args.model / name) for name in ("config.json", "modeling_fly.py", "configuration_fly.py", "tokenizer.json")}
    source["trainer"] = file_hash(__file__)
    manifest = {"reference_repository": "ngxson/fly-llm-hf", "reference_revision": REVISION,
                "model_path": str(args.model.resolve()), "data_path": str(args.data.resolve()),
                "groups_path": str(args.groups.resolve()), "groups_sha256": file_hash(args.groups),
                "node_types": int(groups.max()) + 1, "groups_provenance": group_metadata,
                "parameter_counts": counts,
                "data_sha256": file_hash(args.data), "source_sha256": source,
                "reference_files": reference_files, "trainer_sources_sha256": source_receipt(),
                "connectorch_source": connectorch_receipt(),
                "environment": environment_receipt(), "initial_parameter_audit": initial_audit,
                "data_provenance": data.get("provenance", {}), "config": config,
                "frozen_buffers_sha256": before, "initialization": init,
                "start_from": "random learned values; released frozen graph and interface only",
                "debug": args.max_updates is not None or args.eval_limit is not None,
                "planned_phase_updates": phase_steps, "epoch_updates": epoch_steps,
                "assumptions": {"batch_size": args.batch_size, "seed": args.seed, "betas": [.9, .999],
                                "weight_decay": .01, "gradient_clip_norm": 1.0,
                                "adam_epsilon": 1e-8, "zero_input_recurrent_gain": 1.0,
                                "layernorm_optimizer_group": "readout",
                                "phase2_optimizer": "keep Adam moments; reset learning rates and cosine schedule",
                                "scheduler": "cosine to zero per phase, every chunk optimizer update",
                                "selection": "retain independent minimum validation CE and maximum validation top1 accuracy; earlier on exact ties; final test uses CE winner",
                                "data_order": "random.Random(seed+zero_based_epoch), reshuffle whole stories",
                                "initialization_order": "embedding,input_projection,gain,bias,rec_gain,LayerNorm,readout; edge_theta=0",
                                "edge_gain_optimizer_group": "input_and_neurons",
                                "edge_gain_parameterization": "1 + 0.1*tanh(theta_source[type(src)] + theta_destination[type(dst)])" if args.plasticity == "bounded10" else "fixed at 1",
                                "neuron_gain_parameterization": "original unconstrained gain and rec_gain",
                                "lag_map": list(range(args.history_length)) * (8 // args.history_length),
                                "test_policy": "deferred" if args.skip_final_test else "once after validation selection"},
                "split_sizes": {split: len(data[split]) for split in ("train", "validation", "test")}}
    cursor = {"epoch": 0, "batch": 0, "offset": 0, "updates": 0, "phase": 0,
              "epoch_loss_sum": 0.0, "epoch_tokens": 0, "epoch_correct": 0}
    best = {"cross_entropy": None, "updates": 0, "epoch": 0}
    best_accuracy = {"top1_accuracy": None, "updates": 0, "epoch": 0}
    selected_checkpoints = {"format_version": 1, "selectors": {}}
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("data_sha256", "groups_sha256", "source_sha256", "frozen_buffers_sha256", "config",
                    "trainer_sources_sha256", "connectorch_source"):
            if saved["manifest"][key] != manifest[key]:
                raise ValueError("Resume mismatch: " + key)
        restore_parameters(model, saved["parameters"])
        cursor, best = dict(saved["cursor"]), dict(saved["best"])
        if saved.get("best_accuracy") is None:
            raise ValueError("Resume checkpoint lacks the independent validation accuracy selector")
        best_accuracy = dict(saved["best_accuracy"])
        selected_checkpoints = json.loads((args.output / "selected-checkpoints.json").read_text())
        validate_resume_selection(saved, selected_checkpoints, args.output)
    model.to(args.device)
    manifest["backend"] = model.backend_metadata()
    optimizer = optimizer_for(model, cursor["phase"])
    scheduler = cosine_scheduler(optimizer, phase_steps[cursor["phase"]])
    cache = None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        cache = restore_cache(saved["cache"], args.device)
        restore_rng(saved["rng"], args.device)
        del saved
    atomic_json(args.output / "manifest.json", manifest)

    def status(phase, **extra):
        atomic_json(args.output / "status.json", {"status": "running", "activity": phase, "pid": os.getpid(),
                    **cursor, "elapsed_seconds_this_invocation": time.monotonic() - started,
                    "debug": manifest["debug"], "best": best, "best_accuracy": best_accuracy, **extra})

    def validation(reason):
        nonlocal best, best_accuracy
        status("validation")
        metric = evaluate(model, data["validation"], args.device, args.batch_size, args.chunk_size,
                          model.config.pad_token_id, args.eval_limit)
        record = {"event": "validation", "reason": reason, "updates": cursor["updates"],
                  "epoch": cursor["epoch"], "validation": metric,
                  "model_audit": model_audit(model, before),
                  "elapsed_seconds_this_invocation": time.monotonic() - started}
        if cursor["epoch_tokens"]:
            record["training_epoch_so_far"] = {"cross_entropy": cursor["epoch_loss_sum"] / cursor["epoch_tokens"],
                    "top1_accuracy": cursor["epoch_correct"] / cursor["epoch_tokens"], "tokens": cursor["epoch_tokens"]}
        best, best_accuracy, improved, accuracy_improved = select_validation_checkpoints(metric, cursor, best, best_accuracy)
        record["best"] = best
        record["best_accuracy"] = best_accuracy
        append_json(args.output / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if reason == "epoch_end":
            # The cursor now names the next epoch, so resumable checkpoints must
            # carry fresh epoch accumulators, including best.pt checkpoints.
            cursor["epoch_loss_sum"], cursor["epoch_tokens"], cursor["epoch_correct"] = 0.0, 0, 0
        if improved:
            path = args.output / "best.pt"
            save_checkpoint(path, model, optimizer, scheduler, cursor, cache, best, manifest, args.device, best_accuracy)
            selected_checkpoints["selectors"]["minimum_validation_ce"] = selected_checkpoint_receipt(path, cursor, metric, record["model_audit"])
        if accuracy_improved:
            path = args.output / "best-accuracy.pt"
            save_checkpoint(path, model, optimizer, scheduler, cursor, cache, best, manifest, args.device, best_accuracy)
            selected_checkpoints["selectors"]["maximum_validation_accuracy"] = selected_checkpoint_receipt(path, cursor, metric, record["model_audit"])
        atomic_json(args.output / "selected-checkpoints.json", selected_checkpoints)
        save_checkpoint(args.output / "latest.pt", model, optimizer, scheduler, cursor, cache, best, manifest, args.device, best_accuracy)

    if not args.resume:
        validation("initialization")
    if args.max_updates is not None and cursor["updates"] >= args.max_updates:
        atomic_json(args.output / "status.json", {"status": "debug_stopped", **cursor,
                    "debug": True, "best": best, "best_accuracy": best_accuracy, "test_evaluated": False,
                    "elapsed_seconds_this_invocation": time.monotonic() - started})
        return
    model.train()
    for epoch in range(cursor["epoch"], args.epochs + args.second_epochs):
        phase = int(epoch >= args.epochs)
        if phase != cursor["phase"]:
            # The paper does not specify a moment reset: preserve Adam state.
            for group, rate in zip(optimizer.param_groups, (3e-4, 3e-5)):
                group["lr"], group["initial_lr"] = rate, rate
            scheduler = cosine_scheduler(optimizer, phase_steps[phase])
            cursor["phase"] = phase
        batches = epoch_batches(data["train"], epoch, args.batch_size, args.seed)
        for batch_index in range(cursor["batch"], len(batches)):
            group = [data["train"][i] for i in batches[batch_index]]
            inputs, targets, mask = batch_tensors(group, args.device, model.config.pad_token_id)
            for offset in range(cursor["offset"], inputs.shape[1], args.chunk_size):
                sl = slice(offset, offset + args.chunk_size)
                optimizer.zero_grad(set_to_none=True)
                loss, count, hits, next_cache = chunk_loss(model, inputs[:, sl], targets[:, sl], mask[:, sl],
                                                          cache, model.config.pad_token_id)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                if cursor["updates"] == 0:
                    atomic_json(args.output / "initial-gradients.json", gradient_audit(model))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                used_rates = [group["lr"] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                cache = detach_cache(next_cache)
                cursor["updates"] += 1
                cursor["epoch_loss_sum"] += float(loss.item()) * count
                cursor["epoch_tokens"] += count
                cursor["epoch_correct"] += hits
                cursor["offset"] = offset + args.chunk_size
                if cursor["offset"] >= inputs.shape[1]:
                    cursor["batch"], cursor["offset"], cache = batch_index + 1, 0, None
                append_json(args.output / "training.jsonl", {"updates": cursor["updates"], "epoch": epoch,
                            "batch": batch_index, "offset": offset, "loss": float(loss.item()),
                            "tokens": count, "top1_accuracy": hits / count, "gradient_norm": float(norm.item()),
                            "learning_rates": used_rates})
                status("training")
                limited = args.max_updates is not None and cursor["updates"] >= args.max_updates
                due = args.eval_interval_updates and cursor["updates"] % args.eval_interval_updates == 0
                if due or limited:
                    validation("debug_limit" if limited else "update_interval")
                if limited:
                    atomic_json(args.output / "status.json", {"status": "debug_stopped", **cursor,
                                "debug": True, "best": best, "best_accuracy": best_accuracy, "test_evaluated": False,
                                "elapsed_seconds_this_invocation": time.monotonic() - started})
                    return
            cache = None
        # Keep the next-epoch cursor in the checkpoint for exact resumption.
        cursor["epoch"], cursor["batch"], cursor["offset"] = epoch + 1, 0, 0
        validation("epoch_end")
        cursor["epoch_loss_sum"], cursor["epoch_tokens"], cursor["epoch_correct"] = 0.0, 0, 0
        save_checkpoint(args.output / "latest.pt", model, optimizer, scheduler, cursor, cache, best, manifest, args.device, best_accuracy)

    checks = model_audit(model, before)
    checks["backend"] = model.backend_metadata()
    selected = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    restore_parameters(model, selected["parameters"])
    del selected
    checks["selected_checkpoint_audit"] = model_audit(model, before)
    result = {"status": "debug_completed" if manifest["debug"] else "completed", "debug": manifest["debug"],
              "updates": cursor["updates"], "epochs": cursor["epoch"], "selected": best,
              "selected_accuracy": best_accuracy, "selected_checkpoints": selected_checkpoints["selectors"], "checks": checks,
              "test_evaluated": False, "test_deferred": bool(args.skip_final_test)}
    if not manifest["debug"] and not args.skip_final_test:
        status("final_test")
        result["test"] = evaluate(model, data["test"], args.device, args.batch_size, args.chunk_size, model.config.pad_token_id)
        result["test_evaluated"] = True
    result["elapsed_seconds_this_invocation"] = time.monotonic() - started
    atomic_json(args.output / "results.json", result)
    atomic_json(args.output / "status.json", result)
    print(json.dumps(result), flush=True)


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            atomic_json(args.output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
