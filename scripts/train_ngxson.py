"""Retrain the pinned ngxson reservoir architecture; the released graph stays fixed.

The author released an inference model and a recipe, not the original trainer.
This is a documented reconstruction, with explicit initialization and optimizer
choices. Checkpoints reference immutable graph buffers in --model and contain
all trainable values, optimizer/scheduler/RNG state, and the TBPTT cursor/cache.
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


EXPECTED_PARAMETERS = 52756661
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
    paths = [Path(__file__), ROOT / "scripts/prepare_ngxson.py"]
    paths.extend(ROOT / "fly_wordbrain" / name for name in ("ngxson_mps.py", "metal_sparse.py", "plastic_brain.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def environment_receipt():
    versions = {}
    for name in ("torch", "transformers", "numpy", "safetensors"):
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
    for name in GRAPH_NAMES:
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


def initialize_from_scratch(model, seed):
    """Reset every learned value, retaining only released graph/interface buffers."""
    torch.manual_seed(seed)
    random.seed(seed)
    brain = model.brain
    cfg = model.config
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Initialize on CPU before enabling the device backend")
    with torch.no_grad():
        brain.wte.weight.normal_(std=1.0)
        brain.wte.weight[cfg.pad_token_id].zero_()
        brain.in_proj.normal_(std=1 / math.sqrt(cfg.d_embed))
        brain.gain.fill_(1)
        brain.bias.zero_()
        # Zero-input rows have no recurrent drive; gain=1 avoids infinities.
        sizes = (brain.w_offsets[1:] - brain.w_offsets[:-1]).long()
        rows = torch.repeat_interleave(torch.arange(cfg.n_neurons), sizes)
        sums = torch.zeros(cfg.n_neurons, dtype=brain.w_values.dtype)
        sums.index_add_(0, rows, brain.w_values.abs())
        brain.rec_gain.copy_(torch.where(sums > 0, cfg.rec_target / sums.clamp_min(1e-30), 1.0))
        model.ln.weight.fill_(1)
        model.ln.bias.zero_()
        model.lm_head.weight.normal_(std=min(.02, 1 / math.sqrt(cfg.n_out)))
    return {"zero_input_rows": int((sums == 0).sum()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


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


def save_checkpoint(path, model, optimizer, scheduler, cursor, cache, best, manifest, device):
    hashes = frozen_hashes(model)
    if hashes != manifest["frozen_buffers_sha256"]:
        raise RuntimeError("Frozen graph or interface buffers changed")
    parameters = parameters_cpu(model)
    audit = parameter_audit(parameters, manifest["initial_parameter_audit"])
    if source_receipt() != manifest["trainer_sources_sha256"]:
        raise RuntimeError("Training source changed during the run")
    value = {"format_version": 1, "parameters": parameters, "parameter_audit": audit,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "cursor": dict(cursor), "cache": cache_state(cache), "best": dict(best),
             "rng": rng_state(device), "manifest": manifest, "frozen_buffers_sha256": hashes}
    temp = Path(str(path) + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


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
    if not args.resume and any((args.output / x).exists() for x in ("manifest.json", "latest.pt", "metrics.jsonl")):
        raise FileExistsError("Output already contains a run; use a new directory or explicit --resume")
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError("Resume in the checkpoint's output directory to preserve best.pt and metric history")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = json.loads(args.data.read_text())
    model = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                               local_files_only=True, torch_dtype=torch.float32).cpu()
    validate_data(data, model.config)
    before = frozen_hashes(model)
    init = initialize_from_scratch(model, args.seed)
    initial_audit = parameter_audit(parameters_cpu(model))
    if init["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("Reference architecture parameter count changed: " + str(init))
    if frozen_hashes(model) != before:
        raise RuntimeError("Initialization mutated frozen buffers")
    epoch_steps = [epoch_updates(data["train"], epoch, args.batch_size, args.chunk_size, args.seed)
                   for epoch in range(args.epochs + args.second_epochs)]
    phase_steps = [sum(epoch_steps[:args.epochs]), sum(epoch_steps[args.epochs:])]
    config = {key: getattr(args, key) for key in ("epochs", "second_epochs", "batch_size", "chunk_size", "seed",
                                                "max_updates", "eval_limit", "eval_interval_updates", "device")}
    source = {name: file_hash(args.model / name) for name in ("config.json", "modeling_fly.py", "configuration_fly.py", "tokenizer.json")}
    source["trainer"] = file_hash(__file__)
    manifest = {"reference_repository": "ngxson/fly-llm-hf", "reference_revision": REVISION,
                "model_path": str(args.model.resolve()), "data_path": str(args.data.resolve()),
                "data_sha256": file_hash(args.data), "source_sha256": source,
                "reference_files": reference_files, "trainer_sources_sha256": source_receipt(),
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
                                "selection": "minimum token-weighted validation cross entropy; earlier on exact tie",
                                "data_order": "random.Random(seed+zero_based_epoch), reshuffle whole stories",
                                "initialization_order": "embedding,input_projection,gain,bias,rec_gain,LayerNorm,readout"},
                "split_sizes": {split: len(data[split]) for split in ("train", "validation", "test")}}
    cursor = {"epoch": 0, "batch": 0, "offset": 0, "updates": 0, "phase": 0,
              "epoch_loss_sum": 0.0, "epoch_tokens": 0, "epoch_correct": 0}
    best = {"cross_entropy": None, "updates": 0, "epoch": 0}
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("data_sha256", "source_sha256", "frozen_buffers_sha256", "config", "trainer_sources_sha256"):
            if saved["manifest"][key] != manifest[key]:
                raise ValueError("Resume mismatch: " + key)
        restore_parameters(model, saved["parameters"])
        cursor, best = dict(saved["cursor"]), dict(saved["best"])
    if args.device == "mps":
        from fly_wordbrain.ngxson_mps import enable_mps, mps_metadata
        enable_mps(model)
        manifest["backend"] = mps_metadata(model, verify=True)
    else:
        manifest["backend"] = {"device": "cpu", "explicit_cpu_requested": True, "torch": torch.__version__}
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
                    "debug": manifest["debug"], **extra})

    def validation(reason):
        nonlocal best
        status("validation")
        metric = evaluate(model, data["validation"], args.device, args.batch_size, args.chunk_size,
                          model.config.pad_token_id, args.eval_limit)
        record = {"event": "validation", "reason": reason, "updates": cursor["updates"],
                  "epoch": cursor["epoch"], "validation": metric,
                  "elapsed_seconds_this_invocation": time.monotonic() - started}
        if cursor["epoch_tokens"]:
            record["training_epoch_so_far"] = {"cross_entropy": cursor["epoch_loss_sum"] / cursor["epoch_tokens"],
                    "top1_accuracy": cursor["epoch_correct"] / cursor["epoch_tokens"], "tokens": cursor["epoch_tokens"]}
        improved = best["cross_entropy"] is None or metric["cross_entropy"] < best["cross_entropy"]
        if improved:
            best = {"cross_entropy": metric["cross_entropy"], "updates": cursor["updates"], "epoch": cursor["epoch"]}
        record["best"] = best
        append_json(args.output / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if reason == "epoch_end":
            # The cursor now names the next epoch, so resumable checkpoints must
            # carry fresh epoch accumulators, including best.pt checkpoints.
            cursor["epoch_loss_sum"], cursor["epoch_tokens"], cursor["epoch_correct"] = 0.0, 0, 0
        if improved:
            save_checkpoint(args.output / "best.pt", model, optimizer, scheduler, cursor, cache, best, manifest, args.device)
        save_checkpoint(args.output / "latest.pt", model, optimizer, scheduler, cursor, cache, best, manifest, args.device)

    if not args.resume:
        validation("initialization")
    if args.max_updates is not None and cursor["updates"] >= args.max_updates:
        atomic_json(args.output / "status.json", {"status": "debug_stopped", **cursor,
                    "debug": True, "best": best, "test_evaluated": False,
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
                                "debug": True, "best": best, "test_evaluated": False,
                                "elapsed_seconds_this_invocation": time.monotonic() - started})
                    return
            cache = None
        # Keep the next-epoch cursor in the checkpoint for exact resumption.
        cursor["epoch"], cursor["batch"], cursor["offset"] = epoch + 1, 0, 0
        validation("epoch_end")
        cursor["epoch_loss_sum"], cursor["epoch_tokens"], cursor["epoch_correct"] = 0.0, 0, 0
        save_checkpoint(args.output / "latest.pt", model, optimizer, scheduler, cursor, cache, best, manifest, args.device)

    checks = {"frozen_buffers_sha256": frozen_hashes(model), "frozen_buffers_preserved": frozen_hashes(model) == before}
    if not checks["frozen_buffers_preserved"]:
        raise RuntimeError("Frozen buffers changed at completion")
    if args.device == "mps":
        checks["backend"] = mps_metadata(model, verify=True)
    selected = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    restore_parameters(model, selected["parameters"])
    del selected
    result = {"status": "debug_completed" if manifest["debug"] else "completed", "debug": manifest["debug"],
              "updates": cursor["updates"], "epochs": cursor["epoch"], "selected": best, "checks": checks,
              "test_evaluated": False}
    if not manifest["debug"]:
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
