"""Bounded full-story training of a small readout and shared fast-weight rule.

All arm selections are locked before any test forward pass. Padding freezes both
neural activity and fast synaptic state; the two targets are never model inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pair_decoder import file_sha256, write_json


ARMS = ("frozen", "fixed_fast", "learned_fast")
CALIBRATION_SOURCES = ("plastic_probe.py", "plastic_brain.py", "pair_brain.py", "brain.py", "pair_extract.py")
PROBE_CHECKS = ("frozen_graph_unchanged", "finite_loss", "nonzero_fast_state", "fast_state_affects_features",
                "finite_rule_gradients", "nonzero_rule_gradient", "rule_parameters_changed")
INTERVENTION_CAVEAT = ("Destructive state interventions also damage normal dynamics. "
    "A change in forecast loss does not by itself prove a distinct memory mechanism.")


class PlasticLanguageModel(nn.Module):
    """The full circuit and the only external trainable computation: one affine head."""

    def __init__(self, brain, vocabulary_size, device="cpu"):
        super().__init__()
        self.brain = brain
        self.plasticity_enabled = True
        self.readout = nn.Linear(256, 2 * vocabulary_size, device=device)

    def initial_state(self, batch):
        return self.brain.initial_state(batch)

    def step(self, previous, current, state, plasticity_override=None):
        if plasticity_override is None:
            plasticity_override = self.plasticity_enabled
        return self.brain.step(previous, current, state, plasticity_override=plasticity_override)

    def reset_activity(self, state):
        return self.brain.reset_activity(state)

    def reset_fast(self, state):
        return self.brain.reset_fast(state)

    def detach_state(self, state):
        return self.brain.detach_state(state)

    def set_activity_scales(self, pre, post):
        return self.brain.set_activity_scales(pre, post)


@dataclass
class StoryBatch:
    previous: torch.Tensor
    current: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    active: torch.Tensor
    lengths: torch.Tensor
    story_ids: Sequence[str]


def causal_rows(story: Dict[str, Any]):
    """Same shift/mask contract as pair_extract.story_rows, without native imports."""
    ids, masks = story["word_ids"], story["target_mask"]
    if not ids or ids[0] != 2 or len(ids) != len(masks) or len(story["words"]) > 128:
        raise ValueError("Require BOS, aligned masks, and at most 128 lexical words")
    for p in range(len(ids) - 1):
        if p > 128 or ids[p] == 3 or not masks[p + 1]:
            raise ValueError("Invalid recurrent position or interior masked target")
        second = p + 2 < len(ids) and bool(masks[p + 2]) and ids[p + 1] != 3
        yield (ids[p - 1] if p else 2, ids[p], ids[p + 1], ids[p + 2] if second else 0, second)


def validate_dataset(dataset):
    vocabulary = dataset["vocabulary"]
    if vocabulary[:4] != ["<pad>", "<unk>", "<bos>", "<eos>"] or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Invalid vocabulary or reserved IDs")
    seen = set()
    for split in ("train", "val", "test"):
        if not dataset["splits"][split]:
            raise ValueError("Empty dataset split")
        for story in dataset["splits"][split]:
            if str(story["id"]) in seen:
                raise ValueError("Story IDs overlap within/across splits")
            seen.add(str(story["id"]))
            ids = story["word_ids"]
            if (len(ids) < 2 or any(type(i) is not int or not 0 <= i < len(vocabulary) for i in ids)
                    or any(i in (0, 2) for i in ids[1:]) or 3 in ids[1:-1]
                    or len(ids) != 1 + len(story["words"]) + int(ids[-1] == 3)):
                raise ValueError("Malformed bounded story token IDs")
            list(causal_rows(story))


def collate_stories(stories, device="cpu") -> StoryBatch:
    if not stories:
        raise ValueError("Cannot batch zero stories")
    rows = [list(causal_rows(story)) for story in stories]
    lengths = [len(row) for row in rows]
    if not min(lengths):
        raise ValueError("Each story needs a forecast")
    shape = (len(stories), max(lengths))
    previous = np.full(shape, 2, np.int64)
    current = np.full(shape, 2, np.int64)
    targets = np.zeros(shape + (2,), np.int64)
    mask = np.zeros(shape + (2,), bool)
    active = np.zeros(shape, bool)
    for i, story_rows in enumerate(rows):
        for t, (prev, word, first, second, valid_second) in enumerate(story_rows):
            previous[i, t], current[i, t] = prev, word
            targets[i, t] = first, second
            mask[i, t] = True, valid_second
            active[i, t] = True
    return StoryBatch(*(torch.as_tensor(a, device=device) for a in
        (previous, current, targets, mask, active, np.array(lengths, np.int64))),
        story_ids=[str(s["id"]) for s in stories])


def _masked_state(old, new, active):
    return type(old)(h=torch.where(active[:, None], new.h, old.h),
                     fast=torch.where(active[:, None], new.fast, old.fast))


def rollout_batch(model, batch, feature_mean, feature_std, *, plasticity_override=None,
                  intervention=None):
    """One unbroken autograd graph through a full bounded story batch."""
    if intervention not in (None, "reset_activity", "reset_fast"):
        raise ValueError("Unknown destructive intervention")
    state = model.initial_state(len(batch.story_ids))
    logits = []
    for t in range(batch.current.shape[1]):
        if intervention is not None:
            at_midpoint = (batch.lengths // 2 == t) & (batch.lengths > 1)
            reset = getattr(model, intervention)(state)
            state = _masked_state(state, reset, at_midpoint)
        features, candidate = model.step(batch.previous[:, t], batch.current[:, t], state,
                                        plasticity_override=plasticity_override)
        state = _masked_state(state, candidate, batch.active[:, t])
        scaled = (features - feature_mean) / feature_std
        logits.append(model.readout(scaled).reshape(len(batch.story_ids), 2, -1))
    logits = torch.stack(logits, dim=1)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             batch.targets.reshape(-1), reduction="none").reshape_as(batch.targets)
    selected = losses[batch.target_mask]
    if not bool(torch.isfinite(selected).all()):
        raise RuntimeError("Nonfinite valid forecast loss")
    return selected.mean(), logits, losses, state


def configure_arm(model, arm, head_lr=.001, rule_lr=.003, weight_decay=.0001):
    if arm not in ARMS:
        raise ValueError("Unknown plasticity arm")
    model.plasticity_enabled = arm != "frozen"
    head, rule = [], []
    for name, parameter in model.named_parameters():
        is_head = name.startswith("readout.")
        parameter.requires_grad_(is_head or arm == "learned_fast")
        parameter.grad = None
        (head if is_head else rule).append(parameter)
    if not isinstance(model.readout, nn.Linear) or model.readout.in_features != 256:
        raise ValueError("Require one linear 256-feature readout")
    if sum(p.numel() for p in rule) != 20:
        raise ValueError("Only the declared twenty shared rule parameters may accompany the readout")
    if model.readout.out_features % 2:
        raise ValueError("Require exactly two vocabulary heads")
    groups = [{"params": head, "lr": head_lr, "weight_decay": weight_decay, "name": "head"}]
    if arm == "learned_fast":
        groups.append({"params": rule, "lr": rule_lr, "weight_decay": 0., "name": "rule"})
    optimizer = torch.optim.AdamW(groups)
    counts = {"readout": sum(p.numel() for p in head), "shared_rule": sum(p.numel() for p in rule),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return optimizer, counts


def _batch_records(batch, logits, losses):
    active = batch.active.detach().cpu().numpy()
    mask = batch.target_mask.detach().cpu().numpy()[active]
    loss = losses.detach().cpu().double().numpy()[active]
    prediction = logits.detach().argmax(-1).cpu().numpy()[active]
    loss[~mask], prediction[~mask] = np.nan, -1
    positions = np.broadcast_to(np.arange(active.shape[1]), active.shape)[active]
    mids = (batch.lengths.detach().cpu().numpy() // 2)[:, None]
    return {"losses": loss, "predictions": prediction,
        "y": batch.targets.detach().cpu().numpy()[active], "target_mask": mask,
        "previous_ids": batch.previous.detach().cpu().numpy()[active],
        "current_ids": batch.current.detach().cpu().numpy()[active], "positions": positions,
        "story_ids": np.broadcast_to(np.array(batch.story_ids)[:, None], active.shape)[active],
        "second_half": (np.broadcast_to(np.arange(active.shape[1]), active.shape) >= mids)[active]}


def _combine(records):
    return {key: np.concatenate([record[key] for record in records]) for key in records[0]}


def _selected_metrics(losses, predictions, targets, mask):
    count = int(mask.sum())
    ce = float(losses[mask].mean()) if count else None
    known = mask & (targets >= 4)
    return {"examples": count, "cross_entropy": ce,
        "exp_cross_entropy": math.exp(ce) if ce is not None and ce < 709 else None,
        "accuracy": float((predictions[mask] == targets[mask]).mean()) if count else None,
        "known_lexical_examples": int(known.sum()),
        "known_lexical_cross_entropy": float(losses[known].mean()) if known.any() else None,
        "known_lexical_accuracy": float((predictions[known] == targets[known]).mean()) if known.any() else None}


def _summarize_rows(data):
    losses, predictions, targets, mask = (data[k] for k in ("losses", "predictions", "y", "target_mask"))
    horizons = {"next_" + str(h + 1): _selected_metrics(losses[:, h], predictions[:, h], targets[:, h], mask[:, h])
                for h in range(2)}
    horizons["next_1"]["perplexity"] = horizons["next_1"]["exp_cross_entropy"]
    horizons["next_1"]["interpretation"] = "Ordinary causal next-word CE and perplexity."
    horizons["next_2"]["interpretation"] = "Second unseen word predicted from the same state; no true first future word supplied."
    average = _selected_metrics(losses, predictions, targets, mask)
    average["interpretation"] = "Mean CE per valid overlapping forecast, NOT standard sequence perplexity."
    complete = mask.all(axis=1)
    exact = (predictions == targets).all(axis=1)
    return {"horizons": horizons, "forecast_average": average,
            "complete_pair": {"examples": int(complete.sum()),
                "exact_accuracy": float(exact[complete].mean()) if complete.any() else None}}


def _summarize(data):
    result = _summarize_rows(data)
    second = {key: value[data["second_half"]] for key, value in data.items()}
    result["second_half"] = _summarize_rows(second)
    return result


@torch.no_grad()
def evaluate_stories(model, stories, feature_mean, feature_std, *, arm, batch_size=8,
                     intervention=None):
    model.eval()
    records = []
    for start in range(0, len(stories), batch_size):
        batch = collate_stories(stories[start:start + batch_size], feature_mean.device)
        _, logits, losses, _ = rollout_batch(model, batch, feature_mean, feature_std,
            plasticity_override=arm != "frozen", intervention=intervention)
        records.append(_batch_records(batch, logits, losses))
    arrays = _combine(records)
    return _summarize(arrays), arrays


def _source_hashes():
    names = ("plastic_train.py", "plastic_brain.py", "metal_sparse.py", "brain.py", "pair_brain.py", "pair_extract.py", "pair_decoder.py", "decoder.py")
    return {name: file_sha256(Path(__file__).with_name(name)) for name in names}


def _graph_paths(graph):
    graph = Path(graph)
    graph_file = graph / "graph.npz" if graph.is_dir() else graph
    return graph_file, graph_file.with_name("metadata.json")


def load_calibration(calibration, graph, dataset_path, dataset):
    path = Path(calibration)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    graph_file, graph_meta = _graph_paths(graph)
    identities = {"graph_sha256": file_sha256(graph_file),
                  "graph_metadata_sha256": file_sha256(graph_meta),
                  "dataset_sha256": file_sha256(dataset_path),
                  "calibration_npz_sha256": file_sha256(path)}
    if any(metadata.get(key) != value for key, value in identities.items()):
        raise ValueError("Calibration graph/dataset/array provenance mismatch")
    training_ids = {str(story["id"]) for story in dataset["splits"]["train"]}
    supplied = metadata.get("story_ids", [])
    if (metadata.get("train_only") is not True or not supplied
            or len(set(map(str, supplied))) != len(supplied)
            or not set(map(str, supplied)).issubset(training_ids)):
        raise ValueError("Calibration must use identified training stories only")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in ("feature_mean", "feature_std", "pre_scale", "post_scale")}
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Calibration arrays must be finite float32")
    if arrays["feature_mean"].shape != (256,) or arrays["feature_std"].shape != (256,):
        raise ValueError("Expected 256-feature calibration")
    if (arrays["pre_scale"].ndim != 1 or arrays["post_scale"].shape != arrays["pre_scale"].shape
            or any(np.any(arrays[k] <= 0) for k in ("feature_std", "pre_scale", "post_scale"))):
        raise ValueError("Invalid positive activity/feature scales")
    kwargs = {key: metadata[key] for key in ("global_scale", "internal_steps", "leak", "input_gain", "input_center")}
    if any(not math.isfinite(float(value)) for value in kwargs.values()):
        raise ValueError("Nonfinite calibrated dynamics")
    if (kwargs["global_scale"] <= 0 or int(kwargs["internal_steps"]) != kwargs["internal_steps"]
            or kwargs["internal_steps"] < 1 or not 0 < kwargs["leak"] <= 1):
        raise ValueError("Invalid calibrated dynamics; fractional step counts cannot be rounded")
    sources = metadata.get("source_sha256", {})
    if not isinstance(sources, dict) or not set(CALIBRATION_SOURCES).issubset(sources):
        raise ValueError("Calibration is missing required source identities")
    for name, expected in sources.items():
        if Path(name).name != name:
            raise ValueError("Calibration source names must be local module basenames")
        current = Path(__file__).with_name(name)
        if not current.is_file() or file_sha256(current) != expected:
            raise ValueError("Calibration source changed: " + name)
    probe_path = path.with_name("probe.json")
    if (metadata.get("probe_passed") is not True or metadata.get("probe_file") != "probe.json"
            or not probe_path.is_file() or file_sha256(probe_path) != metadata.get("probe_sha256")):
        raise ValueError("Calibration requires a checksum-bound passed probe receipt")
    probe = json.loads(probe_path.read_text())
    checks = probe.get("checks", {})
    if (not isinstance(checks, dict) or not set(PROBE_CHECKS).issubset(checks)
            or any(value is not True for value in checks.values())):
        raise ValueError("Calibration probe wiring checks did not all pass")
    if (probe.get("config") != kwargs
            or probe.get("calibration_npz_sha256") != identities["calibration_npz_sha256"]
            or any(probe.get("provenance", {}).get(key) != metadata.get(key) for key in
                   ("graph_sha256", "graph_metadata_sha256", "dataset_sha256", "train_only", "story_ids", "source_sha256"))):
        raise ValueError("Calibration and passed probe provenance/config do not match")
    provenance = {**identities, "calibration_metadata_sha256": file_sha256(metadata_path),
                  "calibration_source_sha256": sources, "probe_sha256": metadata["probe_sha256"],
                  "source_sha256": _source_hashes()}
    return arrays, metadata, kwargs, provenance


def save_checkpoint(path, model, scales, config, vocabulary, epoch, validation_ce):
    """Save parameters and small calibration only; never serialize fixed graph buffers."""
    path = Path(path)
    parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    payload = {"schema": 1, "parameters": parameters,
        "scales": {key: torch.as_tensor(value).detach().cpu().clone() for key, value in scales.items()},
        "config": config, "vocabulary": list(vocabulary), "epoch": int(epoch),
        "validation_head1_cross_entropy": float(validation_ce)}
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha256(path)


def restore_parameters(model, payload):
    expected = dict(model.named_parameters())
    if set(expected) != set(payload["parameters"]):
        raise ValueError("Compact checkpoint parameter names differ")
    with torch.no_grad():
        for name, value in payload["parameters"].items():
            if expected[name].shape != value.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid compact checkpoint parameter: " + name)
            expected[name].copy_(value)
    model.set_activity_scales(payload["scales"]["pre_scale"], payload["scales"]["post_scale"])


def load_checkpoint(path, graph, device="cpu", *, model_factory=None, verify_sources=True):
    if model_factory is None:
        from .plastic_brain import PlasticBrain
        model_factory = PlasticBrain
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1:
        raise ValueError("Unknown compact checkpoint schema")
    provenance = payload["config"]["provenance"]
    graph_file, graph_meta = _graph_paths(graph)
    if (file_sha256(graph_file) != provenance["graph_sha256"]
            or file_sha256(graph_meta) != provenance["graph_metadata_sha256"]):
        raise ValueError("Checkpoint graph provenance mismatch")
    if verify_sources and provenance["source_sha256"] != _source_hashes():
        raise ValueError("Checkpoint implementation differs from current source")
    brain = model_factory(graph_path=graph, vocab_size=len(payload["vocabulary"]),
        device=device, **payload["config"]["model_kwargs"])
    model = PlasticLanguageModel(brain, len(payload["vocabulary"]), device)
    restore_parameters(model, payload)
    arm = payload["config"]["arm"]
    if arm not in ARMS:
        raise ValueError("Checkpoint arm is unknown")
    model.plasticity_enabled = arm != "frozen"
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("readout.") or arm == "learned_fast")
    model.eval()
    return model, payload


def _log(output, value):
    text = json.dumps(value, allow_nan=False)
    print(text, flush=True)
    with (Path(output) / "events.jsonl").open("a") as stream:
        stream.write(text + "\n")


def _rule_snapshot(model):
    return {name: p.detach().cpu().tolist() for name, p in model.named_parameters()
            if not name.startswith("readout.")}


def _rule_gradient_norm(model):
    terms = [p.grad.detach().float().square().sum() for name, p in model.named_parameters()
             if not name.startswith("readout.") and p.grad is not None]
    return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.


def train(dataset_path, graph, calibration, output, *, device="cuda", arms=ARMS,
          epochs=10, patience=4, batch_size=8, seed=0, head_lr=.001, rule_lr=.003,
          weight_decay=.0001, threads=4, gradient_clip=1., model_factory=None, strict_full_graph=True):
    if (not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms)
            or epochs < 1 or patience < 1 or batch_size < 1 or threads < 1
            or not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("Invalid bounded training configuration")
    torch.set_num_threads(threads)
    if model_factory is None:
        from .plastic_brain import PlasticBrain
        model_factory = PlasticBrain
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("Output already contains a run; use a fresh directory")
    dataset = json.loads(Path(dataset_path).read_text())
    validate_dataset(dataset)
    scales, calibration_meta, dynamics, provenance = load_calibration(calibration, graph, dataset_path, dataset)
    vocabulary = dataset["vocabulary"]
    model_kwargs = {**dynamics, "internal_steps": int(dynamics["internal_steps"]),
                    "plasticity": True, "feature_dim": 256, "strict_full_graph": strict_full_graph}
    protocol = {"schema": 1, "arms": list(arms), "seed": seed, "epochs": epochs,
        "patience": patience, "batch_size": batch_size, "head_lr": head_lr,
        "rule_lr": rule_lr, "weight_decay_head_only": weight_decay,
        "threads": threads, "gradient_clip": gradient_clip, "device": str(device),
        "torch_version": str(torch.__version__),
        "model_kwargs": model_kwargs, "provenance": provenance,
        "calibration_story_ids": calibration_meta["story_ids"],
        "selection": "minimum validation head1 CE independently per arm; all arms frozen before test",
        "targets": "ordered previous,current -> next1,next2; never feed either future target to same-row model",
        "bptt": "full bounded story; no detach inside story; both states frozen on batch padding",
        "scoring": "two overlapping horizons; average forecast CE is NOT standard sequence perplexity"}
    write_json(output / "protocol.json", protocol)
    feature_mean = torch.as_tensor(scales["feature_mean"], device=device)
    feature_std = torch.as_tensor(scales["feature_std"], device=device)
    # One graph instance is reused across arms. Restore every learned parameter,
    # including the readout initialization, before constructing each optimizer.
    torch.manual_seed(seed)
    brain = model_factory(graph_path=graph, vocab_size=len(vocabulary), device=device, **model_kwargs)
    model = PlasticLanguageModel(brain, len(vocabulary), device)
    model.set_activity_scales(scales["pre_scale"], scales["post_scale"])
    protocol["brain"] = brain.metadata()
    protocol["initial_frozen_graph_sha256"] = brain.verify_frozen()
    write_json(output / "protocol.json", protocol)
    initial = {"parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
               "scales": scales}
    selections = {}
    started = time.perf_counter()
    for arm in arms:
        directory = output / arm
        directory.mkdir()
        restore_parameters(model, initial)
        optimizer, counts = configure_arm(model, arm, head_lr, rule_lr, weight_decay)
        if counts["readout"] != 2 * len(vocabulary) * 257:
            raise ValueError("Readout parameter count differs from two independent linear heads")
        if strict_full_graph and len(vocabulary) == 1024 and counts["readout"] != 526336:
            raise ValueError("Expected exact 526336-parameter decoder")
        config = {**protocol, "arm": arm, "parameters": counts}
        rng = np.random.default_rng(seed)
        best, best_epoch, step, history = math.inf, 0, 0, []
        _log(output, {"event": "arm_start", "arm": arm, "parameters": counts, "rule": _rule_snapshot(model)})
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(dataset["splits"]["train"]))
            records, gradient_norms, total_gradient_norms = [], [], []
            epoch_start = time.perf_counter()
            for start in range(0, len(order), batch_size):
                stories = [dataset["splits"]["train"][i] for i in order[start:start + batch_size]]
                batch = collate_stories(stories, device)
                optimizer.zero_grad(set_to_none=True)
                loss, logits, losses, state = rollout_batch(model, batch, feature_mean, feature_std,
                    plasticity_override=arm != "frozen")
                loss.backward()
                norm = _rule_gradient_norm(model)
                if not math.isfinite(norm) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                                                for p in model.parameters()):
                    raise RuntimeError("Nonfinite optimizer gradient")
                total_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], gradient_clip,
                    error_if_nonfinite=True).detach().cpu())
                optimizer.step()
                step += 1
                gradient_norms.append(norm)
                total_gradient_norms.append(total_norm)
                records.append(_batch_records(batch, logits, losses))
                if step == 1:
                    _log(output, {"event": "first_update", "arm": arm, "global_step": 1,
                        "loss": float(loss.detach().cpu()), "rule_gradient_norm": norm,
                        "total_gradient_norm_before_clip": total_norm, "gradient_clip": gradient_clip,
                        "seconds": time.perf_counter() - epoch_start})
                del loss, logits, losses, state, batch
            train_metrics = _summarize(_combine(records))
            validation, _ = evaluate_stories(model, dataset["splits"]["val"], feature_mean, feature_std,
                                              arm=arm, batch_size=batch_size)
            score = validation["horizons"]["next_1"]["cross_entropy"]
            if not math.isfinite(score):
                raise RuntimeError("Nonfinite validation selection metric")
            latest_hash = save_checkpoint(directory / "latest.pt", model, scales, config, vocabulary, epoch, score)
            if score < best:
                best, best_epoch = score, epoch
                best_hash = save_checkpoint(directory / "best.pt", model, scales, config, vocabulary, epoch, score)
            row = {"event": "epoch", "arm": arm, "epoch": epoch, "global_step": step,
                "train": train_metrics, "validation": validation,
                "seconds": time.perf_counter() - epoch_start,
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "rule_gradient_norm_mean": float(np.mean(gradient_norms)),
                "rule_gradient_norm_max": float(max(gradient_norms)), "rule": _rule_snapshot(model),
                "rule_gradient_norm_measurement": "before global gradient clipping",
                "total_gradient_norm_before_clip_mean": float(np.mean(total_gradient_norms)),
                "clipped_update_fraction": float(np.mean(np.array(total_gradient_norms) > gradient_clip)),
                "best_epoch": best_epoch, "best_validation_head1_ce": best,
                "latest_sha256": latest_hash, "best_sha256": best_hash}
            history.append(row)
            write_json(directory / "history.json", history)
            _log(output, row)
            if epoch - best_epoch >= patience:
                break
        selections[arm] = {"epoch": best_epoch, "validation_head1_cross_entropy": best,
                           "checkpoint_sha256": best_hash, "checkpoint": str(directory / "best.pt")}
        write_json(directory / "selection.json", selections[arm])
        del optimizer
    # This artifact is the boundary: no calls involving test stories precede it.
    write_json(output / "selections-locked.json", {"selections": selections,
        "test_forward_started": False, "provenance": provenance})
    metrics = {}
    for arm in arms:
        directory = output / arm
        if file_sha256(directory / "best.pt") != selections[arm]["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after locking")
        payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        restore_parameters(model, payload)
        result = {"selection": selections[arm], "interventions": {}}
        for split in ("train", "val", "test"):
            summary, arrays = evaluate_stories(model, dataset["splits"][split], feature_mean, feature_std,
                                               arm=arm, batch_size=batch_size)
            result[split] = summary
            if split == "test":
                np.savez_compressed(directory / "test-predictions.npz", **arrays)
                baseline = arrays
        interventions = ("reset_fast", "reset_activity") if arm == "learned_fast" else (("reset_activity",) if arm == "frozen" else ())
        for intervention in interventions:
            summary, arrays = evaluate_stories(model, dataset["splits"]["test"], feature_mean, feature_std,
                arm=arm, batch_size=batch_size, intervention=intervention)
            if any(not np.array_equal(arrays[key], baseline[key]) for key in
                   ("story_ids", "positions", "y", "target_mask", "second_half")):
                raise RuntimeError("Intervention and baseline forecast rows do not align")
            result["interventions"][intervention] = {"metrics": summary,
                "unperturbed_second_half": result["test"]["second_half"], "caveat": INTERVENTION_CAVEAT,
                "timing": "reset before position floor(number_of_forecast_rows/2), separately per story"}
            np.savez_compressed(directory / ("test-" + intervention + ".npz"), **arrays)
        write_json(directory / "metrics.json", result)
        metrics[arm] = result
        _log(output, {"event": "arm_evaluated", "arm": arm, "test": result["test"]})
    final_graph = brain.verify_frozen()
    if final_graph != protocol["initial_frozen_graph_sha256"]:
        raise RuntimeError("Fixed graph changed during training")
    final = {"protocol": protocol, "arms": metrics, "elapsed_seconds": time.perf_counter() - started,
             "final_frozen_graph_sha256": final_graph,
             "selections_locked_sha256": file_sha256(output / "selections-locked.json")}
    write_json(output / "metrics.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "graph", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.)
    args = parser.parse_args()
    train(args.dataset, args.graph, args.calibration, args.output, device=args.device,
          arms=args.arms, epochs=args.epochs, patience=args.patience,
          batch_size=args.batch_size, seed=args.seed, threads=args.threads, gradient_clip=args.gradient_clip)


if __name__ == "__main__":
    main()
