"""Train a top-K action fly selector with train-only n-gram proposals.

The brain is frozen. All training/calibration proposals exclude their entire
own story. Validation proposals use the whole training split. Missing top-K
targets never enter the candidate set and remain misses in overall accuracy.
The test split is never consulted or evaluated.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .action_candidates import ActionCandidates, _story_rows
from .action_model import CandidateActionBrain, FlyActionSelector
from .pair_decoder import file_sha256


ARM = "frozen_action"
SOURCES = ("action_train.py", "action_model.py", "action_candidates.py", "plastic_brain.py",
           "metal_sparse.py", "brain.py", "pair_brain.py", "pair_decoder.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCES}


def subset_identity(stories, dataset_sha256, split="val"):
    identity = {"dataset_sha256": dataset_sha256, "split": split,
                "story_ids": [str(story["id"]) for story in stories]}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "subset_sha256": hashlib.sha256(encoded).hexdigest()}


def validate_training_validation(dataset, top_k=10):
    """Validate the consumed splits without retrieving a test split."""
    vocabulary = dataset["vocabulary"]
    if (type(top_k) is not int or top_k < 1 or top_k > len(vocabulary) - 2):
        raise ValueError("top_k must fit vocabulary excluding PAD/BOS")
    if (len(vocabulary) < 5 or vocabulary[:4] != ["<pad>", "<unk>", "<bos>", "<eos>"]
            or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("Require unique vocabulary with PAD/UNK/BOS/EOS and enough allowed actions")
    seen = set()
    for name in ("train", "val"):
        stories = dataset["splits"][name]
        if not stories:
            raise ValueError(name + " split must be nonempty")
        for story in stories:
            identifier = str(story["id"])
            if identifier in seen:
                raise ValueError("Training/validation IDs overlap or contain duplicates")
            seen.add(identifier)
            if not len(_story_rows(story, len(vocabulary))):
                raise ValueError("Every story requires a forecast target")


@dataclass
class ActionBatch:
    previous: torch.Tensor
    current: torch.Tensor
    candidates: torch.Tensor
    probabilities: torch.Tensor
    targets: torch.Tensor
    active: torch.Tensor
    story_ids: tuple


def collate_actions(stories, proposals, device="cpu", exclude_own_story=False):
    if not stories:
        raise ValueError("Cannot collate an empty story batch")
    rows = [_story_rows(story, proposals.vocabulary_size) for story in stories]
    if not min(len(row) for row in rows):
        raise ValueError("Every story requires at least one target")
    shape = (len(rows), max(len(row) for row in rows))
    previous, current = np.full(shape, 2, np.int64), np.full(shape, 2, np.int64)
    targets, active = np.zeros(shape, np.int64), np.zeros(shape, bool)
    # Valid placeholders remain mandatory for inactive rows; they are never
    # scored and their states are frozen by the model's active mask.
    fallback_ids, fallback_prob = proposals.candidates(2, 2)
    candidates = np.broadcast_to(fallback_ids, shape + (proposals.top_k,)).copy()
    probabilities = np.broadcast_to(fallback_prob.astype(np.float32), shape + (proposals.top_k,)).copy()
    for batch_index, (story, story_rows) in enumerate(zip(stories, rows)):
        excluded = str(story["id"]) if exclude_own_story else None
        for position, (prev, word, target) in enumerate(story_rows):
            ids, probs = proposals.candidates(int(prev), int(word), exclude_story_id=excluded)
            previous[batch_index, position], current[batch_index, position] = prev, word
            targets[batch_index, position] = target
            active[batch_index, position] = True
            candidates[batch_index, position], probabilities[batch_index, position] = ids, probs
    tensors = [torch.as_tensor(value, device=device) for value in
               (previous, current, candidates, probabilities, targets, active)]
    return ActionBatch(*tensors, tuple(str(story["id"]) for story in stories))


def extract_features(brain, batch):
    """Carry neural state over every word, freeze padding, and reset per story."""
    state = brain.initial_state(len(batch.story_ids))
    features = []
    with torch.no_grad():
        for position in range(batch.previous.shape[1]):
            values, state = brain.step(batch.previous[:, position], batch.current[:, position],
                                       batch.candidates[:, position], batch.probabilities[:, position],
                                       state, active=batch.active[:, position])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError("Nonfinite candidate-aware brain features")
            features.append(values)
    return torch.stack(features, dim=1)


def batch_logits(model, batch):
    features = extract_features(model.brain, batch)
    logits = batch.probabilities.log() + model.correction(features)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Nonfinite candidate selection logits")
    return logits


def selection_loss(logits, batch):
    matches = batch.candidates == batch.targets[..., None]
    covered = matches.any(dim=-1) & batch.active
    target_actions = matches.to(torch.int64).argmax(dim=-1)
    count = int(covered.sum().item())
    if not count:
        return logits.sum() * 0., 0
    return F.cross_entropy(logits[covered], target_actions[covered]), count


class ActionMetrics:
    def __init__(self):
        self.target_count = self.correct = self.baseline_correct = self.covered = 0
        self.fixes = self.regressions = self.known_count = self.known_correct = 0
        self.conditional_loss_sum = 0.
        self.top_k = None

    def update(self, logits, batch):
        # Accumulate exact counts; report current-checkpoint metrics, never a
        # running mean of different checkpoints or a coverage-filtered accuracy.
        top_k = batch.candidates.shape[-1]
        if self.top_k is not None and top_k != self.top_k:
            raise ValueError("Cannot mix action counts in one metric accumulator")
        if logits.shape != batch.candidates.shape or batch.probabilities.shape != batch.candidates.shape:
            raise ValueError("Logits, candidate IDs and probabilities must have matching shapes")
        self.top_k = top_k
        logits = logits.detach().cpu().reshape(-1, top_k)
        candidates = batch.candidates.detach().cpu().reshape(-1, top_k)
        probabilities = batch.probabilities.detach().cpu().reshape(-1, top_k)
        targets, active = batch.targets.detach().cpu().flatten(), batch.active.detach().cpu().flatten()
        logits, candidates, probabilities, targets = (value[active] for value in
                                                       (logits, candidates, probabilities, targets))
        if not len(targets):
            return
        chosen = candidates.gather(1, logits.argmax(-1)[:, None]).flatten()
        prior = candidates.gather(1, probabilities.argmax(-1)[:, None]).flatten()
        correct, base = chosen == targets, prior == targets
        matches = candidates == targets[:, None]
        covered = matches.any(-1)
        known = targets >= 4
        self.target_count += len(targets)
        self.correct += int(correct.sum())
        self.baseline_correct += int(base.sum())
        self.covered += int(covered.sum())
        self.fixes += int((correct & ~base).sum())
        self.regressions += int((~correct & base).sum())
        self.known_count += int(known.sum())
        self.known_correct += int((known & correct).sum())
        if bool(covered.any()):
            ranks = matches.to(torch.int64).argmax(-1)
            self.conditional_loss_sum += float(F.cross_entropy(logits[covered], ranks[covered], reduction="sum"))

    def summary(self):
        if not self.target_count:
            raise ValueError("Cannot summarize zero target positions")
        if self.correct != self.baseline_correct + self.fixes - self.regressions:
            raise RuntimeError("Prediction accounting mismatch")
        return {
            "target_count": self.target_count, "top_k": self.top_k, "accuracy": self.correct / self.target_count,
            "baseline_accuracy": self.baseline_correct / self.target_count,
            "topk_coverage": self.covered / self.target_count,
            "conditional_accuracy": self.correct / self.covered if self.covered else None,
            "fixes": self.fixes, "regressions": self.regressions,
            "conditional_cross_entropy": self.conditional_loss_sum / self.covered if self.covered else None,
            "known_word_accuracy": self.known_correct / self.known_count if self.known_count else None,
            "known_word_target_count": self.known_count, "covered_target_count": self.covered,
            "correct_target_count": self.correct, "baseline_correct_target_count": self.baseline_correct,
        }


def evaluate(model, stories, proposals, batch_size=8, baseline_only=False, progress=None):
    """Evaluate current weights. No target changes the sensory candidate set."""
    metrics = ActionMetrics()
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    if model is not None:
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(stories), batch_size):
                device = "cpu" if baseline_only else model.brain.device
                batch = collate_actions(stories[start:start + batch_size], proposals, device=device)
                logits = batch.probabilities.log() if baseline_only else batch_logits(model, batch)
                metrics.update(logits, batch)
                if progress:
                    progress(min(start + batch_size, len(stories)), len(stories))
    finally:
        for module, mode in modes:
            module.training = mode
    return metrics.summary()


def calibrate_features(brain, stories, proposals, batch_size=8, progress=None):
    """Fresh moments from training-only, whole-story-excluded sensory features."""
    count = 0
    total, total_square = np.zeros(256, np.float64), np.zeros(256, np.float64)
    for start in range(0, len(stories), batch_size):
        batch = collate_actions(stories[start:start + batch_size], proposals,
                                device=brain.device, exclude_own_story=True)
        values = extract_features(brain, batch)[batch.active].cpu().numpy().astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_square += (values * values).sum(axis=0)
        if progress:
            progress(min(start + batch_size, len(stories)), len(stories))
    if not count:
        raise ValueError("Calibration has no valid positions")
    mean = total / count
    raw_std = np.sqrt(np.maximum(0., total_square / count - mean * mean))
    floor = max(float(raw_std.max()) * 1e-4, 1e-6)
    std = np.maximum(raw_std, floor)
    finite = bool(np.isfinite(mean).all() and np.isfinite(std).all())
    receipt = {
        "schema": 1, "kind": "action_selection_calibration", "story_count": len(stories),
        "top_k": proposals.top_k,
        "story_ids": [str(story["id"]) for story in stories], "target_count": count,
        "split": "train", "proposal_exclusion": "entire current training story",
        "finite_features": finite, "feature_varying_dimensions": int((raw_std > 1e-10).sum()),
        "feature_std_max": float(raw_std.max()), "feature_std_floor": floor,
        "timestamp_utc": utc_now(),
    }
    if not finite:
        raise RuntimeError("Nonfinite feature calibration")
    return mean.astype(np.float32), std.astype(np.float32), receipt


def candidate_effect_probe(brain, story, proposals, feature_std, word_limit=16):
    """Paired identity perturbation against exact replay noise, train-only."""
    batch = collate_actions([story], proposals, device=brain.device, exclude_own_story=True)
    length = min(word_limit, batch.previous.shape[1])
    batch = ActionBatch(*(getattr(batch, name)[:, :length] for name in
                          ("previous", "current", "candidates", "probabilities", "targets", "active")),
                        batch.story_ids)
    changed_ids = batch.candidates.clone()
    for position in range(length):
        used = set(changed_ids[0, position].cpu().tolist())
        replacement = next((token for token in range(1, proposals.vocabulary_size)
                            if token != 2 and token not in used), None)
        if replacement is None:
            # Tiny test vocabularies may contain exactly K allowed actions.
            changed_ids[0, position] = changed_ids[0, position].roll(1)
        else:
            changed_ids[0, position, 0] = replacement
    changed = ActionBatch(batch.previous, batch.current, changed_ids, batch.probabilities,
                          batch.targets, batch.active, batch.story_ids)
    first, repeat = extract_features(brain, batch), extract_features(brain, batch)
    altered = extract_features(brain, changed)
    scale = torch.as_tensor(feature_std, device=brain.device)
    repeat_max = float(((first - repeat) / scale).abs().max().cpu())
    effect_max = float(((first - altered) / scale).abs().max().cpu())
    threshold = max(10 * repeat_max, 1e-6)
    return {"word_positions": length, "repeat_normalized_max": repeat_max,
            "candidate_identity_normalized_max": effect_max, "required_effect": threshold,
            "candidate_identity_affects_features": bool(math.isfinite(effect_max) and effect_max > threshold)}


class Recorder:
    def __init__(self, output, dataset_sha256, top_k):
        self.output, self.dataset_sha256 = Path(output), dataset_sha256
        self.top_k = top_k
        self.history = []

    def event(self, event, **values):
        row = {"schema": 1, "kind": "action_selection", "arm": ARM,
               "event": event, "top_k": self.top_k, "timestamp_utc": utc_now(), **values}
        line = json.dumps(row, allow_nan=False)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        return row

    def validation(self, scope, stories, metrics, step, epoch, **extras):
        if scope not in ("monitor_subset", "full_validation"):
            raise ValueError("Invalid validation scope")
        if metrics.get("top_k") != self.top_k:
            raise ValueError("Validation metrics action count differs from run")
        row = {"schema": 1, "kind": "action_selection", "arm": ARM,
               "top_k": self.top_k,
               "event": "validation_snapshot", "scope": scope, "phase": "validation",
               "global_step": step, "epoch": epoch, "story_count": len(stories),
               "timestamp_utc": utc_now(), **subset_identity(stories, self.dataset_sha256),
               **metrics, "selection_eligible": scope == "full_validation", **extras}
        self.history.append(row)
        path = self.output / "validation.jsonl"
        temporary = path.with_suffix(".jsonl.partial")
        temporary.write_text("".join(json.dumps(item, allow_nan=False) + "\n" for item in self.history))
        temporary.replace(path)
        self.event("validation_snapshot", **{k: v for k, v in row.items()
                                              if k not in ("event", "schema", "kind", "arm")})
        return row


def selection_improves(candidate, best):
    """Only full validation can select; ties retain the baseline/earlier model."""
    return (candidate["scope"] == "full_validation"
            and (best is None or candidate["accuracy"] > best["accuracy"]))


def save_checkpoint(path, model, protocol, step, epoch, role, metrics=None):
    payload = {
        "schema": 1, "kind": "action_selection", "arm": ARM, "checkpoint_role": role,
        "top_k": protocol["top_k"],
        "global_step": step, "epoch": epoch, "parameters": {
            "readout.weight": model.readout.weight.detach().cpu().clone(),
            "readout.bias": model.readout.bias.detach().cpu().clone()},
        "feature_mean": model.feature_mean.detach().cpu().clone(),
        "feature_std": model.feature_std.detach().cpu().clone(),
        "protocol": protocol, "validation": metrics,
        "inference_only": True, "training_resume_supported": False,
        "selection_eligible": role == "best_full_validation",
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha256(path)


def train(dataset_path, graph, output, *, device="mps", epochs=1, batch_size=8, lr=.001,
          seed=0, monitor_stories=128, monitor_every=64, progress_every=8,
          calibration_stories=32, threads=4, gradient_clip=1., global_scale=.002,
          internal_steps=8, leak=.5, input_high=.02, strict_full_graph=True,
          brain_factory=CandidateActionBrain, top_k=10):
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("monitor_stories", monitor_stories),
                        ("monitor_every", monitor_every), ("progress_every", progress_every),
                        ("calibration_stories", calibration_stories), ("threads", threads)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Learning rate and gradient clipping must be finite and positive")
    output = Path(output)
    allowed = {"launch.json", "process-status.json", "stdout.log"}
    if output.exists() and any(path.name not in allowed for path in output.iterdir()):
        raise ValueError("Refusing nonempty output directory containing prior experiment artifacts")
    dataset_path = Path(dataset_path)
    dataset = json.loads(dataset_path.read_text())
    validate_training_validation(dataset, top_k)
    training, validation = dataset["splits"]["train"], dataset["splits"]["val"]
    if monitor_stories > len(validation) or calibration_stories > len(training):
        raise ValueError("Requested monitor/calibration subset exceeds its split")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    dataset_sha = file_sha256(dataset_path)
    graph_file = Path(graph) / "graph.npz" if Path(graph).is_dir() else Path(graph)
    dynamics = {"global_scale": global_scale, "internal_steps": internal_steps, "top_k": top_k, "seed": seed,
                "leak": leak, "input_high": input_high, "strict_full_graph": strict_full_graph}
    protocol = {
        "schema": 1, "kind": "action_selection", "arm": ARM, "timestamp_utc": utc_now(),
        "top_k": top_k,
        "dataset_path": str(dataset_path.resolve()), "dataset_sha256": dataset_sha,
        "graph_path": str(graph_file.resolve()), "graph_sha256": file_sha256(graph_file),
        "source_sha256": source_hashes(), "vocabulary": dataset["vocabulary"],
        "train_story_count": len(training), "validation_story_count": len(validation),
        "batch_size": batch_size, "epochs": epochs, "lr": lr, "seed": seed,
        "threads": threads, "gradient_clip": gradient_clip, "device": str(device),
        "model_kwargs": dynamics, "trainable_parameters": 257 * top_k, "brain_trainable_parameters": 0,
        "training_proposals": "Full training counts minus the entire current story; excludes its future targets",
        "validation_proposals": "All training stories only; original top-%d probabilities" % top_k,
        "objective": "Conditional cross entropy only where target is in top %d" % top_k,
        "primary_metric": "All-position next-word accuracy; absent top-%d targets are misses" % top_k,
        "checkpoint_selection": "Strictly higher full-validation accuracy; zero-correction baseline eligible; ties keep earlier",
        "test_evaluated": False, "training_resume_supported": False,
        "monitor_every": monitor_every, "progress_every": progress_every,
        "monitor_stories": monitor_stories, "monitor_subset": subset_identity(validation[:monitor_stories], dataset_sha),
        "calibration_subset": subset_identity(training[:calibration_stories], dataset_sha, "train"),
        "calibration_stories": calibration_stories,
    }
    write_json(output / "protocol.json", protocol)
    recorder = Recorder(output, dataset_sha, top_k)
    started = time.perf_counter()
    recorder.event("phase", phase="building_proposals", global_step=0, epoch=0)
    proposals = ActionCandidates(training, len(dataset["vocabulary"]), top_k=top_k)
    recorder.event("phase", phase="loading_brain", global_step=0, epoch=0)
    brain = brain_factory(graph, len(dataset["vocabulary"]), device=device, **dynamics)
    protocol["brain"] = brain.metadata()
    write_json(output / "protocol.json", protocol)
    recorder.event("phase", phase="calibration", global_step=0, epoch=0)
    calibration_subset = training[:calibration_stories]
    mean, std, calibration = calibrate_features(brain, calibration_subset, proposals, batch_size,
        progress=lambda completed, total: recorder.event("progress", phase="calibration", global_step=0,
            epoch=0, stories_completed=completed, total_stories=total))
    calibration.update(candidate_effect_probe(brain, training[0], proposals, std))
    calibration.update(dataset_sha256=dataset_sha, graph_sha256=protocol["graph_sha256"],
                       source_sha256=protocol["source_sha256"],
                       subset_sha256=protocol["calibration_subset"]["subset_sha256"])
    calibration["gate_passed"] = bool(calibration["finite_features"]
        and calibration["candidate_identity_affects_features"]
        and (not strict_full_graph or calibration["feature_varying_dimensions"] == 256))
    write_json(output / "calibration.json", calibration)
    np.savez(output / "calibration.npz", feature_mean=mean, feature_std=std)
    if not calibration["gate_passed"]:
        raise RuntimeError("Candidate-aware calibration failed; inspect calibration.json")
    protocol["calibration_sha256"] = file_sha256(output / "calibration.npz")
    model = FlyActionSelector(brain, mean, std)
    protocol["model"] = model.metadata()
    write_json(output / "protocol.json", protocol)
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=lr)
    if model.trainable_parameter_count() != 257 * top_k:
        raise RuntimeError("Action head must have exactly %d trainable parameters" % (257 * top_k))
    total_steps = epochs * math.ceil(len(training) / batch_size)
    global_step = 0

    # At zero correction the count-only full result is exactly the model's
    # prediction. Avoid an unnecessary full-graph baseline validation pass.
    recorder.event("phase", phase="validation", scope="full_validation", global_step=0, epoch=0)
    baseline = evaluate(None, validation, proposals, batch_size, baseline_only=True)
    best = recorder.validation("full_validation", validation, baseline, 0, 0,
                               evaluation_method="count_model_exact_zero_correction")
    save_checkpoint(output / "best.pt", model, protocol, 0, 0, "best_full_validation", best)
    write_json(output / "selection.json", best)

    def monitor(epoch):
        snapshot_sha = save_checkpoint(output / "partial.pt", model, protocol,
                                       global_step, epoch, "rolling_partial")
        recorder.event("phase", phase="validation", scope="monitor_subset",
                       global_step=global_step, epoch=epoch, total_steps=total_steps)
        validation_started = time.perf_counter()
        metrics = evaluate(model, validation[:monitor_stories], proposals, batch_size)
        recorder.validation("monitor_subset", validation[:monitor_stories], metrics, global_step, epoch,
                            checkpoint_path="partial.pt", checkpoint_sha256=snapshot_sha,
                            evaluation_seconds=time.perf_counter() - validation_started,
                            evaluation_method="current_model_forward")

    monitor(0)
    order_rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = order_rng.permutation(len(training))
        train_metrics = ActionMetrics()
        recorder.event("phase", phase="training", epoch=epoch, global_step=global_step,
                       total_steps=total_steps, stories_completed=0, total_stories=len(training))
        model.train()
        for start in range(0, len(order), batch_size):
            stories = [training[int(index)] for index in order[start:start + batch_size]]
            batch = collate_actions(stories, proposals, device=brain.device, exclude_own_story=True)
            logits = batch_logits(model, batch)
            loss, covered_count = selection_loss(logits, batch)
            train_metrics.update(logits, batch)
            optimizer.zero_grad(set_to_none=True)
            if covered_count:
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite conditional training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.readout.parameters(), gradient_clip)
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Nonfinite action head gradient")
                optimizer.step()
                global_step += 1
            else:
                recorder.event("no_covered_targets", phase="training", epoch=epoch,
                               global_step=global_step, story_ids=list(batch.story_ids))
            completed = min(start + batch_size, len(training))
            if global_step in (1,) or global_step % progress_every == 0 or completed == len(training):
                recorder.event("progress", phase="training", epoch=epoch, global_step=global_step,
                    total_steps=total_steps, stories_completed=completed, total_stories=len(training),
                    conditional_training_loss=float(loss.detach().cpu()) if covered_count else None,
                    covered_targets_in_batch=covered_count, elapsed_seconds=time.perf_counter() - started)
            if covered_count and (global_step == 1 or global_step % monitor_every == 0):
                monitor(epoch)
                recorder.event("phase", phase="training", epoch=epoch, global_step=global_step,
                               total_steps=total_steps, stories_completed=completed, total_stories=len(training))
        recorder.event("phase", phase="validation", scope="full_validation", epoch=epoch,
                       global_step=global_step, total_steps=total_steps)
        validation_started = time.perf_counter()
        metrics = evaluate(model, validation, proposals, batch_size)
        result = recorder.validation("full_validation", validation, metrics, global_step, epoch,
                                      evaluation_seconds=time.perf_counter() - validation_started,
                                      evaluation_method="current_model_forward")
        save_checkpoint(output / "partial.pt", model, protocol, global_step, epoch, "rolling_partial", result)
        if selection_improves(result, best):
            best = result
            save_checkpoint(output / "best.pt", model, protocol, global_step, epoch, "best_full_validation", best)
            write_json(output / "selection.json", best)
        write_json(output / ("epoch-%d.json" % epoch), {"epoch": epoch, "train": train_metrics.summary(),
                                                       "validation": result, "selected": best})
    brain.verify_frozen()
    final = {"schema": 1, "kind": "action_selection", "arm": ARM, "top_k": top_k, "selected": best,
             "latest_full_validation": result, "global_step": global_step, "test_evaluated": False,
             "elapsed_seconds": time.perf_counter() - started, "timestamp_utc": utc_now()}
    write_json(output / "metrics.json", final)
    recorder.event("phase", phase="completed", epoch=epochs, global_step=global_step,
                   total_steps=total_steps, selected_accuracy=best["accuracy"], test_evaluated=False)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "graph", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="mps")
    for name, default in (("epochs", 1), ("batch-size", 8), ("seed", 0), ("monitor-stories", 128),
                          ("monitor-every", 64), ("progress-every", 8), ("calibration-stories", 32),
                          ("threads", 4), ("internal-steps", 8), ("top-k", 10)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", .001), ("gradient-clip", 1.), ("global-scale", .002),
                          ("leak", .5), ("input-high", .02)):
        parser.add_argument("--" + name, type=float, default=default)
    arguments = vars(parser.parse_args())
    arguments["dataset_path"] = arguments.pop("dataset")
    train(**arguments)


if __name__ == "__main__":
    main()
