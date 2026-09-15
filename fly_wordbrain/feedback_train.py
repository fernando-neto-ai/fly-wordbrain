"""Learn delayed, within-window plasticity on the unchanged full connectome.

One target per eight lexical words. The seven historical observations write
only temporary edge state; held-out target eight is used only by the loss.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .action_candidates import ActionCandidates
from .action_train import ActionMetrics, utc_now, write_json, validate_training_validation
from .feedback_data import make_windows, collate_feedback, windows_identity
from .feedback_model import FeedbackActionBrain, FeedbackSelector
from .pair_decoder import file_sha256

KIND, ARM = "feedback_action", "feedback_fast_weights"
SOURCES = ("feedback_train.py", "feedback_model.py", "feedback_data.py", "action_train.py",
           "action_candidates.py", "action_model.py", "plastic_brain.py", "metal_sparse.py",
           "brain.py", "pair_brain.py", "pair_decoder.py")


def source_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCES}


def final_batch(batch):
    return SimpleNamespace(candidates=batch.candidates[:, 7], probabilities=batch.probabilities[:, 7],
                           targets=batch.targets, active=torch.ones_like(batch.targets, dtype=torch.bool))


def window_loss(logits, batch):
    matches = batch.candidates[:, 7] == batch.targets[:, None]
    covered = matches.any(-1)
    count = int(covered.sum().detach().cpu())
    if not count:
        return logits.sum() * 0, 0
    rank = matches.to(torch.long).argmax(-1)
    return F.cross_entropy(logits[covered], rank[covered]), count


def _norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(sum(values)).cpu()) if values else 0.


def _identity(windows, dataset_sha, split="val"):
    identity = windows_identity(windows, dataset_sha, split)
    return {"dataset_sha256": dataset_sha, "subset_sha256": identity["subset_sha256"],
            "story_ids": list(dict.fromkeys(w.story_id for w in windows)),
            "window_count": len(windows), "story_count": len({w.story_id for w in windows})}


def calibrate(model, windows, proposals, batch_size, progress=None):
    """Training-only LOO moments; fast state disabled during calibration."""
    brain = model.brain
    feature_values, pre_values, post_values = [], [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = collate_feedback(windows[start:start + batch_size], proposals,
                                     device=brain.device, exclude_own_story=True)
            state = brain.initial_state(len(batch.targets))
            for position in range(8):
                features, state = brain.predict(batch.previous[:, position], batch.current[:, position],
                    batch.candidates[:, position], batch.probabilities[:, position], state, plasticity=False)
                pre_values.append(state.h[:, brain.candidate_pre].cpu().numpy())
                post_values.append(state.h[:, brain.candidate_post].cpu().numpy())
                if position < 7:
                    # Advance the causal phase even in the disabled-write arm.
                    state = brain.observe(state, batch.candidates[:, position], batch.probabilities[:, position],
                                          batch.observed_ids[:, position], position + 1)
            feature_values.append(features.cpu().numpy())
            if progress:
                progress(min(start + batch_size, len(windows)), len(windows))
    values = np.concatenate(feature_values).astype(np.float64)
    raw_std = values.std(0)
    std_floor = max(float(raw_std.max()) * 1e-4, 1e-6)
    std = np.maximum(raw_std, std_floor).astype(np.float32)
    mean = values.mean(0).astype(np.float32)
    scales = []
    for chunks in (pre_values, post_values):
        a = np.concatenate(chunks).astype(np.float64)
        scales.append(np.maximum(np.sqrt(np.mean(a * a, axis=0)), 1e-6).astype(np.float32))
    brain.set_activity_scales(*scales)
    model.feature_mean.copy_(torch.as_tensor(mean, device=brain.device))
    model.feature_std.copy_(torch.as_tensor(std, device=brain.device))
    model.features_calibrated = True
    if not all(np.isfinite(x).all() for x in (values, mean, std, *scales)):
        raise RuntimeError("Nonfinite training calibration")
    return {"schema": 1, "kind": KIND, "training_only": True, "window_count": len(windows),
            "proposal_exclusion": "entire current training story", "plasticity_enabled": False,
            "varying_features": int((raw_std > 1e-10).sum()), "feature_std_floor": std_floor,
            "feature_std_max": float(raw_std.max()), "pre_scale_min": float(scales[0].min()),
            "pre_scale_max": float(scales[0].max()), "post_scale_min": float(scales[1].min()),
            "post_scale_max": float(scales[1].max()), "timestamp_utc": utc_now()}


def preflight(model, windows, proposals, batch_size):
    """Actual full-graph forward, backward, optimizer and delayed-feedback checks.

    Restore every slow parameter afterward; these train-only probe steps never
    become an unreported warm start. Calibration buffers are retained.
    """
    batch = collate_feedback(windows[:batch_size], proposals, device=model.brain.device,
                             exclude_own_story=True)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    before = model.brain.verify_frozen()
    began = time.perf_counter()
    with torch.no_grad():
        feat, state = model.features(batch, plasticity=True)
        repeat, _ = model.features(batch, plasticity=True)
        disabled, disabled_state = model.features(batch, plasticity=False)
        # A counterfactual write-only intervention holds every sensory input
        # fixed while permuting observed feedback between examples. Normal
        # model inputs rightly reject inconsistent observed/context fields.
        shuffled_feedback = batch.observed_ids.roll(1, dims=0)
        replacement = torch.where(batch.observed_ids == 4, 5, 4)
        shuffled_feedback = torch.where(shuffled_feedback == batch.observed_ids, replacement, shuffled_feedback)
        changed_state = model.initial_state(len(batch.targets))
        for position in range(8):
            changed_feat, changed_state = model.brain.predict(
                batch.previous[:, position], batch.current[:, position], batch.candidates[:, position],
                batch.probabilities[:, position], changed_state, plasticity=True)
            if position < 7:
                changed_state = model.brain.observe(changed_state, batch.candidates[:, position],
                    batch.probabilities[:, position], shuffled_feedback[:, position], position + 1)
        target_changed = replace(batch, targets=(batch.targets + 1) % model.brain.vocab_size)
        hidden_target_feat, _ = model.features(target_changed, plasticity=True)
        noise = float(((feat - repeat).abs() / model.feature_std).max().cpu())
        effect = float(((feat - disabled).abs() / model.feature_std).max().cpu())
        feedback_effect = float(((feat - changed_feat).abs() / model.feature_std).max().cpu())
        fast_effect = float((state.fast - changed_state.fast).abs().max().cpu())
        baseline_exact = torch.equal(model(batch), batch.probabilities[:, 7].log())
        hidden_target_effect = float(((feat - hidden_target_feat).abs() / model.feature_std).max().cpu())
        # Also omit labels entirely: the forward path must not access them.
        inputs_only = SimpleNamespace(**{name: getattr(batch, name) for name in
            ("previous", "current", "candidates", "probabilities", "observed_ids")})
        model(inputs_only)
        target_hidden = hidden_target_effect <= max(10 * noise, 1e-5)
        fast_max = float(state.fast.abs().max().cpu())
        scales = model.brain.effective_candidate_weights(state.fast) / model.brain.candidate_weight
        scales = scales[:, model.brain.candidate_weight != 0]
        sign_bounded = bool(torch.all((scales >= .5) & (scales <= 2)).cpu())
        disabled_zero = bool(torch.all(disabled_state.fast == 0).cpu())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    steps, rule_norms, head_norms, elapsed = [], [], [], []
    for _ in range(3):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss, covered = window_loss(logits, batch)
        if not covered:
            raise RuntimeError("Preflight batch has no covered targets")
        loss.backward()
        head_norms.append(_norm(model.readout.parameters()))
        rule_norms.append(_norm(p for p in model.brain.parameters() if p.requires_grad))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        steps.append(float(loss.detach().cpu()))
        if model.brain.device.type == "mps":
            torch.mps.synchronize()
        elapsed.append(time.perf_counter() - start)
    rule_delta = max(float((p.detach() - initial[name]).abs().max().cpu())
                     for name, p in model.named_parameters() if name.startswith("brain.") and p.requires_grad)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(initial[name])
    model.zero_grad(set_to_none=True)
    after = model.brain.verify_frozen()
    checks = {
        "eighth_target_hidden": target_hidden, "initial_count_prior_exact": baseline_exact,
        "fast_memory_written": fast_max > 0, "disabled_fast_state_zero": disabled_zero,
        "feedback_changes_fast_memory": fast_effect > 1e-8,
        "fast_readout_visible": effect > max(10 * noise, 1e-5),
        "feedback_readout_visible": feedback_effect > max(10 * noise, 1e-5),
        "head_gradients_nonzero_finite": all(math.isfinite(x) and x > 0 for x in head_norms),
        "rule_gradients_nonzero_finite_after_head_warmup": all(math.isfinite(x) and x > 0 for x in rule_norms[1:]),
        "slow_rules_updated": rule_delta > 0, "frozen_graph_preserved": before == after,
        "effective_weight_signs_and_bounds": sign_bounded,
        "parameters_restored": all(torch.equal(p, initial[n]) for n, p in model.named_parameters()),
    }
    return {"schema": 1, "kind": KIND, "passed": all(checks.values()), "checks": checks,
            "training_only": True, "batch_size": len(batch.targets), "covered_targets": covered,
            "graph_fingerprint": before, "feature_replay_noise": noise,
            "hidden_target_feature_difference": hidden_target_effect,
            "fast_feature_effect": effect, "observed_feedback_feature_effect": feedback_effect,
            "observed_feedback_fast_effect": fast_effect, "fast_abs_max": fast_max,
            "head_gradient_norms": head_norms, "rule_gradient_norms": rule_norms,
            "rule_parameter_max_delta": rule_delta, "probe_losses": steps,
            "optimizer_step_seconds": elapsed, "elapsed_seconds": time.perf_counter() - began,
            "mps_allocated_bytes": torch.mps.current_allocated_memory() if model.brain.device.type == "mps" else None,
            "timestamp_utc": utc_now()}


def evaluate(model, windows, proposals, batch_size=16, baseline_only=False, progress=None):
    main, disabled = ActionMetrics(), ActionMetrics()
    overrides, changed_by_fast, logit_difference = 0, 0, 0.
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    if model is not None:
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = collate_feedback(windows[start:start + batch_size], proposals,
                                         device="cpu" if baseline_only else model.brain.device)
                logits = batch.probabilities[:, 7].log() if baseline_only else model(batch)
                ablated = logits if baseline_only else model(batch, plasticity=False)
                final = final_batch(batch)
                main.update(logits, final)
                disabled.update(ablated, final)
                overrides += int((logits.argmax(-1) != final.probabilities.argmax(-1)).sum().cpu())
                changed_by_fast += int((logits.argmax(-1) != ablated.argmax(-1)).sum().cpu())
                logit_difference = max(logit_difference, float((logits - ablated).abs().max().cpu()))
                if progress:
                    progress(min(start + batch_size, len(windows)), len(windows))
    finally:
        for module, mode in modes:
            module.training = mode
    result, control = main.summary(), disabled.summary()
    return {**result, "overrides": overrides, "override_rate": overrides / len(windows),
            "disabled_fast_accuracy": control["accuracy"],
            "disabled_fast_conditional_cross_entropy": control["conditional_cross_entropy"],
            "fast_changed_predictions": changed_by_fast, "fast_max_logit_difference": logit_difference,
            "disabled_fast_method": "Same trained head; all temporary writes and reads disabled; not a separately trained control"}


class Recorder:
    def __init__(self, output, dataset_sha, total_windows, total_steps):
        self.output, self.dataset_sha = Path(output), dataset_sha
        self.total_windows, self.total_steps = total_windows, total_steps
        self.history = []

    def event(self, event, **fields):
        row = {"schema": 1, "kind": KIND, "arm": ARM, "top_k": 10, "window_words": 8,
               "observed_words": 7, "event": event, "timestamp_utc": utc_now(), **fields}
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        return row

    def progress(self, phase, step=0, epoch=0, completed=0, **fields):
        row = self.event("progress", phase=phase, global_step=step, epoch=epoch,
                         windows_completed=completed, total_windows=self.total_windows,
                         total_steps=self.total_steps, status="completed" if phase == "completed" else "running", **fields)
        write_json(self.output / "progress.json", row)

    def validation(self, scope, windows, metrics, step, epoch, **fields):
        row = self.event("validation_snapshot", scope=scope, global_step=step, epoch=epoch,
                         phase="validation", **_identity(windows, self.dataset_sha), **metrics,
                         selection_eligible=scope == "full_validation", **fields)
        self.history.append(row)
        destination = self.output / "validation.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in self.history))
        temporary.replace(destination)
        return row


def save_checkpoint(path, model, protocol, step, epoch, role, metrics=None):
    parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    payload = {"schema": 1, "kind": KIND, "arm": ARM, "checkpoint_role": role,
               "global_step": step, "epoch": epoch, "parameters": parameters,
               "feature_mean": model.feature_mean.detach().cpu(), "feature_std": model.feature_std.detach().cpu(),
               "pre_scale": model.brain.pre_scale.detach().cpu(), "post_scale": model.brain.post_scale.detach().cpu(),
               "protocol": protocol, "validation": metrics, "inference_only": True,
               "training_resume_supported": False, "selection_eligible": role == "best_full_validation"}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha256(path)


def train(dataset_path, graph, output, *, device="mps", epochs=1, batch_size=16, lr=.001,
          seed=0, monitor_stories=128, monitor_every=128, progress_every=8,
          calibration_windows=256, threads=4, global_scale=.002, internal_steps=8,
          strict_full_graph=True, probe_only=False):
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("monitor_stories", monitor_stories),
                        ("monitor_every", monitor_every), ("progress_every", progress_every),
                        ("calibration_windows", calibration_windows), ("threads", threads)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    output = Path(output)
    allowed = {"launch.json", "process-status.json", "stdout.log"}
    if output.exists() and any(p.name not in allowed for p in output.iterdir()):
        raise ValueError("Refusing prior experiment output")
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(dataset_path).read_text())
    validate_training_validation(data, 10)
    train_stories, val_stories = data["splits"]["train"], data["splits"]["val"]
    if monitor_stories > len(val_stories):
        raise ValueError("Monitor exceeds validation split")
    training, validation = make_windows(train_stories), make_windows(val_stories)
    monitor = make_windows(val_stories[:monitor_stories])
    if not min(len(training), len(validation), len(monitor)):
        raise ValueError("Every consumed split must contain complete eight-word windows")
    calibration_rows = training[:calibration_windows]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)
    dataset_sha = file_sha256(dataset_path)
    graph_file = Path(graph) / "graph.npz" if Path(graph).is_dir() else Path(graph)
    total_steps = math.ceil(len(training) / batch_size) * epochs
    recorder = Recorder(output, dataset_sha, len(training) * epochs, total_steps)
    protocol = {"schema": 1, "kind": KIND, "arm": ARM, "top_k": 10, "window_words": 8, "observed_words": 7,
        "timestamp_utc": utc_now(), "dataset_sha256": dataset_sha, "dataset_path": str(Path(dataset_path).resolve()),
        "graph_sha256": file_sha256(graph_file), "graph_path": str(graph_file.resolve()),
        "source_sha256": source_hashes(), "train_windows": len(training), "validation_windows": len(validation),
        "monitor_windows": len(monitor), "train_story_count": len(train_stories), "validation_story_count": len(val_stories),
        "batch_size": batch_size, "epochs": epochs, "lr": lr, "seed": seed, "device": device,
        "config": {"global_scale": global_scale, "internal_steps": internal_steps, "leak": .5,
                   "input_high": .02, "batch_size": batch_size, "epochs": epochs, "lr": lr,
                   "monitor_every": monitor_every, "calibration_windows": len(calibration_rows), "threads": threads},
        "vocabulary": data["vocabulary"], "training_resume_supported": False,
        "window_contract": "Nonoverlapping eight lexical words; tails dropped; no story crossing; BOS,BOS count context and zero neural/fast/eligibility state at each window start",
        "training_proposals": "All training counts minus the entire current story; no target-dependent candidate injection",
        "validation_proposals": "Training counts only; validation/test never fitted",
        "objective": "Eighth-word top-10 conditional cross entropy; targets absent from top10 remain all-window accuracy misses",
        "historical_feedback": "After each of the seven actual words: candidate-rank error with OTHER mass, fixed word identity code, position; prediction-time eligibility",
        "checkpoint_selection": "Strictly higher full-validation window accuracy only; baseline eligible; ties keep earlier",
        "comparison": "New eight-word-window population; do not compare directly with older all-position curves",
        "test_evaluated": False, "calibration_subset": _identity(calibration_rows, dataset_sha, "train"),
        "monitor_subset": _identity(monitor, dataset_sha)}
    write_json(output / "protocol.json", protocol)
    recorder.progress("building_proposals")
    proposals = ActionCandidates(train_stories, len(data["vocabulary"]), top_k=10)
    recorder.progress("loading_brain")
    brain = FeedbackActionBrain(graph, len(data["vocabulary"]), global_scale=global_scale, top_k=10,
        internal_steps=internal_steps, leak=.5, input_high=.02, seed=seed, device=device, strict_full_graph=strict_full_graph)
    model = FeedbackSelector(brain)
    recorder.progress("calibration")
    calibration = calibrate(model, calibration_rows, proposals, batch_size,
        progress=lambda done, total: recorder.progress("calibration", phase_windows_completed=done, phase_total_windows=total))
    write_json(output / "calibration.json", {**calibration, **_identity(calibration_rows, dataset_sha, "train")})
    protocol["model"] = {"neurons": brain.neurons, "edges": brain.edges, "candidate_edges": brain.candidates,
        "trainable_parameters": model.trainable_parameter_count(), "metadata": model.metadata()}
    protocol["trainable_parameters"] = model.trainable_parameter_count()
    protocol["brain_trainable_parameters"] = brain.trainable_parameter_count()
    protocol["brain"] = brain.metadata()
    write_json(output / "protocol.json", protocol)
    recorder.progress("preflight")
    probe = preflight(model, calibration_rows, proposals, batch_size)
    probe.update(dataset_sha256=dataset_sha, graph_sha256=protocol["graph_sha256"], source_sha256=source_hashes())
    write_json(output / "preflight.json", probe)
    if not probe["passed"]:
        recorder.event("preflight_failed", checks=probe["checks"])
        raise RuntimeError("Full-graph preflight failed: " + ", ".join(k for k, v in probe["checks"].items() if not v))
    protocol["preflight_sha256"] = file_sha256(output / "preflight.json")
    protocol["preflight"] = {k: probe[k] for k in ("passed", "fast_feature_effect", "observed_feedback_feature_effect", "rule_gradient_norms", "optimizer_step_seconds")}
    write_json(output / "protocol.json", protocol)
    if probe_only:
        recorder.progress("completed", probe_only=True)
        return {"probe_only": True, **probe}
    recorder.progress("baseline_validation")
    best = None
    for scope, rows in (("full_validation", validation), ("monitor_subset", monitor)):
        metrics = evaluate(None, rows, proposals, batch_size=256, baseline_only=True)
        snapshot = recorder.validation(scope, rows, metrics, 0, 0, evaluation_method="count-only; exact zero-head equivalence verified in preflight")
        if scope == "full_validation":
            best = snapshot
            save_checkpoint(output / "best.pt", model, protocol, 0, 0, "best_full_validation", best)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    step, optimizer_steps, completed, start_time = 0, 0, 0, time.perf_counter()
    last_loss, last_rule_norm, last_head_norm = None, None, None
    for epoch in range(1, epochs + 1):
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            windows = [training[i] for i in order[start:start + batch_size]]
            batch = collate_feedback(windows, proposals, device=brain.device, exclude_own_story=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss, covered = window_loss(logits, batch)
            if covered:
                loss.backward()
                last_rule_norm = _norm(p for p in brain.parameters() if p.requires_grad)
                last_head_norm = _norm(model.readout.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                last_loss = float(loss.detach().cpu())
            else:
                last_loss, last_rule_norm, last_head_norm = None, None, None
            step += 1; completed += len(windows)
            if step == 1 or step % progress_every == 0:
                recorder.progress("training", step, epoch, completed, conditional_training_loss=last_loss,
                    covered_targets=covered, rule_gradient_norm=last_rule_norm, head_gradient_norm=last_head_norm,
                    optimizer_steps=optimizer_steps,
                    elapsed_seconds=time.perf_counter() - start_time,
                    optimizer_step_seconds=probe["optimizer_step_seconds"][-1])
            if step in (1, 32) or step % monitor_every == 0:
                recorder.progress("validation", step, epoch, completed)
                metrics = evaluate(model, monitor, proposals, batch_size,
                    progress=lambda done, total: recorder.progress("validation", step, epoch, completed,
                        phase_windows_completed=done, phase_total_windows=total) if done == total or done % (batch_size * 16) == 0 else None)
                snapshot = recorder.validation("monitor_subset", monitor, metrics, step, epoch)
                save_checkpoint(output / "partial.pt", model, protocol, step, epoch, "monitor_only", snapshot)
                recorder.progress("training", step, epoch, completed, rule_gradient_norm=last_rule_norm,
                                  fast_effect=metrics["fast_max_logit_difference"])
        recorder.progress("full_validation", step, epoch, completed)
        metrics = evaluate(model, validation, proposals, batch_size,
            progress=lambda done, total: recorder.progress("full_validation", step, epoch, completed,
                phase_windows_completed=done, phase_total_windows=total) if done == total or done % (batch_size * 16) == 0 else None)
        snapshot = recorder.validation("full_validation", validation, metrics, step, epoch)
        if snapshot["accuracy"] > best["accuracy"]:
            best = snapshot
            save_checkpoint(output / "best.pt", model, protocol, step, epoch, "best_full_validation", best)
        save_checkpoint(output / "last.pt", model, protocol, step, epoch, "last", snapshot)
    brain.verify_frozen()
    result = {"schema": 1, "kind": KIND, "arm": ARM, "global_step": step, "train_windows": len(training),
              "optimizer_steps": optimizer_steps,
              "validation_windows": len(validation), "best_validation": best, "last_validation": snapshot,
              "test_evaluated": False, "elapsed_seconds": time.perf_counter() - start_time}
    write_json(output / "metrics.json", result)
    recorder.progress("completed", step, epochs, completed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "graph", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="mps")
    for name, default in (("epochs", 1), ("batch-size", 16), ("seed", 0), ("monitor-stories", 128),
                          ("monitor-every", 128), ("progress-every", 8), ("calibration-windows", 256),
                          ("threads", 4), ("internal-steps", 8)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--global-scale", type=float, default=.002)
    parser.add_argument("--probe-only", action="store_true")
    args = vars(parser.parse_args())
    args["dataset_path"] = args.pop("dataset")
    train(**args)


if __name__ == "__main__":
    main()
